#!/usr/bin/env python3
"""Prepare a standard GaVS scene without changing any GaVS source files.

Place this NEW file beside train.py. See PREPARE_VIDEO.md for installation and
Linux examples. Each --steps value runs a separate function. Only generated
data is written; no GaVS source/configuration files are changed. Requires a real
COLMAP reconstruction and at least 64 registered frames. Depth and optical flow
are computed by GaVS itself during training, rather than by this script.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
STEPS = ("extract", "masks", "reconstruct", "geometry", "metadata",
         "inpaint", "inpaint-masks", "assemble", "validate")


def colmap_io():
    """Reuse the repository's binary readers/writers without importing torch."""
    path = ROOT / "datasets" / "colmap_utils.py"
    spec = importlib.util.spec_from_file_location("gavs_colmap_io", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cv2_module():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install ONE OpenCV package at version 4.10.0.84, e.g. "
                           "opencv-contrib-python==4.10.0.84. See PREPARE_VIDEO.md.") from exc
    return cv2


def run(command):
    command = [str(part) for part in command]
    print("$ " + shlex.join(command), flush=True)
    subprocess.run(command, check=True)


def require_program(program):
    if not shutil.which(program) and not Path(program).is_file():
        raise RuntimeError(f"Executable not found: {program}. Install it or provide its absolute path.")


def write_json(path, data, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise RuntimeError(f"Already exists: {path}; use a fresh output folder or --overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def output_file(path, overwrite):
    path = Path(path)
    if path.exists() and not overwrite:
        raise RuntimeError(f"Already exists: {path}; use --overwrite only for pipeline outputs.")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def frames(directory):
    files = sorted(Path(directory).glob("*.png"))
    if not files:
        raise RuntimeError(f"No PNG frames in {directory}")
    if [p.name for p in files] != [f"{i:05d}.png" for i in range(len(files))]:
        raise RuntimeError(f"Frames must be contiguous 00000.png, 00001.png, ...: {directory}")
    return files


def read_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).copy()


def read_mask(path, size):
    with Image.open(path) as image:
        if image.size != size:
            raise RuntimeError(f"Mask size mismatch: {path}: {image.size}, expected {size}")
        # GaVS treats white as dynamic foreground and black as static background.
        value = np.asarray(image.convert("RGB")).max(axis=2)
        return (value > 127).astype(np.uint8) * 255


def save_array(path, array, overwrite):
    Image.fromarray(array).save(output_file(path, overwrite))


def extract_frames(args):
    """Extract a constant-FPS, centered, fixed-size sequence starting at zero."""
    if not args.video or not args.video.is_file():
        raise RuntimeError("--video must point to an existing input video for the extract step.")
    require_program(args.ffmpeg)
    args.frames_dir.mkdir(parents=True, exist_ok=True)
    if any(args.frames_dir.glob("*.png")):
        raise RuntimeError("Extraction requires an empty frames directory (also with --overwrite). "
                           "Use a fresh directory to avoid leftover frames from another video.")
    filters = (f"fps={args.fps},scale=864:480:force_original_aspect_ratio=increase,"
               "crop=864:480,setsar=1")
    command = [args.ffmpeg, "-nostdin", "-n"]
    if args.start:
        command += ["-ss", args.start]
    command += ["-i", args.video]
    if args.duration is not None:
        command += ["-t", str(args.duration)]
    command += ["-an", "-vf", filters, "-start_number", "0", args.frames_dir / "%05d.png"]
    run(command)
    images = frames(args.frames_dir)
    if len(images) < 64:
        raise RuntimeError(f"Only {len(images)} frames extracted. Unmodified GaVS samples 64 "
                           "frames for depth scale; supply at least 64 frames.")
    write_json(args.work_dir / "extraction.json", {"video": str(args.video),
               "fps": args.fps, "frames": len(images), "start": args.start,
               "duration": args.duration, "size": [864, 480]}, args.overwrite)


def estimate_homography(source, target):
    """Estimate source -> target mapping from robust background feature matches."""
    cv2 = cv2_module()
    orb = cv2.ORB_create(nfeatures=5000)
    ks, ds = orb.detectAndCompute(cv2.cvtColor(source, cv2.COLOR_RGB2GRAY), None)
    kt, dt = orb.detectAndCompute(cv2.cvtColor(target, cv2.COLOR_RGB2GRAY), None)
    if ds is None or dt is None:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(ds, dt, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    if len(good) < 12:
        return None
    src = np.float32([ks[m.queryIdx].pt for m in good])
    dst = np.float32([kt[m.trainIdx].pt for m in good])
    matrix, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if matrix is None or inliers is None or inliers.sum() < 12 or not np.isfinite(matrix).all():
        return None
    return matrix


def generate_dynamic_masks(args):
    """Produce foreground masks via motion, Mask R-CNN, static assumption or import."""
    files = frames(args.frames_dir)
    cv2 = cv2_module()
    model = None
    if args.mask_method == "maskrcnn":
        try:
            import torch
            from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
        except ImportError as exc:
            raise RuntimeError("maskrcnn needs a matching torch/torchvision installation. "
                               "Use the working GaVS environment or --mask-method motion.") from exc
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for Mask R-CNN. Use --device cpu.")
        model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).eval().to(args.device)
    for i, path in enumerate(files):
        target = read_rgb(path)
        h, w = target.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        if args.mask_method == "import":
            if args.mask_source_dir is None:
                raise RuntimeError("--mask-source-dir is required for --mask-method import")
            mask = read_mask(args.mask_source_dir / path.name, (w, h))
        elif args.mask_method == "maskrcnn":
            with torch.no_grad():
                result = model([torch.from_numpy(target).permute(2, 0, 1).float().to(args.device) / 255])[0]
            select = (result["scores"] >= args.mask_score)
            labels = result["labels"].cpu().numpy()
            select &= torch.tensor(np.isin(labels, args.dynamic_classes), device=select.device)
            if select.any():
                mask = (result["masks"][select, 0].amax(dim=0).cpu().numpy() > 0.5).astype(np.uint8) * 255
        elif args.mask_method == "motion":
            candidates = []
            for j in (i - args.motion_stride, i + args.motion_stride):
                if j < 0 or j >= len(files):
                    continue
                source = read_rgb(files[j])
                matrix = estimate_homography(source, target)
                if matrix is None:
                    continue
                warped = cv2.warpPerspective(source, matrix, (w, h))
                valid = cv2.warpPerspective(np.ones((h, w), np.uint8), matrix, (w, h), flags=cv2.INTER_NEAREST)
                valid = cv2.erode(valid, np.ones((9, 9), np.uint8))
                a = cv2.GaussianBlur(cv2.cvtColor(target, cv2.COLOR_RGB2GRAY), (5, 5), 0).astype(float)
                b = cv2.GaussianBlur(cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY), (5, 5), 0).astype(float)
                delta = a - b
                if not valid.any():
                    continue
                delta -= np.median(delta[valid > 0])
                candidates.append((np.abs(delta) > args.motion_threshold) & (valid > 0))
            if not candidates:
                raise RuntimeError(f"Cannot estimate background motion for {path.name}. "
                                   "Try --mask-method maskrcnn, import, or static for a genuinely static scene.")
            # A moving object can occupy different pixels in the two temporal
            # comparisons; union the residuals so its full footprint is kept.
            mask = np.logical_or.reduce(candidates).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        if args.mask_dilate:
            kernel = np.ones((2 * args.mask_dilate + 1,) * 2, np.uint8)
            mask = cv2.dilate(mask, kernel)
        save_array(args.masks_dir / path.name, mask, args.overwrite)
        print(f"masks {i + 1}/{len(files)}: {path.name}", flush=True)


def program_options(program, subcommand, *names):
    result = subprocess.run([program, subcommand, "-h"], capture_output=True,
                            text=True, errors="replace", check=True)
    help_text = result.stdout + result.stderr
    for name in names:
        if name in help_text:
            return name
    raise RuntimeError(f"Cannot detect an option for {subcommand}. Check `{program} {subcommand} -h`.")


def colmap_options(args, subcommand, old_name, new_name):
    """COLMAP 3.x/4.x use different option names for GPU feature extraction."""
    return program_options(args.colmap, subcommand, new_name, old_name)


def reconstruct_scene(args):
    """Compute camera intrinsics, poses and a sparse point cloud using COLMAP/GLOMAP."""
    require_program(args.colmap)
    files = frames(args.frames_dir)
    args.sparse_dir.mkdir(parents=True, exist_ok=True)
    database = args.sparse_dir / "database.db"
    if database.exists() or any(args.sparse_dir.glob("*/images.bin")):
        raise RuntimeError("Reconstruction requires a fresh --sparse-dir; existing databases are not reused.")
    sfm_masks = args.sparse_dir / "static_masks"
    for path in files:
        mask = read_mask(args.masks_dir / path.name, (864, 480))
        # COLMAP uses white=keep, opposite to GaVS. COLMAP appends `.png` to
        # the complete image name, so `00000.png` maps to `00000.png.png`.
        save_array(sfm_masks / (path.name + ".png"), 255 - mask, args.overwrite)
    feature_gpu = colmap_options(args, "feature_extractor", "SiftExtraction.use_gpu", "FeatureExtraction.use_gpu")
    match_gpu = colmap_options(args, "sequential_matcher", "SiftMatching.use_gpu", "FeatureMatching.use_gpu")
    command = [args.colmap, "feature_extractor", "--database_path", database,
               "--image_path", args.frames_dir, "--ImageReader.single_camera", "1",
               "--ImageReader.camera_model", args.camera_model, "--ImageReader.mask_path", sfm_masks,
               f"--{feature_gpu}", str(int(args.sfm_gpu))]
    if args.camera_params:
        command += ["--ImageReader.camera_params", args.camera_params]
    run(command)
    run([args.colmap, "sequential_matcher", "--database_path", database,
         "--SequentialMatching.overlap", str(args.match_overlap), f"--{match_gpu}", str(int(args.sfm_gpu))])
    if args.mapper == "glomap":
        require_program(args.glomap)
        command = [args.glomap, "mapper"]
    else:
        command = [args.colmap, "mapper"]
    mapper_image_option = program_options(args.colmap if args.mapper == "colmap" else args.glomap,
                                          "mapper", "--input_path", "--image_path")
    run(command + ["--database_path", database, mapper_image_option, args.frames_dir,
                   "--output_path", args.sparse_dir])


def load_raw_model(args):
    io = colmap_io()
    candidates = [args.sparse_dir] + sorted(p.parent for p in args.sparse_dir.glob("*/images.bin"))
    models = [(io.read_model(str(p), ".bin"), p) for p in candidates if (p / "images.bin").exists()]
    if not models:
        raise RuntimeError("No reconstructed COLMAP model found. Reconstruction may have failed.")
    model, path = max(models, key=lambda item: len(item[0][1]))
    print(f"Using reconstruction: {path}", flush=True)
    return io, model


def camera_matrix(camera):
    """Translate supported COLMAP cameras into OpenCV intrinsics/distortion."""
    p = camera.params
    if camera.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        f, cx, cy = p[:3]
        fx = fy = f
        distortion = np.zeros(5)
        distortion[:len(p) - 3] = p[3:]
    elif camera.model in ("PINHOLE", "OPENCV"):
        fx, fy, cx, cy = p[:4]
        distortion = np.zeros(5)
        if camera.model == "OPENCV":
            distortion[:4] = p[4:8]
    else:
        raise RuntimeError(f"Unsupported camera model {camera.model}; use OPENCV/SIMPLE_RADIAL/PINHOLE.")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]], dtype=float)
    return K, distortion


def undistortion_map(K, distortion):
    """Center the principal point and minimally zoom to avoid invalid borders."""
    cv2 = cv2_module()
    if not np.isfinite(K).all() or not np.isfinite(distortion).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise RuntimeError("Invalid reconstructed camera intrinsics.")

    def candidate(zoom):
        target = K.copy()
        target[0, 0] *= zoom
        target[1, 1] *= zoom
        target[0, 2], target[1, 2] = 432., 240.
        mx, my = cv2.initUndistortRectifyMap(K, distortion, None, target,
                                           (864, 480), cv2.CV_32FC1)
        valid = (np.isfinite(mx) & np.isfinite(my) & (mx >= -1e-3) &
                 (mx <= 863 + 1e-3) & (my >= -1e-3) & (my <= 479 + 1e-3))
        return target, mx, my, bool(valid.all())

    zoom = 1.
    target, mx, my, valid = candidate(zoom)
    if not valid:
        low, high = 1., 1.05
        while high < 4. and not candidate(high)[3]:
            high = min(4., high * 1.1)
        if not candidate(high)[3]:
            raise RuntimeError("Cannot undistort this camera without holes at <=4x zoom. "
                               "Use a calibrated lens model or lens-corrected footage.")
        for _ in range(20):
            mid = (low + high) / 2
            if candidate(mid)[3]:
                high = mid
            else:
                low = mid
        zoom = high
        target, mx, my, valid = candidate(zoom)
    if zoom > 1.001:
        print(f"Undistortion requires {zoom:.3f}x zoom; field of view will be reduced.", flush=True)
    # Clamp only numerical edge noise; genuine out-of-image coordinates triggered zoom.
    return target, np.clip(mx, 0, 863), np.clip(my, 0, 479), zoom


def prepare_geometry(args):
    """Undistort images/masks, center the principal point, and reindex sparse observations."""
    cv2 = cv2_module()
    io, (cameras, images, points) = load_raw_model(args)
    files = frames(args.frames_dir)
    by_name = {image.name: image for image in images.values()}
    missing = [p.name for p in files if p.name not in by_name]
    if missing:
        raise RuntimeError(f"COLMAP did not register {len(missing)}/{len(files)} frames: {missing[:10]}. "
                           "No poses will be fabricated. Try better overlap/texture, fewer dynamic masks, "
                           "or a shorter sequence.")
    if len(cameras) != 1 or len({im.camera_id for im in images.values()}) != 1:
        raise RuntimeError("GaVS needs one shared camera. Use single-camera reconstruction.")
    if not points:
        raise RuntimeError("Reconstruction has no sparse points.")
    if len(files) < 64:
        raise RuntimeError("Unmodified GaVS requires at least 64 registered frames.")
    camera = next(iter(cameras.values()))
    if (camera.width, camera.height) != (864, 480):
        raise RuntimeError("This pipeline uses 864x480 frames to match GaVS's default aspect ratio.")
    K, distortion = camera_matrix(camera)
    target_K, mapx, mapy, zoom = undistortion_map(K, distortion)
    image_id_map = {by_name[p.name].id: i + 1 for i, p in enumerate(files)}
    point_id_map = {old: i + 1 for i, old in enumerate(sorted(points))}
    new_images = {}
    kept_observations = {}
    for i, path in enumerate(files):
        rgb = read_rgb(path)
        mask = read_mask(args.masks_dir / path.name, (864, 480))
        save_array(args.scene_images / path.name, cv2.remap(rgb, mapx, mapy, cv2.INTER_LINEAR), args.overwrite)
        save_array(args.scene_masks / path.name, cv2.remap(mask, mapx, mapy, cv2.INTER_NEAREST), args.overwrite)
        im = by_name[path.name]
        xys = cv2.undistortPoints(im.xys.reshape(-1, 1, 2), K, distortion, P=target_K).reshape(-1, 2) if len(im.xys) else im.xys
        ids = np.array([point_id_map[int(k)] if k != -1 else -1 for k in im.point3D_ids], dtype=np.int64)
        visible = (np.isfinite(xys).all(axis=1) & (xys[:, 0] > 0) &
                   (xys[:, 0] < 864) & (xys[:, 1] > 0) & (xys[:, 1] < 480))
        ids[~visible] = -1
        kept_observations[im.id] = visible
        new_images[i + 1] = im._replace(id=i + 1, camera_id=1, xys=xys, point3D_ids=ids)
    new_points = {}
    for old, point in points.items():
        keep = np.array([int(image_id) in image_id_map and
                         kept_observations[int(image_id)][int(index)]
                         for image_id, index in zip(point.image_ids, point.point2D_idxs)], dtype=bool)
        new_points[point_id_map[old]] = point._replace(id=point_id_map[old],
            image_ids=np.array([image_id_map[int(k)] for k in point.image_ids[keep]], dtype=np.int32),
            point2D_idxs=point.point2D_idxs[keep])
    new_camera = camera._replace(id=1, model="PINHOLE", params=np.array([target_K[0, 0], target_K[1, 1], 432., 240.]))
    args.scene_sparse.mkdir(parents=True, exist_ok=True)
    for filename in ("cameras.bin", "images.bin", "points3D.bin"):
        output_file(args.scene_sparse / filename, args.overwrite)
    io.write_model({1: new_camera}, new_images, new_points, str(args.scene_sparse), ".bin")
    write_json(args.work_dir / "geometry.json", {"undistortion_zoom": zoom,
               "raw_camera_model": camera.model, "output_size": [864, 480]}, args.overwrite)


def export_metadata(args):
    """Convert COLMAP world-to-camera (OpenCV) into GaVS camera-to-world (OpenGL)."""
    io = colmap_io()
    cameras, images, _ = io.read_model(str(args.scene_sparse), ".bin")
    camera = cameras[1]
    fx, fy, cx, cy = camera.params
    axis_flip = np.diag([1., -1., -1., 1.])
    raw_frames = []
    for key in sorted(images):
        image = images[key]
        w2c = np.eye(4)
        w2c[:3, :3] = io.qvec2rotmat(image.qvec)
        w2c[:3, 3] = image.tvec
        c2w = np.linalg.inv(w2c) @ axis_flip
        raw_frames.append({"file_path": "images/" + image.name, "colmap_id": key,
                           "transform_matrix": c2w.tolist()})
    extraction = args.work_dir / "extraction.json"
    fps = json.loads(extraction.read_text(encoding="utf-8"))["fps"] if extraction.exists() else args.fps
    write_json(args.metadata_path, {"w": camera.width, "h": camera.height,
        "fl_x": float(fx), "fl_y": float(fy), "cx": float(cx), "cy": float(cy),
        "fps": fps, "raw_frames": raw_frames}, args.overwrite)


def generate_inpainted_images(args):
    """Fill a 96-pixel border, optionally using neighboring frames before Telea."""
    cv2 = cv2_module()
    files = frames(args.scene_images)
    pad, width, height = 96, 1056, 672
    for i, path in enumerate(files):
        original = read_rgb(path)
        if args.inpaint_method == "import":
            if args.inpaint_source_dir is None:
                raise RuntimeError("--inpaint-source-dir is required for --inpaint-method import")
            canvas = read_rgb(args.inpaint_source_dir / path.name)
            if canvas.shape != (height, width, 3):
                raise RuntimeError(f"Imported completion must be 1056x672: {path.name}")
            if not np.array_equal(canvas[pad:pad+480, pad:pad+864], original):
                raise RuntimeError(f"Imported completion changed the original center: {path.name}")
        elif args.inpaint_method == "reflect":
            canvas = cv2.copyMakeBorder(original, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
        else:
            canvas = np.zeros((height, width, 3), np.uint8)
            canvas[pad:pad+480, pad:pad+864] = original
            known = np.zeros((height, width), np.uint8)
            known[pad:pad+480, pad:pad+864] = 255
            if args.inpaint_method == "temporal":
                offset = np.array([[1., 0, pad], [0, 1., pad], [0, 0, 1.]])
                for delta in range(1, args.inpaint_neighbors + 1):
                    for j in (i - delta, i + delta):
                        if not 0 <= j < len(files):
                            continue
                        neighbor = read_rgb(files[j])
                        matrix = estimate_homography(neighbor, original)
                        if matrix is None:
                            continue
                        matrix = offset @ matrix
                        foreground = read_mask(args.scene_masks / files[j].name, (864, 480))
                        static = cv2.erode(255 - foreground, np.ones((3, 3), np.uint8))
                        coverage = cv2.warpPerspective(static, matrix, (width, height), flags=cv2.INTER_NEAREST)
                        warped = cv2.warpPerspective(neighbor, matrix, (width, height))
                        fill = (known == 0) & (coverage > 127)
                        canvas[fill] = warped[fill]
                        known[fill] = 255
            hole = (known == 0).astype(np.uint8) * 255
            canvas = cv2.inpaint(canvas, hole, 3, cv2.INPAINT_TELEA)
            canvas[pad:pad+480, pad:pad+864] = original
        save_array(args.inpainted_images_dir / path.name, canvas, args.overwrite)
        print(f"inpaint {i+1}/{len(files)}", flush=True)


def generate_inpainted_masks(args):
    """Pad masks with static background, or reflect them with reflected images."""
    cv2 = cv2_module()
    for path in frames(args.scene_images):
        if args.inpaint_mask_source_dir:
            mask = read_mask(args.inpaint_mask_source_dir / path.name, (1056, 672))
        else:
            if args.inpaint_method == "import":
                raise RuntimeError("import completion requires --inpaint-mask-source-dir "
                                   "with 1056x672 masks.")
            mask = read_mask(args.scene_masks / path.name, (864, 480))
            border_mode = cv2.BORDER_REFLECT_101 if args.inpaint_method == "reflect" else cv2.BORDER_CONSTANT
            mask = cv2.copyMakeBorder(mask, 96, 96, 96, 96, border_mode, value=0)
        save_array(args.inpainted_masks_dir / path.name, mask, args.overwrite)


def copy_artifact(source, target, overwrite):
    if source.resolve() == target.resolve():
        return
    shutil.copy2(source, output_file(target, overwrite))


def assemble_scene(args):
    """Copy independently located artifacts into GaVS's fixed scene layout."""
    copy_artifact(args.metadata_path, args.output_dir / "blender.json", args.overwrite)
    for source, destination in ((args.inpainted_images_dir, args.output_dir / "inpainted/images"),
                                (args.inpainted_masks_dir, args.output_dir / "inpainted/fg_mask")):
        expected = frames(args.scene_images)
        for path in expected:
            copy_artifact(source / path.name, destination / path.name, args.overwrite)


def validate_scene(args):
    """Check original GaVS assumptions, dimensions, masks, points and poses."""
    directory = args.output_dir
    data = json.loads((directory / "blender.json").read_text(encoding="utf-8"))
    io = colmap_io()
    cameras, images, points = io.read_model(str(args.scene_sparse), ".bin")
    files = frames(args.scene_images)
    if len(files) < 64 or len(images) != len(files):
        raise RuntimeError("Need >=64 frames and one registered pose for every frame.")
    names = [p.name for p in files]
    for subdir in ("images", "fg_mask", "inpainted/images", "inpainted/fg_mask"):
        actual = {p.name for p in (directory / subdir).iterdir()}
        if actual != set(names):
            raise RuntimeError(f"{subdir} must contain exactly the matching PNG sequence. "
                               "GaVS lists all files there, including hidden/non-image files.")
    if (data["w"], data["h"]) != (864, 480):
        raise RuntimeError("JSON image dimensions must be 864x480 for the default GaVS configuration.")
    if [Path(frame["file_path"]).name for frame in data["raw_frames"]] != names:
        raise RuntimeError("JSON frame order differs from image sequence.")
    if [images[key].name for key in sorted(images)] != names:
        raise RuntimeError("Sparse image ID order differs from sequence.")
    camera = cameras[1]
    if (camera.width, camera.height) != (864, 480) or camera.model != "PINHOLE" or not np.allclose(camera.params,
            [data["fl_x"], data["fl_y"], data["cx"], data["cy"]]):
        raise RuntimeError("Sparse intrinsics differ from JSON.")
    if not np.allclose(camera.params[2:], [432, 240]) or min(camera.params[:2]) <= 0:
        raise RuntimeError("GaVS preprocessing requires positive focal lengths and a centered principal point.")
    if not points or not all(np.isfinite(point.xyz).all() for point in points.values()):
        raise RuntimeError("Sparse reconstruction has no valid points.")
    for key, frame in zip(sorted(images), data["raw_frames"]):
        name = images[key].name
        for subdir, size in (("images", (864, 480)), ("fg_mask", (864, 480)),
                             ("inpainted/images", (1056, 672)), ("inpainted/fg_mask", (1056, 672))):
            with Image.open(directory / subdir / name) as im:
                if im.size != size:
                    raise RuntimeError(f"Unexpected size for {subdir}/{name}: {im.size}")
                if "fg_mask" in subdir and not np.isin(np.asarray(im), [0, 255]).all():
                    raise RuntimeError(f"Mask must be binary: {subdir}/{name}")
        c2w = np.asarray(frame["transform_matrix"]) @ np.diag([1., -1., -1., 1.])
        w2c = np.eye(4)
        w2c[:3, :3] = io.qvec2rotmat(images[key].qvec)
        w2c[:3, 3] = images[key].tvec
        if not np.allclose(w2c @ c2w, np.eye(4), atol=1e-5):
            raise RuntimeError(f"Pose conversion failed: {name}")
        im = images[key]
        visible_ids = im.point3D_ids != -1
        xys = im.xys[visible_ids]
        xyz = np.array([points[int(pid)].xyz for pid in im.point3D_ids[visible_ids]]).reshape(-1, 3)
        depth = (xyz @ io.qvec2rotmat(im.qvec).T + im.tvec)[:, 2]
        valid_depth = (np.isfinite(xys).all(axis=1) & (xys[:, 0] > 0) & (xys[:, 0] < 864) &
                       (xys[:, 1] > 0) & (xys[:, 1] < 480) & np.isfinite(depth) & (depth > 0))
        if valid_depth.sum() < 10:
            raise RuntimeError(f"Fewer than 10 sparse observations in {name}; unreliable depth scale.")
        original = read_rgb(directory / "images" / name)
        completed = read_rgb(directory / "inpainted/images" / name)
        if not np.array_equal(original, completed[96:576, 96:960]):
            raise RuntimeError(f"Completion altered the central original frame: {name}")
        original_mask = read_mask(directory / "fg_mask" / name, (864, 480))
        completed_mask = read_mask(directory / "inpainted/fg_mask" / name, (1056, 672))
        if not np.array_equal(original_mask, completed_mask[96:576, 96:960]):
            raise RuntimeError(f"Completion mask center differs: {name}")
    methods = {}
    for step, name in (("masks", "mask_method"), ("inpaint", "inpaint_method")):
        record = args.work_dir / "steps" / (step + ".json")
        methods[name] = json.loads(record.read_text(encoding="utf-8"))["arguments"][name] if record.exists() else "unknown"
    report = {"frames": len(files), "points": len(points), **methods, "fps": data["fps"],
              "rolling_shutter_compensated": False,
              "note": "Basic motion/Telea/temporal methods are approximations, not paper preprocessing."}
    write_json(directory / "pipeline_report.json", report, args.overwrite)
    print(f"Validated {len(files)} frames, {len(points)} sparse points in {directory}", flush=True)
    print("Training (Linux; activate the GaVS environment, run from GaVS root, and check pretrained weight path):\n" +
          "WANDB_MODE=offline " + shlex.join(["python", "train.py", "+experiment=layered_gavs_overfit",
                      'dataset.data_path=' + str(directory),
                      'hydra.run.dir=' + str(directory.parent / (directory.name + '_results')),
                      'train.load_weights_folder=' + args.pretrained_weights,
                      'train.num_gpus=1']))
    if data["fps"] != 24:
        print("GaVS original evaluate.py encodes at 24 FPS. Re-encode with the FPS in blender.json "
              "to retain the extracted sequence's duration.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--video", type=Path, help="Input MP4/MOV, required for extract")
    parser.add_argument("--output-dir", required=True, type=Path, help="Final GaVS scene directory")
    parser.add_argument("--steps", nargs="+", choices=("all",) + STEPS, default=["all"], help="Independent steps, executed in dependency order")
    parser.add_argument("--work-dir", type=Path, help="Intermediate workspace; default OUTPUT/_pipeline")
    parser.add_argument("--frames-dir", type=Path, help="Extracted distorted frames (not final GaVS images)")
    parser.add_argument("--masks-dir", type=Path, help="Foreground masks for extracted frames")
    parser.add_argument("--sparse-dir", type=Path, help="Raw COLMAP model/database folder")
    parser.add_argument("--metadata-path", type=Path, help="Generated blender.json location, copied to final scene")
    parser.add_argument("--inpainted-images-dir", type=Path, help="Completion output, copied to final scene")
    parser.add_argument("--inpainted-masks-dir", type=Path, help="Completion masks output, copied to final scene")
    parser.add_argument("--fps", type=float, default=24, help="Extraction FPS; 24 matches original GaVS encoder")
    parser.add_argument("--start", default="0", help="Start time in seconds or hh:mm:ss")
    parser.add_argument("--duration", type=float, help="Extract only this many seconds")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--colmap", default="colmap")
    parser.add_argument("--glomap", default="glomap")
    parser.add_argument("--mapper", choices=("colmap", "glomap"), default="colmap")
    parser.add_argument("--sfm-gpu", action="store_true", help="Enable CUDA SIFT extraction/matching; CPU by default")
    parser.add_argument("--match-overlap", type=int, default=20)
    parser.add_argument("--camera-model", choices=("PINHOLE", "SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV"), default="PINHOLE", help="PINHOLE for lens-corrected footage; radial/OPENCV for distorted footage")
    parser.add_argument("--camera-params", help="Optional initial intrinsics in COLMAP camera model order, for the EXTRACTED 864x480 images")
    parser.add_argument("--mask-method", choices=("motion", "maskrcnn", "static", "import"), default="motion")
    parser.add_argument("--mask-source-dir", type=Path, help="Imported masks for extracted distorted frames, white=foreground")
    parser.add_argument("--mask-score", type=float, default=0.7)
    parser.add_argument("--mask-dilate", type=int, default=3)
    parser.add_argument("--dynamic-classes", type=int, nargs="+", default=[1, 2, 3, 4, 6, 7, 8, 17, 18], help="COCO IDs for semantic potential foreground")
    parser.add_argument("--device", default="cuda:0", help="Mask R-CNN device; use cpu if needed")
    parser.add_argument("--motion-stride", type=int, default=2)
    parser.add_argument("--motion-threshold", type=float, default=30)
    parser.add_argument("--inpaint-method", choices=("telea", "temporal", "reflect", "import"), default="temporal")
    parser.add_argument("--inpaint-neighbors", type=int, default=3)
    parser.add_argument("--inpaint-source-dir", type=Path, help="External completion: 1056x672, preserve original 864x480 center")
    parser.add_argument("--inpaint-mask-source-dir", type=Path, help="External 1056x672 foreground masks")
    parser.add_argument("--pretrained-weights", default="./gavs-data/re10k_v2_checkpoints",
                        help="Path printed in the suggested unchanged GaVS training command")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite generated files; never clears folders/databases")
    args = parser.parse_args(argv)
    if not np.isfinite(args.fps) or args.fps <= 0 or (args.duration is not None and
            (not np.isfinite(args.duration) or args.duration <= 0)):
        parser.error("fps and duration must be positive")
    if not 0 <= args.mask_score <= 1 or not np.isfinite(args.motion_threshold) or args.motion_threshold <= 0:
        parser.error("mask-score must be in [0, 1] and motion-threshold must be positive")
    if min(args.mask_dilate, args.inpaint_neighbors) < 0 or min(args.match_overlap, args.motion_stride) < 1:
        parser.error("invalid radius, overlap or stride")
    if args.output_dir.resolve() == ROOT or args.output_dir.resolve() in ROOT.parents:
        parser.error("output-dir must be a dedicated dataset folder, not the repository or its parent")
    args.output_dir = args.output_dir.resolve()
    args.work_dir = (args.work_dir or args.output_dir / "_pipeline").resolve()
    defaults = {"frames_dir": args.work_dir / "frames", "masks_dir": args.work_dir / "fg_mask",
                "sparse_dir": args.work_dir / "reconstruction",
                "metadata_path": args.output_dir / "blender.json",
                "inpainted_images_dir": args.output_dir / "inpainted/images",
                "inpainted_masks_dir": args.output_dir / "inpainted/fg_mask"}
    for key, default in defaults.items():
        setattr(args, key, (getattr(args, key) or default).resolve())
    for key in ("video", "mask_source_dir", "inpaint_source_dir", "inpaint_mask_source_dir"):
        if getattr(args, key):
            setattr(args, key, getattr(args, key).resolve())
    args.scene_images = args.output_dir / "images"
    args.scene_masks = args.output_dir / "fg_mask"
    args.scene_sparse = args.output_dir / "sparse/0"
    # Each artifact is a separate directory; prevent accidental source replacement
    # if the user redirects a mask output to an image folder, including via symlinks.
    directories = [(name, getattr(args, name)) for name in
                   ("frames_dir", "masks_dir", "sparse_dir", "scene_images", "scene_masks",
                    "scene_sparse", "inpainted_images_dir", "inpainted_masks_dir")]
    for name, final in (("final_inpainted_images", args.output_dir / "inpainted/images"),
                        ("final_inpainted_masks", args.output_dir / "inpainted/fg_mask")):
        expected = args.inpainted_images_dir if name == "final_inpainted_images" else args.inpainted_masks_dir
        if final != expected:
            directories.append((name, final))
    for i, (name, path) in enumerate(directories):
        if path == ROOT or path in ROOT.parents:
            parser.error(f"{name} must be a dedicated data directory, not the repository or its parent")
        if args.work_dir == path or path in args.work_dir.parents:
            parser.error(f"work-dir must not be inside an artifact directory: {name}")
        for other_name, other in directories[i + 1:]:
            if path == other or path in other.parents or other in path.parents:
                parser.error(f"Artifact directories must be disjoint: {name} and {other_name}")
    if args.metadata_path.suffix.lower() != ".json":
        parser.error("metadata-path must be a .json file")
    if any(path == args.metadata_path or path in args.metadata_path.parents for _, path in directories):
        parser.error("metadata-path must not be inside an image/mask/sparse artifact directory")
    for source, target in ((args.mask_source_dir, args.masks_dir),
                           (args.inpaint_source_dir, args.inpainted_images_dir),
                           (args.inpaint_mask_source_dir, args.inpainted_masks_dir)):
        if source and (source == target or source in target.parents or target in source.parents):
            parser.error("Imported source and generated output directories must be disjoint")
    return args


def main(argv=None):
    args = parse_args(argv)
    selected = STEPS if "all" in args.steps else tuple(step for step in STEPS if step in args.steps)
    functions = dict(zip(STEPS, (extract_frames, generate_dynamic_masks, reconstruct_scene,
        prepare_geometry, export_metadata, generate_inpainted_images, generate_inpainted_masks,
        assemble_scene, validate_scene)))
    for step in selected:
        print(f"\n[{step}]", flush=True)
        functions[step](args)
        write_json(args.work_dir / "steps" / (step + ".json"), {"step": step,
            "arguments": {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()}}, overwrite=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Pipeline stopped: {exc}", file=sys.stderr)
        sys.exit(1)
