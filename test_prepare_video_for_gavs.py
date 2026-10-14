"""Contract checks for the NEW video preprocessor; no GaVS source is changed.

Run: python -m unittest test_prepare_video_for_gavs -v
Optionally set GAVS_TEST_COLMAP and GAVS_TEST_FFMPEG to executable paths to
exercise the real programs. Test data is created in temporary directories.
"""
import contextlib
import io
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import prepare_video_for_gavs as pipeline


class PipelineContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="gavs-preprocess-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def args(self, *extra):
        return pipeline.parse_args(["--output-dir", str(self.root / "scene"), *extra])

    def raw_model(self, args):
        """Known nontrivial poses with unsorted image IDs and sparse point IDs."""
        model = pipeline.colmap_io()
        camera = model.Camera(9, "PINHOLE", 864, 480, np.array([620., 625., 432., 240.]))
        images = {}
        point_ids = np.arange(17, 17 + 12 * 19, 19, dtype=np.int64)
        xyz = np.column_stack((np.linspace(-.8, .8, 12), np.linspace(-.4, .4, 12),
                               np.linspace(3., 6., 12)))
        ids = np.arange(64, 0, -1) * 7
        for i, image_id in enumerate(ids):
            name = f"{i:05d}.png"
            rgb = np.zeros((480, 864, 3), np.uint8)
            rgb[..., 0] = i * 3
            rgb[80:400, 100:700, 1] = 120
            pipeline.save_array(args.frames_dir / name, rgb, False)
            mask = np.zeros((480, 864), np.uint8)
            mask[160:240, 300:350] = 255
            pipeline.save_array(args.masks_dir / name, mask, False)
            angle = .04 * np.sin(i / 9)
            qvec = np.array([np.cos(angle / 2), 0., np.sin(angle / 2), 0.])
            tvec = np.array([.15 * np.cos(i / 11), .01 * i / 64, 0.])
            camera_xyz = xyz @ model.qvec2rotmat(qvec).T + tvec
            xys = camera_xyz[:, :2] / camera_xyz[:, 2:] * camera.params[:2] + camera.params[2:]
            images[int(image_id)] = model.Image(int(image_id), qvec, tvec, 9, name, xys, point_ids.copy())
        points = {int(pid): model.Point3D(int(pid), position, np.array([120, 80, 20], np.uint8),
                  0., ids.astype(np.int32), np.full(64, index, dtype=np.int32))
                  for index, (pid, position) in enumerate(zip(point_ids, xyz))}
        raw = args.sparse_dir / "0"
        raw.mkdir(parents=True)
        model.write_model({9: camera}, images, points, str(raw), ".bin")
        return model, images, points

    def test_geometry_metadata_and_completion_match_original_loader_contract(self):
        args = self.args("--inpaint-method", "reflect", "--metadata-path", str(self.root / "external/meta.json"),
                         "--inpainted-images-dir", str(self.root / "external/completed"),
                         "--inpainted-masks-dir", str(self.root / "external/masks"))
        model, old_images, _ = self.raw_model(args)
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline.prepare_geometry(args)
            pipeline.export_metadata(args)
            pipeline.generate_inpainted_images(args)
            pipeline.generate_inpainted_masks(args)
            pipeline.assemble_scene(args)
            pipeline.validate_scene(args)
        cameras, images, points = model.read_model(str(args.scene_sparse), ".bin")
        self.assertEqual(list(sorted(images)), list(range(1, 65)))
        self.assertEqual(list(sorted(points)), list(range(1, 13)))
        self.assertEqual(cameras[1].model, "PINHOLE")
        for i in range(64):
            name = f"{i:05d}.png"
            im = images[i + 1]
            old = next(im for im in old_images.values() if im.name == name)
            np.testing.assert_allclose(im.qvec, old.qvec)
            np.testing.assert_allclose(im.tvec, old.tvec)
            np.testing.assert_array_equal(im.point3D_ids, np.arange(1, 13))
        for point in points.values():
            np.testing.assert_array_equal(point.image_ids, np.arange(1, 65))
            for image_id, index in zip(point.image_ids, point.point2D_idxs):
                self.assertEqual(images[int(image_id)].point3D_ids[int(index)], point.id)
        # Original GaVS blindly lists the image folder; validation must reject extras.
        (args.output_dir / "inpainted/images/not_an_image.txt").touch()
        with self.assertRaisesRegex(RuntimeError, "exactly the matching PNG"):
            pipeline.validate_scene(args)

    def test_undistortion_updates_focal_lengths_and_never_samples_missing_border(self):
        K = np.array([[650., 0., 450.], [0., 655., 230.], [0., 0., 1.]])
        target, mx, my, zoom = pipeline.undistortion_map(K, np.array([.16, -.01, 0., 0., 0.]))
        self.assertGreater(zoom, 1.)
        self.assertTrue(np.isfinite(mx).all() and np.isfinite(my).all())
        self.assertTrue(mx.min() >= 0 and mx.max() <= 863 and my.min() >= 0 and my.max() <= 479)
        np.testing.assert_allclose(target.diagonal()[:2], K.diagonal()[:2] * zoom)
        np.testing.assert_array_equal(target[:2, 2], [432., 240.])

    def test_cannot_mix_masks_and_frames_or_override_import_source(self):
        shared = str(self.root / "shared")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args("--frames-dir", shared, "--masks-dir", shared)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args("--mask-source-dir", shared, "--masks-dir", shared, "--overwrite")

    def test_motion_masks_and_telea_preserve_center_and_binary_masks(self):
        args = self.args("--mask-dilate", "0", "--inpaint-method", "telea")
        rng = np.random.default_rng(21)
        base = rng.integers(0, 256, (480, 864, 3), dtype=np.uint8)
        for i in range(5):
            rgb = base.copy()
            rgb[100:160, 150+i*45:210+i*45] = 255
            pipeline.save_array(args.frames_dir / f"{i:05d}.png", rgb, False)
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline.generate_dynamic_masks(args)
        for path in pipeline.frames(args.masks_dir):
            mask = pipeline.read_mask(path, (864, 480))
            self.assertGreater(np.count_nonzero(mask), 100)
            self.assertLess(np.count_nonzero(mask), mask.size // 5)
        # Completion functions can be exercised independently of reconstruction.
        pipeline.save_array(args.scene_images / "00000.png", base, False)
        pipeline.save_array(args.scene_masks / "00000.png", np.zeros((480, 864), np.uint8), False)
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline.generate_inpainted_images(args)
        pipeline.generate_inpainted_masks(args)
        completed = pipeline.read_rgb(args.inpainted_images_dir / "00000.png")
        self.assertEqual(completed.shape, (672, 1056, 3))
        np.testing.assert_array_equal(completed[96:576, 96:960], base)

    @unittest.skipUnless(os.getenv("GAVS_TEST_COLMAP"), "Set GAVS_TEST_COLMAP for executable check")
    def test_real_colmap_uses_appended_png_mask_name(self):
        args = self.args("--colmap", os.environ["GAVS_TEST_COLMAP"])
        rng = np.random.default_rng(4)
        rgb = rng.integers(0, 256, (480, 864, 3), dtype=np.uint8)
        images, masks = self.root / "images", self.root / "masks"
        for name in ("00000.png", "00001.png"):
            pipeline.save_array(images / name, rgb, False)
        pipeline.save_array(masks / "00000.png.png", np.zeros((480, 864), np.uint8), False)
        pipeline.save_array(masks / "00001.png.png", np.full((480, 864), 255, np.uint8), False)
        gpu_option = pipeline.colmap_options(args, "feature_extractor", "SiftExtraction.use_gpu", "FeatureExtraction.use_gpu")
        thread_option = pipeline.colmap_options(args, "feature_extractor", "SiftExtraction.num_threads", "FeatureExtraction.num_threads")
        result = subprocess.run([args.colmap, "feature_extractor", "--database_path", str(self.root / "test.db"),
            "--image_path", str(images), "--ImageReader.mask_path", str(masks),
            "--"+gpu_option, "0", "--"+thread_option, "1"], capture_output=True, text=True, errors="replace")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        with contextlib.closing(sqlite3.connect(self.root / "test.db")) as database:
            counts = dict(database.execute("SELECT images.name, keypoints.rows FROM images JOIN keypoints USING(image_id)"))
        self.assertEqual(counts["00000.png"], 0)
        self.assertGreater(counts["00001.png"], 50)

    @unittest.skipUnless(os.getenv("GAVS_TEST_FFMPEG"), "Set GAVS_TEST_FFMPEG for executable check")
    def test_real_ffmpeg_extracts_64_correctly_named_and_sized_frames(self):
        ffmpeg = os.environ["GAVS_TEST_FFMPEG"]
        video = self.root / "input.mp4"
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                        "testsrc=size=960x540:rate=24", "-frames:v", "64", str(video)], check=True)
        args = self.args("--ffmpeg", ffmpeg, "--video", str(video))
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline.extract_frames(args)
        files = pipeline.frames(args.frames_dir)
        self.assertEqual(len(files), 64)
        for file in files:
            with Image.open(file) as image:
                self.assertEqual(image.size, (864, 480))


if __name__ == "__main__":
    unittest.main()
