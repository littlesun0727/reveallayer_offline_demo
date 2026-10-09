"""CPU checks for coordinate mapping and alpha export (no torch/checkpoints needed)."""
import ast
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import run_overlays as r


class GeometryTests(unittest.TestCase):
    def test_crop_at_border_uses_edge_padding(self):
        im = Image.new('RGB', (80, 60), (25, 80, 170))
        crop = r.extract_crop(im, [-20, -30, 100, 90])
        self.assertEqual(crop.size, (120, 120))
        self.assertTrue(np.all(np.asarray(crop) == [25, 80, 170]))

    def test_roundtrip_position_and_alpha(self):
        image = Image.new('RGB', (1080, 1435), 'white')
        target = dict(id='starfish', bbox=[51, 44, 270, 207])
        box = r.crop_box(target['bbox'], image.size, 0.35, 128)
        model, g = r.prepare_input(image, box, [target], 1024)
        self.assertTrue(all(v % 16 == 0 for v in model.size))
        self.assertTrue(all(v % 16 == 0 for v in g['boxes'][0]))
        raw = Image.new('RGBA', model.size, (10, 20, 30, 128))
        restored = r.restore_layer(raw, g, image.size)
        self.assertEqual(restored.size, image.size)
        self.assertEqual(restored.getpixel((100, 100))[3], 128)
        self.assertEqual(restored.getpixel((800, 1200))[3], 0)

    def test_full_image_roundtrip_keeps_entire_canvas(self):
        im = Image.new('RGB', (1080, 1435), 'white')
        _, g = r.prepare_input(im, [0, 0, 1080, 1435], [], 1024)
        raw = Image.new('RGBA', g['model_size'], (20, 30, 40, 255))
        full = r.restore_layer(raw, g, im.size)
        self.assertEqual(full.getchannel('A').getextrema(), (255, 255))

    def test_reference_size_scaling_and_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'analysis.json'
            r.dump(path, dict(reference_size=[100, 100], objects=[
                dict(id='photo', kind='photo', bbox=[0, 0, 100, 100]),
                dict(id='b', kind='overlay', bbox=[10, 20, 30, 40]),
                dict(id='a', kind='overlay', bbox=[0, 0, 10, 10])], layer_order=['a', 'b']))
            targets = r.load_targets(path, (200, 300))
            self.assertEqual([t['id'] for t in targets], ['a', 'b'])
            self.assertEqual(targets[1]['bbox'], [20, 60, 60, 120])
            padded = r.load_targets(path, (200, 300), ['b'], margin=8)[0]
            self.assertEqual(padded['bbox'], [12, 52, 68, 128])
            self.assertEqual(padded['analysis_bbox'], [20, 60, 60, 120])
            with self.assertRaises(ValueError):
                r.load_targets(path, (200, 300), ['missing'])

    def test_model_check_rejects_lfs_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flux = root/'FLUX.1-dev'
            ckpt = root/'RevealLayer'
            ckpt.mkdir()
            flux.mkdir()
            for name in ['pytorch_lora_weights.safetensors', 'Refiner.pt', 'layer_pe.pt', 'transparent_decoder_ckpt.pth']:
                (ckpt/name).write_bytes(b'dummy')
            for component in ['transformer', 'vae', 'text_encoder', 'text_encoder_2']:
                (flux/component).mkdir()
                (flux/component/'config.json').write_text('{}')
                (flux/component/'model.safetensors').write_bytes(b'dummy')
            for component in ['tokenizer', 'tokenizer_2', 'scheduler']:
                (flux/component).mkdir()
                for name in ['tokenizer_config.json', 'tokenizer.json', 'scheduler_config.json']:
                    (flux/component/name).write_text('{}')
            (flux/'model_index.json').write_text('{}')
            (ckpt/'Refiner.pt').write_bytes(b'version https://git-lfs.github.com/spec/v1\n')
            with self.assertRaisesRegex(ValueError, 'Git LFS pointer'):
                r.model_paths(SimpleNamespace(models=root, flux=None, ckpt=None, decoder=None))


if __name__ == '__main__':
    unittest.main()
