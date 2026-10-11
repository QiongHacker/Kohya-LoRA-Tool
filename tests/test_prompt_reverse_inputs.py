import ast
import base64
import hashlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kohya_core.captioning import CaptionError, _image_data
from kohya_core import prompt_reverse
from kohya_core.training_media import image_preview
from gui.modern_host import ModernUIBridge
from PIL import Image, features


class PromptReverseInputTests(unittest.TestCase):
    def test_avif_only_folder_matches_feedback_with_nested_and_uppercase_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / '图集' / '老八'
            (root / '子目录').mkdir(parents=True)
            expected = [root / 'sample.jpg@1192w.avif', root / 'sample.png@1192w.AVIF',
                        root / '子目录' / '副本.avif']
            for image in expected:
                image.write_bytes(b'scanner fixture, decoding is tested separately')
            (root / 'tags.txt').write_text('existing caption', encoding='utf-8')
            selection = prompt_reverse.prepare({'path': str(root), 'method': 'wd14'})
            self.assertEqual(set(selection['images']), set(expected))
            self.assertEqual(selection['root'], root.resolve())

    def test_single_avif_is_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / 'sample.avif'
            image.write_bytes(b'scanner fixture')
            self.assertEqual(prompt_reverse.prepare({'path': str(image)})['images'], [image])

    def test_formats_match_existing_preprocessor_without_importing_its_side_effects(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'preprocess.py').read_text(encoding='utf-8-sig'))
        formats = next(ast.literal_eval(node.value) for node in tree.body
                       if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == 'IMAGE_EXTS' for target in node.targets))
        self.assertEqual(prompt_reverse.IMAGE_EXTS, formats)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for extension in formats:
                (root / ('sample' + extension.upper())).write_bytes(b'scanner fixture')
            self.assertEqual(len(prompt_reverse.prepare({'path': temp})['images']), len(formats))

    def test_empty_or_sidecar_only_folder_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / 'tags.txt').write_text('caption', encoding='utf-8')
            with self.assertRaisesRegex(CaptionError, '这个文件夹没有支持的图片'):
                prompt_reverse.prepare({'path': temp})

    def test_unsupported_single_file_lists_avif_as_a_supported_choice(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'tags.txt'
            path.write_text('caption', encoding='utf-8')
            with self.assertRaisesRegex(CaptionError, 'AVIF'):
                prompt_reverse.prepare({'path': str(path)})

    def test_avif_counts_toward_batch_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            for index in range(3):
                (Path(temp) / f'{index}.avif').write_bytes(b'scanner fixture')
            with patch.object(prompt_reverse, 'MAX_IMAGES', 2):
                with self.assertRaisesRegex(CaptionError, '一次最多反推'):
                    prompt_reverse.prepare({'path': temp})

    def test_recursive_scan_keeps_hidden_directories_excluded_and_sorts_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / '.cache').mkdir()
            (root / '.cache' / 'hidden.avif').write_bytes(b'scanner fixture')
            for name in ('z.avif', 'A.png', 'b.AVIF'):
                (root / name).write_bytes(b'scanner fixture')
            names = [path.name for path in prompt_reverse.prepare({'path': temp})['images']]
            self.assertEqual(names, ['A.png', 'b.AVIF', 'z.avif'])

    def test_image_picker_uses_the_same_format_set_as_the_scanner(self):
        with tempfile.TemporaryDirectory() as temp:
            settings = {}
            core = SimpleNamespace(data_sub=lambda *parts: str(Path(temp).joinpath(*parts)),
                                   _load_app_settings=lambda: dict(settings),
                                   _save_app_settings=lambda values: settings.update(values))
            bridge = ModernUIBridge(core)
            calls = []
            image = Path(temp) / 'sample.avif'
            image.write_bytes(b'scanner fixture')
            def choose(dialog_type, **options):
                calls.append((dialog_type, options))
                return (str(image),)
            bridge._window = SimpleNamespace(create_file_dialog=choose)
            webview = SimpleNamespace(FileDialog=SimpleNamespace(OPEN='open', FOLDER='folder'))
            with patch.dict(sys.modules, {'webview': webview}):
                result = bridge.choose_path('image', str(image), 'prompt_reverse_input')
            self.assertTrue(result['ok'], result)
            self.assertEqual(result['path'], str(image))
            filters = calls[0][1]['file_types']
            self.assertIn('*.avif', filters[0])
            self.assertEqual(filters, ('Image files (' + ';'.join('*' + ext for ext in sorted(prompt_reverse.IMAGE_EXTS)) + ')',))


@unittest.skipUnless(features.check('avif'), 'This Pillow build has no AVIF decoder')
class PromptReverseAvifDecodeTests(unittest.TestCase):
    def make_avif(self, root):
        image = Path(root) / 'sample.jpg@1192w.avif'
        Image.new('RGB', (1600, 64), '#517ca0').save(image, format='AVIF')
        return image

    def test_wd14_stages_real_avif_without_changing_original_or_existing_caption(self):
        with tempfile.TemporaryDirectory() as temp:
            image = self.make_avif(temp)
            before = hashlib.sha256(image.read_bytes()).hexdigest()
            caption = image.with_suffix('.txt')
            caption.write_text('keep existing tags', encoding='utf-8')
            staged = []
            def run_stream(command, **kwargs):
                root = Path(command[5])
                for target in root.glob('*.png'):
                    with Image.open(target) as decoded:
                        decoded.load()
                        staged.append((decoded.format, decoded.mode, decoded.size))
                    target.with_suffix('.txt').write_text('test keyword', encoding='utf-8')
            core = SimpleNamespace(_pick_preprocess_python=lambda: sys.executable,
                                   find_python=lambda: (sys.executable,), check_stop=lambda: None,
                                   _ensure_preprocess_deps=lambda *args, **kwargs: True,
                                   get_kohya_dir=lambda: temp, KIT_DIR=temp,
                                   build_direct_env=lambda: {}, run_stream=run_stream)
            selection = prompt_reverse.prepare({'path': str(image), 'method': 'wd14'})
            result = prompt_reverse._keywords(core, selection, lambda line: None, lambda *args: None)
            self.assertEqual(result[image], {'caption': 'test keyword', 'status': 'generated'})
            self.assertEqual(len(staged), 1)
            self.assertEqual(staged[0][:2], ('PNG', 'RGBA'))
            self.assertLessEqual(max(staged[0][2]), 1536)
            self.assertEqual(hashlib.sha256(image.read_bytes()).hexdigest(), before)
            self.assertEqual(caption.read_text(encoding='utf-8'), 'keep existing tags')

    def test_natural_description_and_preview_decode_real_avif(self):
        with tempfile.TemporaryDirectory() as temp:
            image = self.make_avif(temp)
            before = image.read_bytes()
            with Image.open(io.BytesIO(base64.b64decode(_image_data(image)))) as decoded:
                decoded.load()
                self.assertEqual(decoded.format, 'JPEG')
                self.assertLessEqual(max(decoded.size), 1024)
            preview = image_preview(temp, image.name)
            self.assertTrue(preview['data_url'].startswith('data:image/webp;base64,'))
            self.assertEqual((preview['width'], preview['height']), (1600, 64))
            self.assertEqual(image.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()