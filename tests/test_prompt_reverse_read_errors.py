import errno
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, PngImagePlugin
from kohya_core.captioning import CaptionError
from kohya_core import prompt_reverse


class PromptReverseReadErrorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.image = Path(self.directory.name) / 'sample.png'
        Image.new('RGB', (32, 64), '#517ca0').save(self.image)
        self.original = self.image.read_bytes()
        self.logs = []
        self.core = SimpleNamespace(_pick_preprocess_python=lambda: sys.executable,
                                    find_python=lambda: (sys.executable,), check_stop=lambda: None)
        self.selection = prompt_reverse.prepare({'path': str(self.image), 'method': 'wd14'})

    def run_read(self):
        return prompt_reverse._keywords(self.core, self.selection, self.logs.append, lambda *args: None)[self.image]

    def assert_failure(self, result, stage, exception):
        self.assertEqual(result['status'], 'failed')
        self.assertIn(stage, result['error'])
        self.assertIn(exception, result['error'])
        self.assertTrue(any(stage in line and exception in line for line in self.logs), self.logs)
        self.assertEqual(self.image.read_bytes(), self.original)

    def test_source_access_error_is_distinct_from_temporary_write_error(self):
        with patch('PIL.Image.open', side_effect=PermissionError(errno.EACCES, 'access denied to original')):
            result = self.run_read()
        self.assert_failure(result, '读取原图', 'PermissionError')
        self.assertNotIn('写入临时图片', result['error'])

    def test_temporary_save_permission_error_is_not_reported_as_unreadable_original(self):
        with patch.object(Image.Image, 'save', side_effect=PermissionError(errno.EACCES, 'access denied to temp image')):
            result = self.run_read()
        self.assert_failure(result, '写入临时图片', 'PermissionError')
        self.assertIn('13', result['error'])

    def test_full_temporary_disk_preserves_errno_and_stage(self):
        with patch.object(Image.Image, 'save', side_effect=OSError(errno.ENOSPC, 'No space left on device')):
            result = self.run_read()
        self.assert_failure(result, '写入临时图片', 'OSError')
        self.assertIn(str(errno.ENOSPC), result['error'])

    def test_transform_failure_has_its_own_stage(self):
        with patch.object(Image.Image, 'thumbnail', side_effect=ValueError('cannot resize input')):
            result = self.run_read()
        self.assert_failure(result, '处理图片', 'ValueError')

    def test_actual_invalid_png_reports_decoder_exception_and_keeps_safety_limits(self):
        self.image.write_bytes(b'not a real PNG')
        self.original = self.image.read_bytes()
        limits = (Image.MAX_IMAGE_PIXELS, PngImagePlugin.MAX_TEXT_CHUNK, PngImagePlugin.MAX_TEXT_MEMORY)
        result = self.run_read()
        self.assert_failure(result, '读取原图', 'UnidentifiedImageError')
        self.assertEqual(limits, (Image.MAX_IMAGE_PIXELS, PngImagePlugin.MAX_TEXT_CHUNK, PngImagePlugin.MAX_TEXT_MEMORY))

    def test_temp_directory_creation_failure_is_actionable_without_a_separate_log_file(self):
        with patch.object(prompt_reverse.tempfile, 'TemporaryDirectory', side_effect=PermissionError(errno.EACCES, 'cannot create temp directory')):
            with self.assertRaisesRegex(CaptionError, '创建临时目录.*PermissionError'):
                self.run_read()
        self.assertTrue(any('创建临时目录' in line for line in self.logs), self.logs)
        self.assertEqual(self.image.read_bytes(), self.original)

    def test_exception_details_are_redacted_in_ui_and_log(self):
        detail = 'api_key=FAKE_TEST_KEY; path C:\\Users\\private-user\\image.png'
        with patch.object(Image.Image, 'save', side_effect=OSError(detail)):
            result = self.run_read()
        self.assert_failure(result, '写入临时图片', 'OSError')
        combined = result['error'] + '\n' + '\n'.join(self.logs)
        self.assertNotIn('FAKE_TEST_KEY', combined)
        self.assertNotIn('private-user', combined)
        self.assertIn('<REDACTED>', combined)
        self.assertIn('%USERPROFILE%', combined)

    def test_stop_request_is_not_swallowed_as_an_image_error(self):
        class StopRequested(Exception):
            pass
        self.core.check_stop = lambda: (_ for _ in ()).throw(StopRequested('stop'))
        with self.assertRaises(StopRequested):
            self.run_read()
        self.assertEqual(self.image.read_bytes(), self.original)


if __name__ == '__main__':
    unittest.main()