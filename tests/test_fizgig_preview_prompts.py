"""Exercise the real v7 adapter command construction without starting a GPU engine."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from kohya_core import fizgig_adapter as adapter


class FizgigPreviewPromptTests(unittest.TestCase):
    def run_adapter(self, extra=None, caption="1girl, blue hair, white dress, standing in a garden"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "data"
            media.mkdir()
            (media / "image.png").write_bytes(b"fixture")
            (media / "image.txt").write_text(caption, encoding="utf-8")
            checkpoint = root / "model.safetensors"
            checkpoint.write_bytes(b"fixture")
            core = MagicMock()
            core.MIN_IMAGES = {"flux2_fz": 1}
            core.FIZGIG_IMAGE_EXTS = {".png"}
            core.dataset_train_dir.return_value = str(media)
            core._fizgig_qwen_image_subsets.return_value = ([(str(media), 1, 1)], [])
            core._sanitize_dirname.side_effect = lambda text: text or ""
            core.output_name_for.return_value = "preview_test"
            core.data_sub.side_effect = lambda *parts: str(root.joinpath(*parts))
            core._sample_preview_enabled.return_value = True
            core._fizgig_sample_epochs.return_value = 1
            core._sample_seed.return_value = 1234
            core._save_every_note.return_value = 1
            core._fizgig_warmup_log_filter.side_effect = lambda fn: fn
            commands, effective = [], {}
            core.run_stream.side_effect = lambda cmd, **kw: commands.append(cmd) or 0
            core._record_effective.side_effect = lambda **kw: effective.update(kw)
            files = {key: str(checkpoint) for key in ("dit", "te", "vae")}
            params = {"project": "preview_test", "trigger": "my_subject", "sample_prompt": "", **(extra or {})}
            record = {"version": adapter.VERSION, "source": str(root / "source"), "python": sys.executable}
            with patch.object(adapter, "source_version", return_value=adapter.VERSION), \
                    patch.object(adapter, "models", return_value=files), \
                    patch.object(adapter, "_validate"), \
                    patch.object(adapter, "_env", return_value={}), \
                    patch.object(adapter, "_helper_configs", return_value=None):
                adapter.train_lora(core, record, lambda line: None, "flux2_fz", params, 24, None, None)
            cmd = commands[-1]
            prompt_file = Path(cmd[cmd.index("--sample_prompts") + 1])
            return prompt_file.read_text(encoding="utf-8").strip(), effective, cmd

    def test_blank_preview_uses_caption_and_trigger(self):
        prompt, _, _ = self.run_adapter()
        self.assertIn("blue hair", prompt)
        self.assertIn("my_subject", prompt)
        self.assertNotIn("--d", prompt)

    def test_custom_prompts_keep_their_content_and_lines(self):
        prompt, _, _ = self.run_adapter({"sample_prompt": "a red car\na mountain"})
        self.assertEqual(prompt, "a red car\na mountain")

    def test_effective_settings_include_actual_prompt(self):
        prompt, effective, cmd = self.run_adapter()
        self.assertEqual(effective["sample_prompt"], prompt)
        self.assertEqual(effective["sample_prompt_source"], "dataset_caption")
        self.assertEqual(cmd[cmd.index("--sample_seed") + 1], "1234")

    def test_whitespace_uses_caption(self):
        prompt, _, _ = self.run_adapter({"sample_prompt": "   \n  "})
        self.assertIn("blue hair", prompt)


if __name__ == "__main__":
    unittest.main()
