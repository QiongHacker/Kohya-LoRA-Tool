import contextlib
import copy
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kohya_gui
from kohya_core.project_config import training_params

class Var:
    def __init__(self, value=''): self.value = value
    def get(self): return self.value

class LearningRateCompatibilityTests(unittest.TestCase):
    def app(self, value):
        app = kohya_gui.App.__new__(kohya_gui.App)
        app.current_project = None
        app._utility_only = False
        app._label_editor = None
        app.root = None
        app.mode, app.base_type = 'anima_fz', 'anima'
        app._at_sub_label = lambda: 'character'
        app._concept_type_key = lambda: 'form'
        app._clean_concept_flag = lambda: True
        app.param_vars = {'te_lr': Var(value), 'unet_lr': Var('8.50e-5')}
        values = {'trigger': 'subject', 'strong_bind': True, 'sample_preview': False,
                  'sample_prompt': '', 'crop_ratio': '不裁剪', 'reg': '', 'raw_dir': '',
                  'base_model': '', 'unet_only': True, 'style': '动漫', 'global_pos': '',
                  'global_neg': '', 'amd': False, 'train_env': ''}
        for key, val in values.items(): setattr(app, key + '_var', Var(val))
        app._log = Mock()
        return app

    def test_actual_label_launch_accepts_legacy_malformed_zero(self):
        app = self.app('0.00e+')
        with patch.object(kohya_gui.tk, 'StringVar', Var), patch.object(kohya_gui, 'LabelEditorWindow') as editor, contextlib.redirect_stderr(io.StringIO()):
            kohya_gui.App.cmd_label_editor(app)
        self.assertTrue(editor.called, app._log.call_args_list)
        self.assertEqual(editor.call_args.args[2]['te_lr'], 0.0)
        self.assertEqual(editor.call_args.args[2]['unet_lr'], 8.5e-5)

    def test_only_zero_mantissa_is_repaired_in_classic_parameter_collection(self):
        for value in ('0.00e+', '0.00e-', '-0.00E+', ' 0.00e+ '):
            with self.subTest(value=value), patch.object(kohya_gui.tk, 'StringVar', Var):
                self.assertEqual(kohya_gui.App._collect_params(self.app(value))['te_lr'], 0.0)
        for value in ('1.00e+', '0.10e+', 'bad'):
            with self.subTest(value=value), patch.object(kohya_gui.tk, 'StringVar', Var):
                with self.assertRaises(ValueError): kohya_gui.App._collect_params(self.app(value))

    def test_modern_training_repairs_saved_zero_without_mutating_project(self):
        core = SimpleNamespace(preset_for=lambda *args: {}, normalize_crop_ratio=lambda value: value)
        config = {'mode': 'anima_fz', 'base_type': 'anima', 'params': {'unet_lr': '8.50e-5', 'te_lr': '0.00e+'}}
        before = copy.deepcopy(config)
        params = training_params(core, config, 'test-project')
        self.assertEqual(params['te_lr'], '0')
        self.assertEqual(params['unet_lr'], '8.50e-5')
        self.assertEqual(config, before)

    def test_modern_training_does_not_silently_repair_nonzero_incomplete_exponents(self):
        core = SimpleNamespace(preset_for=lambda *args: {}, normalize_crop_ratio=lambda value: value)
        config = {'mode': 'anima_fz', 'params': {'unet_lr': '1.00e+'}}
        self.assertEqual(training_params(core, config, 'test-project')['unet_lr'], '1.00e+')

if __name__ == '__main__': unittest.main()