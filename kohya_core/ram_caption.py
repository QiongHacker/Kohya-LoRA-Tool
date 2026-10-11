"""Task-owned, offline RAM++ image tagging in an isolated Python environment."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from .captioning import CaptionError
from .diagnostics import redact
from .prompt_reverse_models import _manifest, _valid, model_root, status


def _runtime(core, log, progress, device):
    cuda = device == 'auto' and core.detect_gpu_vendor() == 'nvidia'
    root = Path(core.data_sub('runtimes', 'prompt_reverse_ram_cuda' if cuda else 'prompt_reverse_ram_cpu')).resolve()
    python = root / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    root.parent.mkdir(parents=True, exist_ok=True)
    env = core.build_direct_env()
    if not python.is_file():
        base = core.find_python()[0]
        if not base or not Path(base).is_file():
            raise CaptionError('RAM++ 需要 Python，请先在“环境准备”中安装 Python。')
        progress(0, 1, {'name': '创建独立 RAM++ 环境'})
        if core.run_stream([base, '-m', 'venv', str(root)], cwd=str(root.parent), env=env, logf=log) != 0:
            core.check_stop()
            raise CaptionError('RAM++ 独立环境创建失败，请查看运行日志。')
    core.check_stop()
    probe = ("import torch, torchvision, timm, transformers, scipy, fairscale, numpy; from PIL import Image; "
             "assert torch.__version__.split('+')[0]=='2.7.1'; "
             "assert torchvision.__version__.split('+')[0]=='0.22.1'; "
             "assert timm.__version__=='0.4.12'; assert transformers.__version__=='4.40.2'")
    if cuda:
        probe += '; assert torch.version.cuda is not None'
    if core.run_stream([str(python), '-c', probe], cwd=str(root), env=env, logf=lambda _: None) != 0:
        core.check_stop()
        progress(0, 1, {'name': '安装 RAM++ 推理依赖（首次使用）'})
        log('[RAM++] 首次使用，准备独立%s环境…' % (' CUDA' if cuda else ' CPU'))
        args = [str(python), '-m', 'pip', 'install', '--no-input', '--timeout', '60', '--retries', '2']
        if cuda:
            try:
                core._preinstall_torch(str(python), str(root), logf=log, torch_ver='2.7.1',
                                       tv_ver='0.22.1', cu='cu128', label='RAM++')
            except core.StopRequested:
                raise
            except RuntimeError:
                core.check_stop()
                log('[RAM++] CUDA 依赖准备失败，改为独立 CPU 环境；可在日志中查看下载原因。')
                return _runtime(core, log, progress, 'cpu')
        elif core.run_stream(args + ['--index-url', 'https://download.pytorch.org/whl/cpu',
                                      'torch==2.7.1', 'torchvision==0.22.1'],
                             cwd=str(root), env=env, logf=log) != 0:
            core.check_stop()
            raise CaptionError('RAM++ 的 CPU 依赖下载失败，请检查网络后重试。')
        installed = False
        for mirror in core.pip_mirror_urls(3):
            core.check_stop()
            if core.run_stream(args + ['--index-url', mirror, 'timm==0.4.12', 'transformers==4.40.2',
                                       'fairscale==0.4.13', 'scipy==1.15.3', 'numpy==2.2.6', 'pillow==11.3.0'],
                               cwd=str(root), env=env, logf=log) == 0:
                installed = True
                break
        core.check_stop()
        if not installed:
            raise CaptionError('RAM++ 依赖安装失败，请查看日志后重试。')
    return str(python), root, env


def generate(core, selection, report, log, progress, publish):
    if not status(core, 'ram')['ready']:
        raise CaptionError('请先下载 RAM++ 关键词模型。')
    progress(0, 1, {'name': '校验 RAM++ 模型与组件'})
    root, manifest = model_root(core, 'ram'), _manifest('ram')
    if not all(_valid(root / name, meta, core.check_stop) for name, meta in manifest['files'].items()):
        raise CaptionError('RAM++ 模型或组件文件已变化，请重新下载修复。')
    python, runtime, env = _runtime(core, log, progress, selection.get('device', 'auto'))
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONIOENCODING='utf-8')
    script = Path(core.KIT_DIR) / 'kohya_core' / 'ram_worker.py'
    if not script.is_file():
        raise CaptionError('缺少 RAM++ 打标程序，请使用完整安装包。')
    progress(0, report['total'], {'name': '加载 RAM++ 关键词模型'})
    with tempfile.TemporaryDirectory(prefix='kohya-ram-') as temporary:
        job = Path(temporary) / 'job.json'
        job.write_text(json.dumps({'model': str(root), 'checkpoint': manifest['checkpoint'],
                                   'images': [str(path) for path in selection['images']],
                                   'language': selection['language'], 'threshold': selection['threshold'],
                                   'device': selection.get('device', 'auto')}), encoding='utf-8')
        received = set()
        prefix = '[RAM_RESULT] '
        def output(line):
            if not str(line).startswith(prefix):
                log(redact(str(line)))
                return
            value = json.loads(str(line)[len(prefix):])
            index = value.pop('index', None)
            if (type(index) is not int or not 0 <= index < report['total'] or index in received
                    or value.get('status') not in ('generated', 'failed')):
                raise CaptionError('RAM++ 程序返回了无效结果。')
            if value['status'] == 'generated' and (not isinstance(value.get('caption'), str)
                                                   or not value['caption'].strip() or len(value['caption']) > 8192):
                raise CaptionError('RAM++ 程序返回了无效关键词。')
            received.add(index)
            image = selection['images'][index]
            value['name'] = image.relative_to(selection['root']).as_posix()
            report['items'].append(value)
            report['failed' if value['status'] == 'failed' else 'generated'] += 1
            publish(report)
            progress(len(received), report['total'], value)
        # Keep application modules out of third-party imports (notably stdlib queue).
        code = core.run_stream([python, '-I', '-u', str(script), str(job)], cwd=str(runtime), env=env, logf=output)
        core.check_stop()
        if code != 0 or len(received) != report['total']:
            raise CaptionError('RAM++ 打标未完成，请查看运行日志；已生成结果保留。')
    log('[RAM++] 打标已结束，模型进程已退出。')
    return report
