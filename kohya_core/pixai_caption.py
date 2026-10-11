"""Pinned PixAI anime tagger shared by training and standalone image tagging."""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import tempfile

from .captioning import CaptionError
from .diagnostics import redact
from .prompt_reverse_models import _manifest, model_root, status, download


def options(values=None):
    values = values or {}
    result = {}
    for category in ('character', 'copyright', 'style'):
        value = values.get('pixai_' + category, False)
        if type(value) is not bool:
            raise CaptionError('PixAI 标签类别需要为开关值。')
        result[category] = value
    try:
        threshold = float(values.get('pixai_threshold', .17))
    except (TypeError, ValueError):
        raise CaptionError('PixAI 内容标签阈值需要为 0.01 到 0.99。') from None
    if not math.isfinite(threshold) or not .01 <= threshold <= .99:
        raise CaptionError('PixAI 内容标签阈值需要为 0.01 到 0.99。')
    result['threshold'] = threshold
    return result


def _runtime(core, log, progress, device):
    cuda = device == 'auto' and core.detect_gpu_vendor() == 'nvidia'
    root = Path(core.data_sub('runtimes', 'prompt_reverse_pixai_cuda' if cuda else 'prompt_reverse_pixai_cpu')).resolve()
    python = root / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    root.parent.mkdir(parents=True, exist_ok=True)
    env = core.build_direct_env()
    if not python.is_file():
        base = core.find_python()[0]
        if not base or not Path(base).is_file():
            raise CaptionError('PixAI 需要 Python，请先在“环境准备”中安装 Python。')
        progress(0, 1, {'name': '创建独立 PixAI 环境'})
        if core.run_stream([base, '-m', 'venv', str(root)], cwd=str(root.parent), env=env, logf=log) != 0:
            core.check_stop()
            raise CaptionError('PixAI 独立环境创建失败，请查看运行日志。')
    core.check_stop()
    probe = ("import torch, torchvision, timm, transformers, safetensors, numpy; from PIL import Image; "
             "assert torch.__version__.split('+')[0]=='2.7.1'; "
             "assert torchvision.__version__.split('+')[0]=='0.22.1'; "
             "assert timm.__version__=='1.0.20'; assert transformers.__version__=='4.57.1'")
    if cuda:
        probe += '; assert torch.version.cuda is not None'
    if core.run_stream([str(python), '-c', probe], cwd=str(root), env=env, logf=lambda _: None) != 0:
        core.check_stop()
        progress(0, 1, {'name': '安装 PixAI 推理依赖（首次使用）'})
        log('[PixAI] 首次使用，准备独立%s环境…' % (' CUDA' if cuda else ' CPU'))
        args = [str(python), '-m', 'pip', 'install', '--no-input', '--timeout', '60', '--retries', '2']
        if cuda:
            try:
                core._preinstall_torch(str(python), str(root), logf=log, torch_ver='2.7.1',
                                       tv_ver='0.22.1', cu='cu128', label='PixAI')
            except core.StopRequested:
                raise
            except RuntimeError:
                core.check_stop()
                log('[PixAI] CUDA 依赖准备失败，改为独立 CPU 环境；可在日志中查看下载原因。')
                return _runtime(core, log, progress, 'cpu')
        elif core.run_stream(args + ['--index-url', 'https://download.pytorch.org/whl/cpu',
                                      'torch==2.7.1', 'torchvision==0.22.1'],
                             cwd=str(root), env=env, logf=log) != 0:
            core.check_stop()
            raise CaptionError('PixAI 的 CPU 依赖下载失败，请检查网络后重试。')
        installed = False
        for mirror in core.pip_mirror_urls(3):
            core.check_stop()
            if core.run_stream(args + ['--index-url', mirror, 'timm==1.0.20', 'transformers==4.57.1',
                                       'safetensors==0.6.2', 'numpy==2.2.6', 'pillow==11.3.0'],
                               cwd=str(root), env=env, logf=log) == 0:
                installed = True
                break
        core.check_stop()
        if not installed:
            raise CaptionError('PixAI 依赖安装失败，请查看日志后重试。')
    if core.run_stream([str(python), '-c', probe], cwd=str(root), env=env, logf=lambda _: None) != 0:
        core.check_stop()
        raise CaptionError('PixAI 推理依赖校验失败，请查看环境安装日志。')
    return str(python), root, env


def _job(core, selected, log, progress, device='auto', auto_download=False):
    if not status(core, 'pixai')['ready']:
        if not auto_download:
            raise CaptionError('请先下载 PixAI Tagger v1.0。')
        log('[PixAI] 首次训练，下载并校验官方模型（约 1.95 GB）…')
        download(core, log, progress, 'pixai')
    python, runtime, env = _runtime(core, log, progress, device)
    script = Path(core.KIT_DIR) / 'kohya_core' / 'pixai_worker.py'
    if not script.is_file():
        raise CaptionError('缺少 PixAI 打标程序，请使用完整安装包。')
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONIOENCODING='utf-8')
    return {'python': python, 'script': str(script), 'runtime': str(runtime),
            'model': str(model_root(core, 'pixai')), 'files': _manifest('pixai')['files'],
            'device': device, 'options': selected}, env


@contextmanager
def training_job(core, values, log):
    last = [None]
    def progress(done, total, item):
        percent = int(done * 100 / max(total, 1))
        key = (item['name'], percent)
        if key != last[0]:
            log('[PixAI] %s · %s%%' % (item['name'], percent))
            last[0] = key
    job, env = _job(core, options(values), log, progress, auto_download=True)
    with tempfile.TemporaryDirectory(prefix='kohya-pixai-training-') as temporary:
        path = Path(temporary) / 'job.json'
        path.write_text(json.dumps(job), encoding='utf-8')
        yield str(path), env


def generate(core, selection, report, log, progress, publish):
    job, env = _job(core, selection['pixai_options'], log, progress, selection.get('device', 'auto'))
    job['images'] = [str(path) for path in selection['images']]
    received = set()
    prefix = '[PIXAI_RESULT] '
    def output(line):
        if not str(line).startswith(prefix):
            log(redact(str(line)))
            return
        value = json.loads(str(line)[len(prefix):])
        index = value.pop('index', None)
        if (type(index) is not int or not 0 <= index < report['total'] or index in received
                or value.get('status') not in ('generated', 'failed')):
            raise CaptionError('PixAI 程序返回了无效结果。')
        if value['status'] == 'generated' and (not isinstance(value.get('caption'), str)
                or not value['caption'].strip() or len(value['caption']) > 8192):
            raise CaptionError('PixAI 程序返回了无效关键词。')
        received.add(index)
        image = selection['images'][index]
        value['name'] = image.relative_to(selection['root']).as_posix()
        report['items'].append(value)
        report['failed' if value['status'] == 'failed' else 'generated'] += 1
        publish(report)
        progress(len(received), report['total'], value)
    with tempfile.TemporaryDirectory(prefix='kohya-pixai-') as temporary:
        path = Path(temporary) / 'job.json'
        path.write_text(json.dumps(job), encoding='utf-8')
        # Isolated mode excludes the script directory, whose queue.py shadows the stdlib.
        code = core.run_stream([job['python'], '-I', '-u', job['script'], str(path)],
                              cwd=job['runtime'], env=env, logf=output)
        core.check_stop()
        if code != 0 or len(received) != report['total']:
            raise CaptionError('PixAI 打标未完成，请查看运行日志；已生成结果保留。')
    log('[PixAI] 反推完成，模型进程已退出。')
    return report
