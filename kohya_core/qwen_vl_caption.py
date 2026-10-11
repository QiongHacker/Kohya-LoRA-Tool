"""Managed Qwen3-VL inference using a task-owned, loopback llama.cpp process."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path, PurePosixPath
import platform
import socket
import stat
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
import zipfile

from .captioning import CaptionError, CaptionServiceError, describe_image
from .diagnostics import redact
from .prompt_reverse_models import _manifest, _valid, model_root, status
from .prompt_reverse import reverse_prompt


def _extract(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if len(members) > 500 or sum(item.file_size for item in members) > 512 * 1024 * 1024:
            raise CaptionError('本地视觉运行组件的归档大小无效。')
        for item in members:
            name = PurePosixPath(item.filename)
            if (name.is_absolute() or '\\' in item.filename or ':' in item.filename or '..' in name.parts
                    or stat.S_ISLNK(item.external_attr >> 16)):
                raise CaptionError('本地视觉运行组件包含无效文件路径。')
            target = destination.joinpath(*name.parts).resolve()
            if not target.is_relative_to(root):
                raise CaptionError('本地视觉运行组件路径越界。')
        bundle.extractall(destination)
    servers = list(destination.rglob('llama-server.exe'))
    if len(servers) != 1:
        raise CaptionError('本地视觉运行组件缺少唯一的启动程序，请重新下载。')
    return servers[0]


def _health(url, key):
    request = urllib.request.Request(url + '/health', headers={'Authorization': 'Bearer ' + key})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=1) as response:
        value = json.loads(response.read(4096))
    return isinstance(value, dict) and value.get('status') == 'ok'


def _stop_process(process, reader):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    if reader:
        reader.join(timeout=2)
    if process.stdout:
        process.stdout.close()


@contextmanager
def _server(core, executable, model, projector, backend, workspace, log, progress):
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    key = uuid.uuid4().hex + uuid.uuid4().hex
    key_file = workspace / ('auth-' + uuid.uuid4().hex + '.txt')
    key_file.write_text(key + '\n', encoding='utf-8')
    env = {name: value for name, value in core.build_direct_env().items()
           if not name.startswith('LLAMA_ARG_') and name != 'LLAMA_API_KEY'}
    env.update(HF_HUB_OFFLINE='1', LLAMA_LOG_COLORS='0')
    command = [str(executable), '-m', str(model), '--mmproj', str(projector), '--host', '127.0.0.1',
               '--port', str(port), '--api-key-file', str(key_file), '--alias', 'kohya-photo-qwen',
               '--ctx-size', '4096', '--parallel', '1', '--batch-size', '512', '--ubatch-size', '128',
               '--image-max-tokens', '1024', '--jinja', '--no-webui', '--n-gpu-layers',
               'auto' if backend == 'vulkan' else '0']
    if backend == 'vulkan':
        command += ['--fit', 'on', '--fit-target', '768']
    else:
        command += ['--no-mmproj-offload', '--threads', str(max(1, min(os.cpu_count() or 4, 8)))]
    process, reader = None, None
    url = 'http://127.0.0.1:%d' % port
    try:
        log('[反推] 正在加载 Qwen3-VL 4B（%s）…' % ('自动加速 / Vulkan' if backend == 'vulkan' else 'CPU'))
        progress(0, 1, {'name': '加载 Qwen3-VL 4B（%s）' % backend})
        process = subprocess.Popen(command, cwd=str(executable.parent), env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace',
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        def read_output():
            try:
                for line in process.stdout:
                    text = redact(line.rstrip()).replace(key, '<REDACTED>')
                    log('[本地视觉] ' + text)
            except (OSError, ValueError):
                pass
        reader = threading.Thread(target=read_output, daemon=True, name='local-vision-log')
        reader.start()
        deadline = time.monotonic() + 240
        while True:
            core.check_stop()
            if process.poll() is not None:
                raise CaptionError('本地视觉模型加载失败（退出码 %s）。请查看运行日志。' % process.returncode)
            try:
                if _health(url, key):
                    break
            except (urllib.error.URLError, TimeoutError, OSError, ValueError):
                pass
            if time.monotonic() >= deadline:
                raise CaptionError('本地视觉模型加载超时，请关闭占用内存的程序或改用 CPU。')
            time.sleep(.2)
        core.check_stop()
        yield {'provider': 'compatible', 'base_url': url + '/v1', 'model': 'kohya-photo-qwen',
               'api_key': key, 'local': True}, process
    except OSError as exc:
        raise CaptionError('本地视觉运行组件无法启动（%s），请检查完整安装包和系统运行库。' % type(exc).__name__) from exc
    finally:
        try:
            if process is not None:
                _stop_process(process, reader)
        finally:
            key_file.unlink(missing_ok=True)


def _description_prompt(language, length):
    prompt = reverse_prompt(language, length)
    if length == 'detailed':
        prompt += ('优先描述可见的颜色、服饰材质、镜头视角、主体之间的空间关系以及背景层次；'
                   '看不清的细节直接省略，不将真实照片默认描述成插画。' if language == 'zh' else
                   ' Prioritize visible colors, clothing materials, viewpoint, spatial relationships and background depth. '
                   'Omit unclear details. Do not describe a photograph as an illustration without visual evidence.')
    return prompt


def _caption_images(core, selection, service, process, report, log, progress, publish):
    failures = 0
    for done, image in enumerate(selection['images'], 1):
        core.check_stop()
        if process.poll() is not None:
            raise CaptionError('本地视觉进程已退出；已生成的结果保留。')
        item = {'name': image.relative_to(selection['root']).as_posix()}
        signature = None
        try:
            signature = [image.stat().st_size, image.stat().st_mtime_ns]
            progress(done - 1, report['total'], {'name': '描述 ' + item['name']})
            caption = describe_image(service, image, selection['language'], selection['length'], core.check_stop,
                                     prompt=_description_prompt(selection['language'], selection['length']), timeout=360)
            core.check_stop()
            if signature != [image.stat().st_size, image.stat().st_mtime_ns]:
                raise CaptionError('反推期间图片已修改，请重新生成。')
            item.update(status='generated', caption=caption)
            failures = 0
        except CaptionError as exc:
            item.update(status='failed', error=str(exc))
            if isinstance(exc, CaptionServiceError):
                failures += 1
        except (OSError, ValueError):
            item.update(status='failed', error='无法读取这张图片，请检查文件。')
        item['image_signature'] = signature
        report['items'].append(item)
        report['failed' if item['status'] == 'failed' else 'generated'] += 1
        publish(report)
        progress(done, report['total'], item)
        if failures >= 3:
            report['error'] = '本地视觉模型连续失败 3 次，已停止；请查看运行日志后重试。'
            break


def generate(core, selection, report, log, progress, publish):
    if os.name != 'nt' or platform.machine().lower() not in ('amd64', 'x86_64'):
        raise CaptionError('内置 Qwen3-VL 运行组件目前支持 Windows x64；其他平台请使用视觉服务。')
    if not status(core, 'qwen_vl')['ready']:
        raise CaptionError('请先下载 Qwen3-VL 反推模型和运行组件。')
    root, manifest = model_root(core, 'qwen_vl'), _manifest('qwen_vl')
    progress(0, 1, {'name': '校验 Qwen3-VL 模型与运行组件'})
    if not all(_valid(root / name, meta, core.check_stop) for name, meta in manifest['files'].items()):
        raise CaptionError('本地视觉模型或运行组件已损坏，请重新下载修复。')
    # The official verified archives are extracted per task; modified installed binaries are never reused.
    with tempfile.TemporaryDirectory(prefix='kohya-qwen-vl-') as temporary:
        workspace = Path(temporary)
        version = manifest['runtime_version']
        options = ('vulkan', 'cpu') if selection.get('device', 'auto') == 'auto' else ('cpu',)
        for backend in options:
            core.check_stop()
            executable = _extract(root / ('runtime/llama-%s-bin-win-%s-x64.zip' % (version, backend)), workspace / backend)
            try:
                with _server(core, executable, root / manifest['text_model'], root / manifest['projector'],
                             backend, workspace, log, progress) as (service, process):
                    _caption_images(core, selection, service, process, report, log, progress, publish)
                break
            except CaptionError:
                # Only fall back before any image results have been published.
                if backend != 'vulkan' or report['items']:
                    raise
                log('[反推] 自动加速启动不可用，改用 CPU 加载同一模型。')
        core.check_stop()
    log('[反推] 本地 Qwen3-VL 任务已结束，推理进程和显存已释放。')
    return report
