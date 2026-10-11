"""Pinned local caption model assets, downloads and integrity records."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .captioning import CaptionError, _atomic_bytes


def _manifest(model='ram'):
    names = {'ram': 'ram_assets.json', 'qwen_vl': 'qwen_vl_assets.json', 'pixai': 'pixai_assets.json'}
    if model not in names:
        raise CaptionError('本地反推模型选项无效。')
    return json.loads(Path(__file__).with_name(names[model]).read_text(encoding='utf-8'))


def model_root(core, model='ram'):
    manifest = _manifest(model)
    return Path(core.data_sub('models', 'prompt_reverse', manifest.get('directory', 'ram-plus-swin-large-384'))).resolve()


def _signature(root, manifest):
    result = {}
    for name, meta in manifest['files'].items():
        path = root / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != meta['size']:
            return None
        stat = path.stat()
        result[name] = [stat.st_size, stat.st_mtime_ns]
    return result


def status(core, model='ram'):
    root, manifest = model_root(core, model), _manifest(model)
    ready = False
    try:
        record = json.loads((root / 'verified.json').read_text(encoding='utf-8'))
        signatures = _signature(root, manifest)
        ready = bool(signatures and record == {'commit': manifest['commit'], 'files': signatures})
    except (OSError, ValueError):
        pass
    return {'ok': True, 'ready': ready, 'name': manifest.get('name', 'RAM++'), 'model': model, 'directory': str(root),
            'size_bytes': sum(file['size'] for file in manifest['files'].values()), 'language': 'en' if model == 'pixai' else 'zh/en'}


def _valid(path, meta, stop):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != meta['size']:
        return False
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            stop()
            digest.update(chunk)
    return digest.hexdigest() == meta['sha256']


def download(core, log, progress, model='ram'):
    root, manifest = model_root(core, model), _manifest(model)
    root.mkdir(parents=True, exist_ok=True)
    total = sum(meta['size'] for meta in manifest['files'].values())
    completed = 0
    for name, meta in manifest['files'].items():
        core.check_stop()
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if _valid(destination, meta, core.check_stop):
            completed += meta['size']
            progress(completed, total, {'name': name})
            continue
        partial = root / (name + '.download')
        if partial.is_symlink():
            raise CaptionError('反推模型下载临时文件无效，请检查模型目录。')
        def advance(done, _total):
            progress(completed + min(max(done, 0), meta['size']), total, {'name': '下载 ' + name})
        success = False
        urls = meta.get('urls') or ['%s/%s/resolve/%s/%s' % (host, manifest['repo'], manifest['commit'], name)
                                    for host in ('https://hf-mirror.com', 'https://huggingface.co')]
        for url in urls:
            core.check_stop()
            log('[反推模型] 下载 ' + name)
            got = core._download_with_resume(url, str(partial), log, progress_cb=advance, direct=True,
                                             quick_fail=meta['size'] < 16 * 1024 * 1024)
            core.check_stop()
            progress(completed, total, {'name': '校验 ' + name})
            if got and _valid(partial, meta, core.check_stop):
                if destination.is_symlink():
                    raise CaptionError('反推模型文件为链接，请检查模型目录。')
                os.replace(partial, destination)
                success = True
                break
            # Retain incomplete data for retry; reject complete but corrupt files.
            if partial.is_file() and partial.stat().st_size >= meta['size']:
                partial.unlink()
        if not success:
            raise CaptionError('反推模型文件下载或校验失败：%s。请检查网络后重试，已下载文件保留。' % name)
        completed += meta['size']
        progress(completed, total, {'name': name})
    core.check_stop()
    _atomic_bytes(root / 'verified.json', json.dumps({'commit': manifest['commit'], 'files': _signature(root, manifest)}).encode('utf-8'))
    log('[反推模型] %s 已下载并校验，可离线使用。' % manifest.get('name', 'RAM++'))


