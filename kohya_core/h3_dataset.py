"""H3 caption preparation without resizing or moving training media."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import time
import uuid

from .captioning import CaptionError, _atomic_bytes, _snapshot, save_caption
from .project_config import read_caption

TRAINING_TYPES = {"character": "人物", "style": "画风", "concept": "概念"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a"}


def _linked(path):
    return path.is_symlink() or getattr(path, "is_junction", lambda: False)()


def media_files(core, directory):
    root = Path(directory).resolve()
    if not str(directory or "").strip() or not root.is_dir():
        raise ValueError("请选择有效的 H3 混合媒体目录。")
    result, sidecars = [], set()
    def failed(exc):
        raise OSError("无法读取 H3 数据目录：%s" % exc.filename) from exc
    for folder, dirs, names in os.walk(root, onerror=failed):
        parent = Path(folder)
        dirs[:] = sorted(name for name in dirs if not name.startswith(".") and not _linked(parent / name))
        for name in sorted(names):
            path = parent / name
            ext = path.suffix.lower()
            kind = "images" if ext in core.FIZGIG_IMAGE_EXTS else "videos" if ext == ".mp4" else "audio" if ext in AUDIO_EXTS else None
            if not kind or _linked(path) or not path.is_file():
                continue
            caption = path.with_suffix(".txt")
            identity = str(caption).casefold()
            if identity in sidecars:
                raise ValueError("同名媒体共用一个 .txt，请先改名：%s" % path.relative_to(root).as_posix())
            sidecars.add(identity)
            if _linked(caption):
                raise ValueError("字幕文件是链接，请先改为本地文本：%s" % caption.relative_to(root).as_posix())
            result.append((path, kind))
    return root, result


def has_trigger(text, trigger):
    return not trigger or bool(re.search(r"(^|[\s,，])" + re.escape(trigger.strip()) + r"([\s,，.!?。！？]|$)", text, re.IGNORECASE))


def with_trigger(text, trigger):
    if not trigger or has_trigger(text, trigger):
        return text
    return trigger + ("\n" if "\n" in text else ", ") + text


def scan(core, directory, fallback_repeats=1, trigger=""):
    root, media = media_files(core, directory)
    summary = {"images": 0, "videos": 0, "audio": 0, "total": len(media),
               "missing_captions": [], "missing_images": [], "missing_videos": [], "missing_audio": [],
               "missing_trigger": [], "captioned": 0}
    folders = {}
    for path, kind in media:
        summary[kind] += 1
        folders[path.parent] = folders.get(path.parent, 0) + 1
        name = path.relative_to(root).as_posix()
        previous = _snapshot(path.with_suffix(".txt"))
        text = read_caption(path) if previous is not None else ""
        if not text:
            summary["missing_captions"].append(name)
            summary["missing_" + kind].append(name)
        else:
            summary["captioned"] += 1
            if not has_trigger(text, trigger):
                summary["missing_trigger"].append(name)
    subsets = []
    for folder, count in folders.items():
        match = re.match(r"^(\d+)[_-]", folder.name) if folder != root else None
        repeats = int(match[1]) if match else int(fallback_repeats)
        subsets.append((str(folder), max(1, repeats), count))
    return summary, subsets


def validate_options(params):
    training_type = str(params.get("at_sub_mode") or "character")
    if training_type not in TRAINING_TYPES:
        raise ValueError("H3 训练类型无效，请选择人物、画风或概念。")
    method = str(params.get("caption_method") or "wd14")
    if method not in ("wd14", "natural", "existing"):
        raise ValueError("H3 图片打标方式无效，请重新选择。")
    if training_type == "concept" and params.get("concept_type", "form") not in ("form", "outfit", "object", "bodypart"):
        raise ValueError("H3 概念类型无效，请重新选择。")
    return training_type, method


def _keyword_caption(text, params, training_type):
    # Keep existing captions out of keyword filters. Only generated tags reach here.
    from preprocess import filter_character_tags, filter_concept_tags
    if training_type == "style":
        text = filter_character_tags(text)
        extra = str(params.get("style_caption") or "").strip()
        if extra:
            text = ", ".join(filter(None, (extra, text)))
    elif training_type == "concept" and params.get("clean_concept", True):
        text, _removed = filter_concept_tags(text, params.get("concept_type") or "form")
    if not text.strip():
        raise CaptionError("标签过滤后为空，请调整打标选项或填写描述，不使用占位文本。")
    return text


def prepare(core, params, log=print, report_path=None):
    training_type, method = validate_options(params)
    root, media = media_files(core, params.get("raw_dir") or params.get("train_data_dir") or "")
    if not media:
        raise ValueError("目录中没有可训练的图片、MP4 视频或音频样本。")
    replace = method == "wd14" and bool(params.get("h3_caption_replace", False))
    trigger = str(params.get("trigger") or "").strip()
    sync = bool(params.get("h3_sync_trigger", True))
    backup_root = root / ".caption_backups"
    if _linked(backup_root):
        raise CaptionError("字幕备份目录是链接，请先检查 .caption_backups。")
    backup_root /= "h3_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    report = {"status": "running", "directory": str(root), "training_type": training_type,
              "caption_method": method, "generated": 0, "preserved": 0, "trigger_synced": 0,
              "failed": 0, "backup_directory": str(backup_root), "items": []}
    pending, snapshots, signatures = [], {}, {}
    log("[H3 数据准备] 训练类型：%s；只处理文本，媒体保持原文件。" % TRAINING_TYPES[training_type])
    try:
        for path, kind in media:
            core.check_stop()
            name = path.relative_to(root).as_posix()
            previous = _snapshot(path.with_suffix(".txt"))
            text = read_caption(path) if previous is not None else ""
            if kind == "images" and method == "wd14" and (not text or replace):
                pending.append(path)
                snapshots[path] = previous
                stat = path.stat()
                signatures[path] = [stat.st_size, stat.st_mtime_ns]
            elif text:
                updated = with_trigger(text, trigger) if sync else text
                if updated != text:
                    core.check_stop()
                    save_caption(path, updated, previous, backup_root / path.relative_to(root).parent)
                    report["trigger_synced"] += 1
                report["preserved"] += 1
                report["items"].append({"name": name, "status": "preserved", "caption": updated})
        if pending:
            from .prompt_reverse import _keywords
            model = str(params.get("wd14_model") or "swinv2-v3")
            log("[H3 图片打标] %s 张；%s；%s。" % (len(pending), model, "重新生成，原文本备份" if replace else "只补缺失，已有内容保留"))
            def write_item(item):
                path = root / item["name"]
                if path not in snapshots:
                    raise CaptionError("打标结果不属于本次 H3 图集。")
                if item.get("status") == "failed":
                    report["failed"] += 1
                    report["items"].append(item)
                    log("[H3 打标失败] %s：%s" % (item["name"], item.get("error") or "未生成有效标签"))
                    return
                core.check_stop()
                stat = path.stat()
                if signatures[path] != [stat.st_size, stat.st_mtime_ns]:
                    raise CaptionError("打标期间图片已修改，未写入标签：%s" % item["name"])
                text = _keyword_caption(item["caption"], params, training_type)
                if sync:
                    text = with_trigger(text, trigger)
                save_caption(path, text, snapshots[path], backup_root / path.relative_to(root).parent)
                report["generated"] += 1
                report["items"].append({"name": item["name"], "status": "generated", "caption": text})
                log("[H3 图片打标] %s/%s · %s" % (report["generated"] + report["failed"], len(pending), item["name"]))
            # Bound temporary images and model results for large nested datasets.
            for offset in range(0, len(pending), 200):
                batch = pending[offset:offset + 200]
                selection = {"root": root, "images": batch, "threshold": .35, "wd14_model": model,
                             "device": "auto"}
                if model == "pixai-v1":
                    from .pixai_caption import generate, options
                    selection["pixai_options"] = options(params)
                    local_report = {"total": len(batch), "generated": 0, "failed": 0, "items": []}
                    written = set()
                    def publish(value):
                        for item in value["items"]:
                            if item["name"] not in written:
                                write_item(item)
                                written.add(item["name"])
                    generate(core, selection, local_report, log, lambda *_args: None, publish, auto_download=True)
                else:
                    if model not in ("swinv2-v3", "moat-v2"):
                        raise CaptionError("H3 自动打标模型无效，请重新选择。")
                    results = _keywords(core, selection, log, lambda *_args: None)
                    for path in batch:
                        write_item({"name": path.relative_to(root).as_posix(), **results[path]})
        summary, _ = scan(core, str(root), trigger=trigger)
        report["summary"] = summary
        log("[H3 数据准备] 图片 %s，视频 %s，音频 %s；新打标 %s，保留 %s，同步触发词 %s。" % (
            summary["images"], summary["videos"], summary["audio"], report["generated"], report["preserved"], report["trigger_synced"]))
        if report["trigger_synced"] or replace:
            log("[H3 数据准备] 被修改的原文本备份：%s" % backup_root)
        if summary["missing_captions"] or report["failed"]:
            hints = []
            if summary["missing_images"]: hints.append("图片缺描述：选择自动打标，或先批量生成自然语言描述")
            if summary["missing_videos"]: hints.append("视频缺描述：点击视频 AI 描述")
            if summary["missing_audio"]: hints.append("音频缺描述：准备同名 .txt")
            raise CaptionError("H3 文本未齐全：%s。缺失样本：%s" % ("；".join(hints), "、".join(summary["missing_captions"][:8])))
        if summary["missing_trigger"]:
            log("[H3 提醒] %s 个文本没有项目触发词；同步触发词目前关闭。" % len(summary["missing_trigger"]))
        report["status"] = "completed"
        return report
    except BaseException:
        report["status"] = "incomplete"
        raise
    finally:
        if report_path:
            try:
                _atomic_bytes(Path(report_path), json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"))
            except OSError as exc:
                # The report is supplementary; a Windows rename/replace denial
                # must not discard a successfully prepared dataset or mask the
                # original preparation error.
                log("[H3 数据准备] 报告保存失败（不影响数据准备结果）：%s" % exc)
