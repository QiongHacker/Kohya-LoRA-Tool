"""Desktop bridge for the local LoRA workbench; jobs share the training task lock."""
from __future__ import annotations
import base64
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import threading
import uuid

from kohya_core.fizgig_engine import runtime, source_version, VERSION, SUPPORTED_V7, _env, _write, _sha
from kohya_core.fizgig_adapter import models, _helper_configs, validate_base
from kohya_core.diagnostics import redact

FAMILIES = {"krea2": "Krea 2", "anima": "Anima"}


def number(value, low, high, integer=False):
    n = float(value)
    if not math.isfinite(n) or not low <= n <= high or (integer and n != int(n)):
        raise ValueError("预览参数超出允许范围。")
    return int(n) if integer else n


class LoRAWorkbenchBridge:
    def _workbench_root(self):
        return Path(self.core.data_sub("cache", "lora_workbench"))

    def _workbench_settings(self):
        return Path(self.core.data_sub("settings")) / "lora_workbench.json"

    def _workbench_presets_path(self):
        return Path(self.core.data_sub("settings")) / "lora_workbench_presets.json"

    def _workbench_presets(self):
        path = self._workbench_presets_path()
        if not path.is_file():
            return []
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("预设文件过大，请检查用户数据目录中的工作台预设文件。")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("schema") != 1 or not isinstance(data.get("presets"), list):
            raise ValueError("预设文件格式无效，未修改原文件。")
        for item in data["presets"]:
            if not isinstance(item, dict) or item.get("family") not in FAMILIES or not isinstance(item.get("blocks"), dict):
                raise ValueError("预设内容无效，未修改原文件。")
            if not isinstance(item.get("id"), str) or not isinstance(item.get("block_ids"), list) or not all(isinstance(b, str) for b in item["block_ids"]):
                raise ValueError("预设层级结构无效。")
            if uuid.UUID(item["id"]).hex != item["id"] or set(item["blocks"]) != set(item["block_ids"]):
                raise ValueError("预设层级结构无效。")
            for value in item["blocks"].values(): number(value, -3, 3)
        return data["presets"]

    def get_lora_workbench_presets(self):
        try:
            with self._task_lock:
                return {"ok": True, "presets": self._workbench_presets()}
        except (OSError, ValueError, TypeError, KeyError) as exc:
            return {"ok": False, "error": redact(str(exc))}

    def save_lora_workbench_preset(self, name, task_id, blocks):
        guard = self._agent_guard()
        if guard: return guard
        try:
            name = str(name).strip()
            if not name or len(name) > 80 or any(ord(c) < 32 for c in name):
                raise ValueError("预设名称需为 1 至 80 个字符。")
            task_id = uuid.UUID(str(task_id)).hex
            task = self.get_task_status(task_id)
            if task.get("ok") and task.get("status") != "completed":
                raise ValueError("请先完成工作台任务。")
            report = json.loads((self._workbench_root() / task_id / "result.json").read_text(encoding="utf-8"))
            if not report.get("ok"): raise ValueError("请先读取 LoRA。")
            info = report["info"]
            ids = sorted(b["id"] for g in info["groups"] for b in g["blocks"])
            if info["family"] not in FAMILIES or not isinstance(blocks, dict) or set(blocks) != set(ids):
                raise ValueError("预设与当前 LoRA 的层级不匹配，请重新读取。")
            if _sha(info["source"]) != info["fingerprint"]:
                raise ValueError("LoRA 文件已发生变化，请重新读取。")
            item = {"id": uuid.uuid4().hex, "name": name, "family": info["family"], "block_ids": ids,
                    "fingerprint": info["fingerprint"], "source_name": info["name"], "kernel": "Fizgig " + runtime(self.core)["version"],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "blocks": {key: number(value, -3, 3) for key, value in blocks.items()}}
            with self._task_lock:
                presets = self._workbench_presets()
                if len(presets) >= 100: raise ValueError("已保存 100 个预设，请先删除不需要的预设。")
                presets.insert(0, item)
                _write(self._workbench_presets_path(), {"schema": 1, "presets": presets})
            return {"ok": True, "presets": presets, "id": item["id"]}
        except (OSError, ValueError, TypeError, KeyError) as exc:
            return {"ok": False, "error": redact(str(exc))}

    def delete_lora_workbench_preset(self, preset_id):
        guard = self._agent_guard()
        if guard: return guard
        try:
            preset_id = uuid.UUID(str(preset_id)).hex
            with self._task_lock:
                presets = self._workbench_presets()
                remaining = [p for p in presets if p["id"] != preset_id]
                if len(remaining) == len(presets): raise ValueError("预设不存在，请重新打开工作台。")
                _write(self._workbench_presets_path(), {"schema": 1, "presets": remaining})
            return {"ok": True, "presets": remaining}
        except (OSError, ValueError, TypeError, KeyError) as exc:
            return {"ok": False, "error": redact(str(exc))}

    def use_lora_workbench_profile_baseline(self, task_id, name, fingerprint, explore=None):
        """Adopt the complete condition behind a Profiler image, including legacy reports."""
        guard = self._agent_guard()
        if guard: return guard
        try:
            task_id = uuid.UUID(str(task_id)).hex
            result = self.get_lora_workbench_result(task_id)
            report = result.get("report") or {}
            if not result.get("ok") or not report.get("ok") or report.get("action") != "profile":
                raise ValueError("请先完成 Profiler 分析。")
            info, profile = report["info"], report["profile"]
            if info["family"] not in FAMILIES or not info.get("can_preview") or profile.get("scope") != "original":
                raise ValueError("此分析记录暂不支持设为探索基准。")
            if info["fingerprint"] != fingerprint or _sha(info["source"]) != fingerprint:
                raise ValueError("LoRA 文件已改变，请重新读取并分析。")
            ids = {b["id"] for group in info["groups"] for b in group["blocks"]}
            choices = {profile["baseline"]["name"]: (profile["baseline"], "原版 LoRA", ids)}
            for window in profile["windows"]:
                keep = set(window["blocks"])
                if not keep or keep - ids: raise ValueError("分析记录中的层级结构无效。")
                choices[window["alone"]["name"]] = (window["alone"], "仅保留 " + window["label"], keep)
                choices[window["removed"]["name"]] = (window["removed"], "关闭 " + window["label"], ids - keep)
            if name not in choices: raise ValueError("请选择原版、仅保留本组或关闭本组的分析图。")
            picture, label, enabled = choices[name]
            # Older reports contain condition names/group IDs but no explicit
            # vector. Those conditions are whole / keep-window / drop-window.
            values = {bid: 1.0 if bid in enabled else 0.0 for bid in sorted(ids)}
            if "blocks" in picture:
                recorded = picture["blocks"]
                if not isinstance(recorded, dict) or set(recorded) != ids:
                    raise ValueError("图片对应的层级配置不完整。")
                recorded = {bid: number(value, 0, 1) for bid, value in recorded.items()}
                if recorded != values: raise ValueError("图片对应的层级配置与分析条件不一致。")
                values = recorded
            image = self.get_lora_workbench_image(task_id, name)
            if not image.get("ok"): raise ValueError("所选基准图无法读取，请重新分析。")
            config = explore or {}
            if not isinstance(config, dict): raise ValueError("变体探索配置无效。")
            frozen = config.get("frozen") or []
            if not isinstance(frozen, list) or any(not isinstance(bid, str) for bid in frozen) or set(frozen) - ids:
                raise ValueError("冻结层与当前 LoRA 不匹配。")
            exploration = {"intensity": number(config.get("intensity", 0.2), 0, 1),
                           "structure": number(config.get("structure", 0), 0, 1),
                           "mutations": number(config.get("mutations", 3), 1, 64, True),
                           "mutation_seed": number(config.get("mutation_seed", 42), 0, 2**32 - 1, True),
                           "frozen": sorted(set(frozen)), "last_pick": [],
                           "state": {bid: {"strength": 1.0, "enabled": bid in enabled} for bid in values}}
            # Profiler disables INT8 attention during its measurements. Preserve
            # that rendering mode when moving a measured image into Explorer.
            recipe = dict(report["recipe"], exact_attention=True)
            baseline = {"task_id": task_id, "name": name, "label": label, "fingerprint": fingerprint,
                        "blocks": values, "recipe": recipe}
            with self._task_lock:
                if self._task and self._task.get("status") == "running":
                    raise ValueError("已有任务正在运行，请先完成或停止。")
                try:
                    saved = json.loads(self._workbench_settings().read_text(encoding="utf-8"))
                except (OSError, ValueError): saved = {}
                if not isinstance(saved, dict) or saved.get("source") != info["source"] or saved.get("family") != info["family"]: saved = {}
                _write(self._workbench_settings(), dict(saved, source=info["source"], family=info["family"], fingerprint=fingerprint,
                       blocks=values, preview=recipe, profile={"class_prompt": profile.get("class_prompt", ""), "references": profile.get("references", [])},
                       explore=exploration, profile_explore_baseline=baseline, last_task=task_id, last_profile_task=task_id))
            return {"ok": True, "baseline": baseline, "explore": exploration, "data_url": image["data_url"]}
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
            return {"ok": False, "error": redact(str(exc))}

    def _workbench_record(self):
        record = runtime(self.core)
        if not Path(record["python"]).is_file() or source_version(record["source"]) not in SUPPORTED_V7:
            raise ValueError("请先准备新版 Fizgig，再使用 LoRA 工作台。")
        for name in ("families/workbench.py", "families/extract.py", "families/lorafile.py", "repair_studio/state.py", "repair_studio/bake.py", "families/block_profile.py"):
            if not (Path(record["source"]) / "src" / "fizgig" / name).is_file():
                raise ValueError("Fizgig 安装缺少工作台组件，请重新准备引擎。")
        return record

    def get_lora_workbench(self, project_name="", family=""):
        try:
            try:
                saved = json.loads(self._workbench_settings().read_text(encoding="utf-8")) if self._workbench_settings().is_file() else {}
            except (OSError, ValueError):
                saved = {}
            if not isinstance(saved, dict): saved = {}
            if project_name and (not isinstance(project_name, str) or self.core._sanitize_dirname(project_name) != project_name):
                raise ValueError("项目名称无效。")
            config = self.core.load_project(project_name) or {} if project_name else {}
            if not isinstance(config, dict): config = {}
            params = config.get("params") or {}
            if not isinstance(params, dict): params = {}
            selected = family if family in FAMILIES else params.get("base_type", "anima")
            if selected not in FAMILIES: selected = "anima"
            if params.get("base_type") != selected: params = {}
            defaults = models(self.core, selected, params)
            recent = []
            if project_name:
                folder = Path(self.core.data_sub("output", project_name))
                # Ignore optimiser state/snapshots and limit depth/file count.
                for root, dirs, files in os.walk(folder):
                    depth = len(Path(root).relative_to(folder).parts)
                    dirs[:] = [d for d in dirs if d not in ("snapshots", "cache") and not d.endswith("-state")] if depth < 3 else []
                    for name in files:
                        if name.lower().endswith(".safetensors"):
                            path = Path(root) / name
                            recent.append({"name": str(path.relative_to(folder)), "path": str(path), "mtime": path.stat().st_mtime})
                    if len(recent) >= 200: break
                recent = sorted(recent, key=lambda item: item["mtime"], reverse=True)[:40]
            try:
                self._workbench_record()
                ready, reason = True, ""
            except ValueError as exc:
                ready, reason = False, str(exc)
            with self._task_lock:
                task = self._task
                active = {k: task.get(k) for k in ("id", "status", "message", "progress")} if task and task.get("kind") == "lora_workbench" else None
            return {"ok": True, "ready": ready, "reason": reason, "families": FAMILIES, "saved": saved,
                    "defaults": {k: defaults.get(k) or "" for k in ("dit", "vae", "te")}, "family": selected,
                    "recent": recent, "task": active}
        except (OSError, ValueError, TypeError) as exc:
            return {"ok": False, "error": redact(str(exc))}

    def start_lora_workbench(self, action, options):
        guard = self._agent_guard()
        if guard: return guard
        staged = None
        try:
            if action not in ("inspect", "analyze", "profile", "preview", "explore", "export", "compress") or not isinstance(options, dict):
                raise ValueError("工作台操作无效。")
            record = self._workbench_record()
            source = Path(str(options.get("source") or "")).resolve()
            if not source.is_file() or source.suffix.lower() != ".safetensors":
                raise ValueError("请选择已有的 safetensors LoRA 文件。")
            family = str(options.get("family") or "auto")
            if family not in (*FAMILIES, "auto"):
                raise ValueError("模型架构无效。")
            values = options.get("blocks") or {}
            if not isinstance(values, dict) or len(values) > 1024:
                raise ValueError("分层配置无效。")
            values = {str(key): number(value, -3, 3) for key, value in values.items()}
            selection = {"source": str(source), "family": family, "fingerprint": str(options.get("fingerprint") or ""), "blocks": values}
            if action != "inspect" and len(selection["fingerprint"]) != 64:
                raise ValueError("请先读取 LoRA 文件。")
            if action == "compress": selection["rank"] = number(options.get("rank"), 1, 1024, True)
            if action in ("preview", "profile", "explore"):
                if family == "auto":
                    raise ValueError("请先读取 LoRA。")
                p = options.get("preview") or {}
                if not isinstance(p, dict): raise ValueError("预览配置无效。")
                prompt = str(p.get("prompt") or "").strip()
                if not prompt or len(prompt) > 12000: raise ValueError("请填写预览提示词。")
                preview = {"prompt": prompt, "negative": str(p.get("negative") or "")[:12000],
                           "seed": number(p.get("seed", 42), 0, 2**32 - 1, True),
                           "width": number(p.get("width", 512), 256, 1536, True), "height": number(p.get("height", 512), 256, 1536, True),
                           "steps": number(p.get("steps", 25), 1, 100, True), "cfg": number(p.get("cfg", 4.5), 0, 20),
                           "strength": number(p.get("strength", 1), -3, 3), "swap": number(p.get("swap", 0), 0, 100, True),
                           "precision": str(p.get("precision") or "auto")}
                exact_attention = p.get("exact_attention", False)
                if not isinstance(exact_attention, bool): raise ValueError("预览精度设置无效。")
                preview["exact_attention"] = True if action == "profile" else exact_attention
                if preview["precision"] not in ("auto", "bf16", "int8", "nf4"): raise ValueError("底模精度无效。")
                if preview["width"] % 64 or preview["height"] % 64: raise ValueError("预览宽高需为 64 的整数倍。")
                for key in (("dit",) if family == "sdxl" else ("dit", "vae", "te")):
                    path = Path(str(p.get(key) or ""))
                    if not path.is_file(): raise ValueError("请先选择预览底模、VAE 和文本编码器；SDXL 使用完整底模。")
                    preview[key] = str(path.resolve())
                if family == "sdxl": preview["te"] = preview["vae"] = preview["dit"]
                validate_base(self.core, family, preview["dit"])
                preview["prediction"] = "v" if family == "sdxl" and self.core._looks_like_vpred(preview["dit"]) else "epsilon"
                selection["preview"] = preview
                if action == "explore":
                    if preview["strength"] == 0: raise ValueError("变体探索的 LoRA 使用强度不能为 0。")
                    config = options.get("explore") or {}
                    if not isinstance(config, dict): raise ValueError("变体探索配置无效。")
                    frozen, last_pick = config.get("frozen") or [], config.get("last_pick") or []
                    for ids in (frozen, last_pick):
                        if not isinstance(ids, list) or len(ids) > 1024 or any(not isinstance(b, str) or len(b) > 64 for b in ids):
                            raise ValueError("冻结或保护的层级无效。")
                    state = config.get("state") or {}
                    if not isinstance(state, dict) or len(state) > 1024:
                        raise ValueError("变体层级状态无效。")
                    states = {}
                    for bid, value in state.items():
                        if not isinstance(bid, str) or len(bid) > 64 or not isinstance(value, dict) or not isinstance(value.get("enabled"), bool):
                            raise ValueError("变体层级状态无效。")
                        states[bid] = {"strength": number(value.get("strength"), -3, 3), "enabled": value["enabled"]}
                    selection["explore"] = {"intensity": number(config.get("intensity", 0.2), 0, 1),
                        "structure": number(config.get("structure", 0), 0, 1),
                        "mutations": number(config.get("mutations", 3), 1, 64, True),
                        "mutation_seed": number(config.get("mutation_seed", 42), 0, 2**32 - 1, True),
                        "frozen": sorted(set(frozen)), "last_pick": sorted(set(last_pick)), "state": states}
                if action == "profile":
                    if preview["strength"] == 0: raise ValueError("Profiler 的 LoRA 使用强度不能为 0。")
                    config = options.get("profile") or {}
                    if not isinstance(config, dict): raise ValueError("Profiler 配置无效。")
                    plain = str(config.get("class_prompt") or "").strip()
                    if len(plain) > 12000: raise ValueError("不含触发词的提示词过长。")
                    refs = config.get("references") or []
                    if not isinstance(refs, list) or len(refs) > 3: raise ValueError("最多选择 3 张参考照片。")
                    if refs and family != "krea2": raise ValueError("Anima 首版不使用人脸评分，请移除参考照片。")
                    references = []
                    for value in refs:
                        path = Path(str(value)).resolve()
                        if not path.is_file() or path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                            raise ValueError("请选择有效的参考照片。")
                        references.append(str(path))
                    selection["profile"] = {"class_prompt": plain, "references": references}
            # Reserve the task before the save dialog, so a second click cannot race it.
            title = {"inspect": "读取 LoRA", "analyze": "分析 LoRA 权重", "profile": "Profiler 快速分组分析", "explore": "LoRA 变体探索", "preview": "LoRA 原版 / 调整版对照", "export": "另存分层调整版", "compress": "压缩 LoRA"}[action]
            task_id = self._begin_task(title, "lora_workbench")
            if not task_id: return {"ok": False, "error": "已有任务正在运行，请先完成或停止。"}
            self.core.reset_stop()
            target = None
            try:
                if action in ("export", "compress"):
                    if self._window is None: raise ValueError("请在桌面程序中另存 LoRA。")
                    import webview
                    suffix = "_adjusted" if action == "export" else "_rank%d" % selection["rank"]
                    selected = self._window.create_file_dialog(webview.FileDialog.SAVE, save_filename=source.stem + suffix + ".safetensors", file_types=("LoRA (*.safetensors)",))
                    if not selected:
                        with self._task_lock: self._task.update(status="cancelled", message="已取消另存。")
                        return {"ok": True, "cancelled": True}
                    target = Path(selected[0] if isinstance(selected, (list, tuple)) else selected).resolve()
                    if target.suffix.lower() != ".safetensors": target = target.with_name(target.name + ".safetensors")
                    if target == source or (target.exists() and os.path.samefile(target, source)):
                        raise ValueError("请另选文件名，不能覆盖原始 LoRA。")
                    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".lora_workbench_", suffix=".tmp", delete=False) as stream:
                        staged = Path(stream.name)
                folder = self._workbench_root() / task_id
                folder.mkdir(parents=True, exist_ok=False)
                job = {"action": action, "options": selection}
                if staged: job["output"] = str(staged)
                _write(folder / "job.json", job)
                try:
                    previous = json.loads(self._workbench_settings().read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    previous = {}
                if not isinstance(previous, dict) or previous.get("source") != selection["source"] or previous.get("family") != family:
                    previous = {}
                _write(self._workbench_settings(), dict(previous, **selection, last_task=task_id))
                script = Path(__file__).resolve().parents[1] / "kohya_core" / "lora_workbench_worker.py"
                if not script.is_file(): raise ValueError("缺少 LoRA 工作台运行文件，请使用完整安装包。")
                threading.Thread(target=self._workbench_execute, args=(task_id, action, selection, record, folder, script, staged, target), daemon=True, name="lora-workbench").start()
                staged = None  # Ownership transferred to worker; cleanup happens after process completion.
            except Exception:
                with self._task_lock: self._task.update(status="failed", message="工作台任务未启动。")
                raise
            return {"ok": True, "task_id": task_id}
        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            return {"ok": False, "error": redact(str(exc))}
        finally:
            if staged is not None:
                try: staged.unlink(missing_ok=True)
                except OSError: pass

    def _workbench_execute(self, task_id, action, options, record, folder, script, staged, target):
        status, message = "completed", "处理完成。"
        def log(line):
            if line.startswith("WORKBENCH:"):
                try:
                    update = json.loads(line[len("WORKBENCH:"):])
                    with self._task_lock:
                        if self._task and self._task.get("id") == task_id:
                            self._task.update(message=str(update["message"]), progress=update.get("progress"))
                    return
                except (ValueError, KeyError, TypeError): pass
            self._task_log(task_id, line)
        try:
            self.core.check_stop()
            env = _env(self.core, record)
            env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
            if action in ("preview", "profile", "explore"):
                self._release_agent_model_for_task(task_id)
                helpers = _helper_configs(self.core, options["family"], log)
                if helpers:
                    root, repo = helpers
                    env["FIZGIG_LOCAL_HELPERS"] = json.dumps({repo: root})
                    if options["family"] == "anima": env["FIZGIG_ANIMA_HELPER_DIR"] = root
                    if options["family"] == "sdxl": env["FIZGIG_SDXL_CONFIG_DIR"] = root
                if options["family"] == "qwen_image21":
                    env["FIZGIG_QWEN21_PROCESSOR_DIR"] = str(Path(self.core._ensure_at_image_qwen21_assets(log)) / "processor")
            code = self.core.run_stream([record["python"], "-I", str(script), "--source", record["source"], "--job", str(folder / "job.json")], cwd=str(folder), env=env, logf=log)
            self.core.check_stop()
            report = json.loads((folder / "result.json").read_text(encoding="utf-8")) if (folder / "result.json").is_file() else {}
            if code or not report.get("ok"):
                raise ValueError(report.get("error") or "工作台处理失败，请查看运行日志。")
            if target and staged:
                if not self.core._safetensors_complete(str(staged)): raise ValueError("导出的文件不完整，结果未保存。")
                staged.replace(target)
                report["output"] = str(target)
                report["output_bytes"] = target.stat().st_size
                message = "已另存：%s" % target.name
                _write(folder / "result.json", report)
            elif action == "preview": message = "对照图已生成。"
            elif action == "explore":
                message = "四个变体已生成，可选为新基准或返回分层调整。"
                with self._task_lock:
                    saved = json.loads(self._workbench_settings().read_text(encoding="utf-8"))
                    if isinstance(saved, dict) and saved.get("source") == options["source"] and saved.get("family") == options["family"]:
                        _write(self._workbench_settings(), dict(saved, last_explore_task=task_id))
            elif action == "profile":
                message = "Profiler 分析完成，可查看分组对照与调整建议。"
                with self._task_lock:
                    saved = json.loads(self._workbench_settings().read_text(encoding="utf-8"))
                    if isinstance(saved, dict) and saved.get("source") == options["source"] and saved.get("family") == options["family"]:
                        _write(self._workbench_settings(), dict(saved, last_profile_task=task_id))
            elif action == "inspect": message = "已读取 LoRA，可调整层级倍率。"
        except self.core.StopRequested:
            status, message = "cancelled", "任务已停止。"
        except Exception as exc:
            status, message = "failed", redact(str(exc))
            self._task_log(task_id, "[LoRA 工作台] " + message)
        finally:
            if action == "explore":
                # A stopped child may not execute its own finally block.
                try: (folder / "explore_current.safetensors").unlink(missing_ok=True)
                except OSError: pass
            if staged:
                try: staged.unlink(missing_ok=True)
                except OSError: pass
            if status != "completed":
                try: _write(folder / "result.json", {"ok": False, "action": action, "error": message})
                except OSError: pass
            with self._task_lock:
                if self._task and self._task.get("id") == task_id:
                    self._task.update(status=status, message=message, progress=1.0 if status == "completed" else None)

    def get_lora_workbench_result(self, task_id):
        try:
            task_id = uuid.UUID(str(task_id)).hex
            folder = self._workbench_root() / task_id
            task = self.get_task_status(task_id)
            path = folder / "result.json"
            # Results become visible only after atomic export/worker completion.
            report = json.loads(path.read_text(encoding="utf-8")) if path.is_file() and (not task.get("ok") or task.get("status") == "completed") else None
            if task.get("ok") and task.get("status") == "running": report = None
            return {"ok": True, "report": report, "task": task if task.get("ok") else None}
        except (ValueError, OSError, TypeError):
            return {"ok": False, "error": "工作台记录无法读取，请重新执行。"}

    def get_lora_workbench_image(self, task_id, name):
        try:
            task_id = uuid.UUID(str(task_id)).hex
            if name not in ("original.png", "adjusted.png") and not re.fullmatch(r"(?:profile|explore)_[0-9]{4}\.png", str(name)): raise ValueError("图片无效。")
            folder = self._workbench_root() / task_id
            report = json.loads((folder / "result.json").read_text(encoding="utf-8"))
            if not report.get("ok") or name not in {item["name"] for item in report.get("images", []) + report.get("profile_images", [])}: raise ValueError("对照图尚未完成。")
            path = folder / name
            if path.stat().st_size > 20 * 1024 * 1024: raise ValueError("对照图过大。")
            return {"ok": True, "data_url": "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")}
        except (ValueError, OSError, TypeError, KeyError):
            return {"ok": False, "error": "对照图无法读取。"}
