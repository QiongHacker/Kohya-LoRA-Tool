"""Windows ROCm device selection and installed architecture-package inspection.

The probe uses the target engine interpreter. It allocates no GPU tensors and
never derives HIP indices from DXGI/WMI ordering. Importing this module is inert.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import threading

VISIBILITY_KEYS = ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")
SELECTION_ERROR = "KOHYA_ROCM_SELECTION_ERROR"
_CACHE = {}
_LOCK = threading.RLock()
_INVENTORY = r"""
import json, torch
out = {"hip": torch.version.hip, "devices": []}
if torch.version.hip:
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        out["devices"].append({"index": index, "name": props.name,
            "arch": getattr(props, "gcnArchName", "").split(":")[0],
            "bytes": props.total_memory})
print("KLT_ROCM_INVENTORY=" + json.dumps(out, ensure_ascii=False))
"""
_DEVICE_REPORT = r"""
import importlib.metadata as md, json, sys
from packaging.requirements import Requirement
arch = sys.argv[1]
extra = "device-" + arch
required, issues, versions = {}, {}, {}
for root in ("torch", "torchvision", "rocm"):
    dist = md.distribution(root)
    if extra not in (dist.metadata.get_all("Provides-Extra") or []):
        raise RuntimeError(root + " 未声明 " + extra + " 显卡组件")
    selected = []
    for line in (dist.requires or []):
        req = Requirement(line)
        if "-device-" in req.name.replace("_", "-").lower() and req.marker and req.marker.evaluate({"extra": extra}):
            required[req.name] = str(req)
            selected.append(req.name)
            try:
                package = md.distribution(req.name)
                versions[req.name] = package.version
                if package.version not in req.specifier:
                    issues[req.name] = "版本不匹配：" + package.version
                elif package.files is None:
                    issues[req.name] = "安装文件清单缺失"
                else:
                    absent = [str(f) for f in package.files if f.suffix != ".pyc" and not package.locate_file(f).is_file()]
                    if absent:
                        issues[req.name] = "安装文件缺失：" + ", ".join(absent[:3])
            except md.PackageNotFoundError:
                issues[req.name] = "未安装"
    if not selected:
        raise RuntimeError(root + " 未找到 " + arch + " 的组件依赖")
print("KLT_ROCM_PACKAGES=" + json.dumps({"required": required, "issues": issues, "versions": versions}, ensure_ascii=False))
"""


def _run_json(python, code, marker, env, arguments=(), timeout=60):
    process = subprocess.run([str(python), "-I", "-X", "utf8", "-u", "-c", code, *arguments],
        capture_output=True, encoding="utf-8", errors="replace", env=env, timeout=timeout,
        **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}))
    if process.returncode:
        raise RuntimeError((process.stderr or process.stdout or "ROCm 检查进程失败").strip()[-1200:])
    for line in reversed(process.stdout.splitlines()):
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    raise RuntimeError("ROCm 检查进程未返回设备记录")


def _name(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower().replace("(tm)", "").replace("(r)", ""))


def _integrated(name):
    name = str(name).lower()
    return ("radeon" in name and "rx" not in name and "pro" not in name
            and ("graphics" in name or "vega" in name or re.search(r"\b[6789]\d\dm\b", name)))


def choose_device(devices, preferred_name=""):
    if not devices:
        raise RuntimeError("ROCm 未枚举到可用 AMD 显卡")
    if preferred_name:
        matches = [device for device in devices if _name(device["name"]) == _name(preferred_name)]
        if not matches:
            raise RuntimeError("ROCm 未枚举到目标显卡 %s；当前可见：%s" %
                (preferred_name, "、".join(str(device["name"]) for device in devices)))
        candidates = matches
    else:
        candidates = [device for device in devices if not _integrated(device["name"])] or devices
    return max(candidates, key=lambda device: (int(device.get("bytes") or 0), -int(device["index"])))


def _fingerprint(python, env, preferred_name):
    root = Path(python).parent.parent / "Lib/site-packages"
    watched = [root / "torch/__init__.py"]
    watched.extend(root.glob("*device*.dist-info"))
    stamp = tuple((str(path), path.stat().st_mtime_ns) for path in watched if path.exists())
    return (str(Path(python).resolve()), stamp, preferred_name,
            tuple(env.get(key, "") for key in ("PATH", "HIP_PATH", "ROCM_PATH", "HSA_OVERRIDE_GFX_VERSION", *VISIBILITY_KEYS)))


def bind_runtime_env(python, env, preferred_name="", strict=False, logf=None):
    result = dict(env)
    for key in (SELECTION_ERROR, "KOHYA_ROCM_DEVICE_NAME", "KOHYA_ROCM_DEVICE_ARCH"):
        result.pop(key, None)
    # Preserve an explicit user mask. HIP indices are interpreted by HIP itself.
    explicit = any(str(result.get(key, "")).strip() for key in VISIBILITY_KEYS)
    if not (Path(python).parent.parent / "Lib/site-packages/torch/__init__.py").is_file():
        return result  # Installation has not yet provided torch.
    try:
        key = _fingerprint(python, result, preferred_name)
        with _LOCK:
            inventory = _CACHE.get(key)
        if inventory is None:
            inventory = _run_json(python, _INVENTORY, "KLT_ROCM_INVENTORY=", result)
            if inventory.get("hip") and inventory.get("devices"):
                with _LOCK:
                    if len(_CACHE) >= 32:
                        _CACHE.clear()
                    _CACHE[key] = inventory
        if not inventory.get("hip"):
            return result  # The installer may still need to replace a CUDA build.
        devices = inventory.get("devices") or []
        if explicit:
            if not devices:
                raise RuntimeError("当前显卡可见设置没有暴露可用 AMD GPU")
            selected = devices[0]
        else:
            selected = choose_device(devices, preferred_name)
            result["HIP_VISIBLE_DEVICES"] = str(selected["index"])
        result["KOHYA_ROCM_DEVICE_NAME"] = str(selected["name"])
        result["KOHYA_ROCM_DEVICE_ARCH"] = str(selected.get("arch") or "")
        if logf:
            logf("[AMD] 训练显卡：%s（%s），HIP_VISIBLE_DEVICES=%s" %
                 (selected["name"], selected.get("arch") or "未知架构", result.get("HIP_VISIBLE_DEVICES", "用户指定")))
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        if strict:
            raise RuntimeError("ROCm 选卡失败：" + str(exc)) from exc
        result[SELECTION_ERROR] = str(exc)
    return result


def device_package_report(python, arch, env):
    if not re.fullmatch(r"gfx[0-9a-z]+", str(arch)):
        raise ValueError("无效 AMD 架构")
    return _run_json(python, _DEVICE_REPORT, "KLT_ROCM_PACKAGES=", env, (arch,))


def require_device_packages(python, arch, env):
    report = device_package_report(python, arch, env)
    if report["issues"]:
        raise RuntimeError("ROCm %s 显卡组件不完整：%s" % (arch,
            "；".join(name + "（" + reason + "）" for name, reason in report["issues"].items())))
    return report
