"""Project-scoped Krea 2 Fizgig component paths; inspect headers, never load weights."""
from __future__ import annotations
from functools import lru_cache
import os
from pathlib import Path

# Reuse the bounded safetensors/quantization inspector, not H3's shape contracts.
from .h3_models import _header

KEYS = ("raw", "turbo", "vae", "te", "te_fp8")
ENCODERS = ("te", "te_fp8")
CHOICES = ("auto", *ENCODERS)
TITLES = dict(raw="Krea 2 RAW 底模", turbo="Krea 2 Turbo", vae="Qwen-Image VAE",
              te="Qwen3-VL-4B BF16 文本编码器", te_fp8="Qwen3-VL-4B FP8 scaled 文本编码器")
_FLOATS = {"BF16", "F16", "F32", "F64"}
_FP8 = {"F8_E4M3", "F8_E4M3FN", "F8_E5M2"}


def settings(config=None):
    value = (config or {}).get("krea2_models", {})
    if not isinstance(value, dict):
        return {}
    result = {key: path for key, path in value.items() if key in KEYS and isinstance(path, str)}
    if value.get("text_encoder") in CHOICES:
        result["text_encoder"] = value["text_encoder"]
    return result


def encoder_choice(selection=None):
    selection = selection or {}
    if selection.get("text_encoder") in CHOICES:
        return selection["text_encoder"]
    # Preserve older projects whose only explicit selection was a local TE path.
    return next((key for key in ("te_fp8", "te") if selection.get(key)), "auto")


def change(selection, key, action="select", path=""):
    result = settings({"krea2_models": selection})
    # File management and encoder selection are separate operations.
    result["text_encoder"] = encoder_choice(result)
    if action == "default":
        result.pop(key, None)
    else:
        result[key] = path
    return result


def encoder_key(files, selection=None, allow_fp8=True):
    if not allow_fp8:
        return "te"
    choice = encoder_choice(selection)
    # An explicit selection must not silently fall back if the file goes missing.
    if choice in ENCODERS:
        return choice
    return "te_fp8" if files.get("te_fp8") or not files.get("te") else "te"


def encoder_state(state, selection=None, allow_fp8=True):
    resolved = {key: row["path"] if row["present"] else None for key, row in state.items()}
    key = encoder_key(resolved, selection, allow_fp8)
    choice = encoder_choice(selection) if allow_fp8 else "te"
    row = state[key]
    reason = ("旧版引擎仅支持 BF16 文本编码器。" if not allow_fp8 else
              "自动推荐：优先使用已就绪的 FP8 scaled；没有时使用 BF16。" if choice == "auto" else
              "已固定选择；下载或指定备用文件不会切换，缺少所选文件会阻止训练。")
    return dict(choice=choice, active_key=key, path=row["path"], label="FP8 scaled" if key == "te_fp8" else "BF16",
                ready=row["present"], supports_fp8=allow_fp8, reason=reason)


def _validate_header(header, formats, key):
    tensors = {name: value for name, value in header.items() if name != "__metadata__"}
    def shape(name, wanted):
        return tensors.get(name, {}).get("shape") == wanted
    if key in ("raw", "turbo"):
        if formats or any(item.get("dtype") not in _FLOATS for item in tensors.values()):
            raise ValueError("Krea 2 需要完整浮点 RAW / Turbo 文件，不能使用推理用 FP8、INT8 或 GGUF 预量化底模。")
        if (not shape("first.weight", [6144, 64]) or not shape("last.linear.weight", [64, 6144])
                or any(not shape(f"blocks.{i}.attn.wq.weight", [6144, 6144]) for i in range(28))):
            raise ValueError("所选文件不是标准 Krea 2 底模；请选择 RAW / Turbo，而不是 LoRA 或其他模型。")
        return "Krea 2 " + ("RAW" if key == "raw" else "Turbo") + " · 浮点文件头检查通过"
    if key == "vae":
        if (not shape("encoder.conv1.weight", [96, 3, 3, 3, 3])
                or not shape("decoder.conv1.weight", [384, 16, 3, 3, 3])
                or not shape("conv2.weight", [16, 16, 1, 1, 1])
                or formats or any(item.get("dtype") not in _FLOATS for item in tensors.values())):
            raise ValueError("请选择 Krea 2 使用的 Qwen-Image 浮点 VAE，不支持其他架构或量化 VAE。")
        return "Qwen-Image VAE · 文件头检查通过"
    prefix = "model.language_model" if "model.language_model.embed_tokens.weight" in tensors else "model"
    if (not shape(prefix + ".embed_tokens.weight", [151936, 2560])
            or not shape("model.visual.patch_embed.proj.weight", [1024, 3, 2, 16, 16])
            or any(not shape(f"{prefix}.layers.{i}.input_layernorm.weight", [2560]) for i in range(36))):
        raise ValueError("请选择完整的 Qwen3-VL-4B 文本编码器；其他尺寸、纯文本模型或 LoRA 不能替代。")
    quantized = {name: item for name, item in tensors.items() if item.get("dtype") in _FP8}
    if key == "te":
        if formats or any(item.get("dtype") not in _FLOATS for item in tensors.values()):
            raise ValueError("这里需要浮点 BF16 文本编码器；FP8 scaled 文件请在 FP8 行选择。")
        return "Qwen3-VL-4B BF16 · 文件头检查通过"
    if not quantized or any(item.get("dtype") not in _FLOATS | _FP8 | {"U8"} for item in tensors.values()):
        raise ValueError("这里需要 FP8 scaled 文本编码器；浮点 BF16 文件请在 BF16 行选择。")
    for name in quantized:
        module = name.removesuffix(".weight")
        scale = tensors.get(module + ".weight_scale", {})
        if not name.endswith(".weight") or scale.get("dtype") not in _FLOATS or scale.get("shape") not in ([], [1], [1, 1]):
            raise ValueError("FP8 文本编码器缺少兼容的 weight_scale，请使用 FP8 scaled 版本。")
    if any(value.get("format") not in ("float8_e4m3fn", "float8_e5m2") for value in formats.values()):
        raise ValueError("文本编码器的预量化格式不受支持，请使用 FP8 scaled 版本。")
    return "Qwen3-VL-4B FP8 scaled · 文件头检查通过"


@lru_cache(maxsize=32)
def _validate_cached(path, key, size, modified, changed):
    header, formats = _header(path, size, modified, changed)
    return _validate_header(header, formats, key)


def validate(path, key):
    if key not in KEYS:
        raise ValueError("未知的 Krea 2 组件类型。")
    path = os.path.abspath(os.path.expanduser(str(path or "").strip().strip('"')))
    if not Path(path).is_file():
        raise ValueError("文件不存在，请重新选择：" + path)
    if Path(path).suffix.lower() != ".safetensors":
        raise ValueError("请选择单个 .safetensors 模型文件。")
    stat = Path(path).stat()
    return _validate_cached(path, key, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def rows(core, selection=None, allow_fp8=True):
    selection = selection or {}
    links = core.KREA2_FZ_MODEL_LINKS if allow_fp8 else core.KREA2_MODEL_LINKS
    defaults = core.krea2_model_files()
    folder = Path(core.krea2_models_dir())
    result = {}
    for key, entry in links.items():
        manual = key in selection
        path = selection[key] if manual else str(defaults.get(key) or folder / entry[0])
        disabled = key == "turbo" and manual and not path
        error, detail, present = "", "不使用" if disabled else "未准备", False
        if not disabled:
            try:
                detail = validate(path, key)
                present = True
            except (ValueError, OSError) as exc:
                if manual or Path(path).exists():
                    error = str(exc)
        result[key] = dict(path=path, manual=manual, disabled=disabled, present=present, error=error, detail=detail)
    selected = encoder_key({key: row["path"] if row["present"] else None for key, row in result.items()}, selection, allow_fp8)
    for key, row in result.items():
        row["selected"] = key in ("raw", "vae", selected)
        row["in_use"] = row["selected"] and row["present"]
    return result


def files(core, selection=None, allow_fp8=True):
    return {key: row["path"] if row["present"] else None for key, row in rows(core, selection, allow_fp8).items()}


def missing(core, selection=None, allow_fp8=True):
    state = rows(core, selection, allow_fp8)
    resolved = {key: row["path"] if row["present"] else None for key, row in state.items()}
    required = ("raw", "vae", encoder_key(resolved, selection, allow_fp8))
    return [state[key]["error"] or "缺少 " + TITLES[key] for key in required if not state[key]["present"]]


def validate_selected(core, selection=None, allow_fp8=True):
    state = rows(core, selection, allow_fp8)
    encoder = encoder_key({key: row["path"] if row["present"] else None for key, row in state.items()}, selection, allow_fp8)
    for key, row in state.items():
        if key in ENCODERS and key != encoder:
            continue
        if row["manual"] and row["error"]:
            raise ValueError(TITLES[key] + "：" + row["error"])
