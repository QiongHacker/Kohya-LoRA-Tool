"""MOSS-Image component paths and CPU-only structural validation for Fizgig v8.5.0."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

PROJECT_KEY = "moss_models"
KEYS = ("dit", "te", "vae", "preview_checkpoint")
REQUIRED = ("dit", "te", "vae")
REPO = "Comfy-Org/MOSS-Image"
REVISION = "f28da8f93449c16cfa7efa07dcd1303a1c0c44d1"


def _entry(filename, label, repo, path):
    return filename, label, (
        "https://modelscope.cn/models/" + repo + "/resolve/master/" + path,
        "https://hf-mirror.com/" + repo + "/resolve/main/" + path,
        "https://huggingface.co/" + repo + "/resolve/main/" + path,
    )


LINKS = {
    "dit": _entry("moss_image_base_bf16.safetensors", "MOSS-Image Base 训练底模（10.2 GB）", REPO,
                  "diffusion_models/moss_image_base_bf16.safetensors"),
    "te": _entry("qwen3.5_9b_bf16.safetensors", "Qwen3.5-9B BF16 文本编码器（19.3 GB）", "Comfy-Org/Qwen3.5",
                 "text_encoders/qwen3.5_9b_bf16.safetensors"),
    "vae": _entry("flux2-vae_small.safetensors", "FLUX.2 小型 VAE（0.25 GB）", REPO,
                  "vae/flux2-vae_small.safetensors"),
    "preview_checkpoint": _entry("moss_image_sft_int8_convrot.safetensors", "MOSS-Image SFT 预览底模（5.4 GB）", REPO,
                                 "diffusion_models/moss_image_sft_int8_convrot.safetensors"),
}


def settings(config=None):
    value = (config or {}).get(PROJECT_KEY) or {}
    return {k: v.strip() for k, v in value.items() if k in KEYS and isinstance(v, str) and v.strip()} if isinstance(value, dict) else {}


def models_dir(core):
    return str(Path(core.data_sub("models", "moss_image")))


def component_files(path):
    p = Path(path)
    match = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.safetensors", p.name)
    if not match:
        return [p]
    stem, _part, total = match.groups()
    if not 1 <= int(total) <= 100:
        raise ValueError("模型分片数量无效。")
    return [p.with_name(f"{stem}-{i:05d}-of-{int(total):05d}.safetensors") for i in range(1, int(total) + 1)]


@lru_cache(maxsize=32)
def _validated(identities, kind):
    # The shared H3 reader checks every byte span and quantization descriptor without loading tensors.
    from .h3_models import _header
    tensors, formats = {}, {}
    for path, size, modified, changed in identities:
        header, quant = _header(path, size, modified, changed)
        current = {k: v for k, v in header.items() if k != "__metadata__"}
        if tensors.keys() & current.keys():
            raise ValueError("模型分片含重复权重。")
        tensors.update(current)
        formats.update(quant)
    def normalize(key):
        return key.removeprefix("model.diffusion_model.").removeprefix("diffusion_model.").removeprefix("model.")
    tensors = {normalize(k): v for k, v in tensors.items()}
    def shape(key):
        return tensors.get(key, {}).get("shape")
    if kind in ("dit", "preview_checkpoint"):
        if shape("video_embedder.weight") != [3072, 128] or shape("final_linear_video.weight") != [128, 3072] or shape("text_embedder.proj.weight") != [3072, 4096]:
            raise ValueError("请选择 MOSS-Image 5B DiT，文本编码器或其他模型不能作为底模。")
        for i in range(42):
            if shape(f"layers.{i}.proj.qkv.weight") != [9216, 3072] and not all(shape(f"layers.{i}.proj.{part}.weight") == [3072, 3072] for part in ("q", "k", "v")):
                raise ValueError("MOSS-Image 底模需要完整的 42 层权重。")
        if kind == "dit":
            if any("sft" in Path(item[0]).stem.lower() for item in identities):
                raise ValueError("SFT 用于预览和出图；训练请选择 MOSS-Image Base。")
            if formats or any(v.get("dtype") not in ("BF16", "F16", "F32") for k, v in tensors.items() if k.endswith("weight") and len(v.get("shape", [])) >= 2):
                raise ValueError("训练底模请选择 BF16 / FP16 Base，训练精度由引擎设置。")
        elif any(v.get("format") not in ("int8_tensorwise",) for v in formats.values()):
            raise ValueError("SFT 预览仅支持浮点或 int8 ConvRot 权重。")
    elif kind == "te":
        if shape("language_model.embed_tokens.weight") != [248320, 4096] or not all(shape(f"language_model.layers.{i}.input_layernorm.weight") == [4096] for i in range(32)):
            raise ValueError("MOSS-Image 需要完整 Qwen3.5-9B 文本编码器；选择任一分片时其余分片须在同一目录。")
        if formats or any(v.get("dtype") not in ("BF16", "F16", "F32") for k, v in tensors.items() if k.startswith("language_model.") and k.endswith("weight")):
            raise ValueError("文本编码器请选择 BF16 / FP16 权重；小显存会自动逐层加载。")
    elif kind == "vae":
        if shape("encoder.conv_in.weight") != [128, 3, 3, 3] or shape("decoder.conv_in.weight") not in ([512, 32, 3, 3], [384, 32, 3, 3]) or shape("bn.running_mean") != [128]:
            raise ValueError("请选择 FLUX.2 VAE（优先小型版本），FLUX.1 / SDXL VAE 不兼容。")
    else:
        raise ValueError("未知 MOSS-Image 组件。")
    return "结构检查通过"


def validate(path, kind):
    if kind not in KEYS:
        raise ValueError("未知 MOSS-Image 组件。")
    identities = []
    for p in component_files(path):
        if not p.is_file() or p.suffix.lower() != ".safetensors":
            raise ValueError("文件不存在或模型分片不完整：" + str(p))
        st = p.stat()
        identities.append((str(p.resolve()), st.st_size, st.st_mtime_ns, st.st_ctime_ns))
    return _validated(tuple(identities), kind)


def rows(core, selection=None):
    selection = selection or {}
    root = Path(models_dir(core))
    result = {}
    for key, entry in LINKS.items():
        path = selection.get(key) or str(root / entry[0])
        present, error, detail = False, "", "未准备"
        try:
            detail = validate(path, key)
            present = True
        except (OSError, ValueError) as exc:
            if key in selection or Path(path).exists():
                error = str(exc)
        result[key] = dict(path=path, present=present, error=error, detail=detail, manual=key in selection,
                           selected=True, in_use=present, disabled=False)
    return result


def files(core, selection=None):
    return {key: row["path"] if row["present"] else None for key, row in rows(core, selection).items()}


def missing(core, selection=None, preview=False, base_preview=False):
    required = (*REQUIRED, "preview_checkpoint") if preview and not base_preview else REQUIRED
    return [row["error"] or "缺少 " + LINKS[key][1] for key, row in rows(core, selection).items()
            if key in required and not row["present"]]
