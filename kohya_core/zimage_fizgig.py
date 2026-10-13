"""Z-Image Turbo Fizgig component contract; original local files stay in place."""
from __future__ import annotations
import hashlib
from functools import lru_cache
from pathlib import Path
from . import zimage_models

PROJECT_KEY = "zimage_fz_models"
KEYS = ("dit", "te", "vae", "training_adapter")
REPO = "Comfy-Org/z_image_turbo"
ADAPTER_REPO = "ShootTheSound/Fizgig-Z-Image-Turbo-Training-Adapter"
ADAPTER_REVISION = "22a6a059ee04deeebc8d0d4c6d2c8d6e74c943bf"
ADAPTER_SHA256 = "2da9920c1b27e07cbf4f9959a7d35af335c12ed73fa615b8b88beb2b64e835bf"
ADAPTER_BYTES = 70117384

def _entry(filename, label, repo, relative):
    return (filename, label, (
        "https://modelscope.cn/models/" + repo + "/resolve/master/" + relative,
        "https://hf-mirror.com/" + repo + "/resolve/main/" + relative,
        "https://huggingface.co/" + repo + "/resolve/main/" + relative,
    ))

LINKS = {
    "dit": _entry("z_image_turbo_bf16.safetensors", "Z-Image Turbo BF16 底模（约 12.3 GB）", REPO, "split_files/diffusion_models/z_image_turbo_bf16.safetensors"),
    "te": _entry("qwen_3_4b.safetensors", "Qwen3-4B 文本编码器（约 8 GB）", REPO, "split_files/text_encoders/qwen_3_4b.safetensors"),
    "vae": _entry("ae.safetensors", "FLUX.1 VAE（约 0.34 GB）", REPO, "split_files/vae/ae.safetensors"),
    "training_adapter": ("fizgig_z_image_turbo_training_adapter.safetensors", "Fizgig Turbo 训练适配器（约 70 MB）", (
        "https://modelscope.cn/models/FGtiancai/Kohya-LoRA-Tool/resolve/master/engine_sources/models/zimage/fizgig_z_image_turbo_training_adapter.safetensors",
        "https://hf-mirror.com/" + ADAPTER_REPO + "/resolve/" + ADAPTER_REVISION + "/fizgig_z_image_turbo_training_adapter.safetensors",
        "https://huggingface.co/" + ADAPTER_REPO + "/resolve/" + ADAPTER_REVISION + "/fizgig_z_image_turbo_training_adapter.safetensors",
    )),
}

def settings(config=None):
    values = (config or {}).get(PROJECT_KEY) or {}
    return {key: str(values[key]).strip() for key in KEYS if isinstance(values.get(key), str) and values[key].strip()}

def models_dir(core):
    return str(Path(core.data_sub("models", "zimage_fizgig")))

@lru_cache(maxsize=32)
def _validated(path, key, size, modified, changed):
    if key in ("dit", "te", "vae"):
        zimage_models.validate(path, {"dit":"transformer", "te":"text_encoder", "vae":"vae"}[key])
    else:
        if size != ADAPTER_BYTES:
            raise ValueError("训练适配器文件不完整或不是当前官方版本，请重新下载。")
        h = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""): h.update(chunk)
        if h.hexdigest() != ADAPTER_SHA256:
            raise ValueError("训练适配器校验失败；请使用当前官方 Fizgig Turbo 适配器。")
        tensors = zimage_models.read_header(path)
        keys = {k.removeprefix("diffusion_model.") for k in tensors}
        modules = ("attention.to_q", "attention.to_k", "attention.to_v", "attention.to_out.0", "feed_forward.w1", "feed_forward.w2", "feed_forward.w3")
        for layer in range(30):
            for module in modules:
                prefix = f"layers.{layer}.{module}."
                if not all(prefix + part + ".weight" in keys for part in ("lora_A", "lora_B")):
                    raise ValueError("请选择官方 Fizgig Z-Image Turbo 训练适配器；其他模型的 LoRA 不能替代。")
    return "文件结构检查通过"

def validate(path, key):
    if key not in KEYS: raise ValueError("未知 Z-Image Turbo 组件。")
    p = Path(path)
    if not p.is_file(): raise ValueError("文件不存在，请重新选择：" + str(path))
    st = p.stat()
    return _validated(str(p.resolve()), key, st.st_size, st.st_mtime_ns, st.st_ctime_ns)

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
            if key in selection or Path(path).exists(): error = str(exc)
        result[key] = dict(path=path, present=present, error=error, detail=detail, manual=key in selection,
                           selected=True, in_use=present, disabled=False)
    return result

def files(core, selection=None):
    return {key: row["path"] if row["present"] else None for key, row in rows(core, selection).items()}

def missing(core, selection=None):
    return [row["error"] or "缺少 " + LINKS[key][1] for key, row in rows(core, selection).items() if not row["present"]]
