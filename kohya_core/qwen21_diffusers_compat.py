"""Qwen-Image-2.1 split FFN compatibility, copied into the AI Toolkit runtime.

Only an official Diffusers directory with gate_layer/proj keys needs the repair.
Each pair is read from the checkpoint and row-concatenated in gate, up order.
No missing model parameter is randomly initialized or allocated with to_empty.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path

PATCH_MARKER = "KOHYA_QWEN21_DIFFUSERS_FFN_V1"
_MODEL_REL = "extensions_built_in/diffusion_models/qwen_image_2/src/transformer.py"
_RUNTIME_REL = "toolkit/models/v2/qwen21_diffusers_compat.py"


def _local_weight_map(path, subfolder):
    root = Path(path)
    if not root.is_dir():
        return None, None
    root = root.resolve()
    if subfolder and (root / subfolder / "config.json").is_file():
        root = (root / subfolder).resolve()
    index = root / "diffusion_pytorch_model.safetensors.index.json"
    if index.is_file():
        data = json.loads(index.read_text(encoding="utf-8-sig"))
        mapping = data.get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise RuntimeError("Qwen-Image-2.1 transformer index is empty or invalid")
        return root, mapping
    single = root / "diffusion_pytorch_model.safetensors"
    if single.is_file():
        from safetensors import safe_open
        with safe_open(str(single), framework="pt", device="cpu") as handle:
            return root, {name: single.name for name in handle.keys()}
    return None, None


def _checkpoint_file(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise RuntimeError("Invalid Qwen-Image-2.1 shard path")
    file = (root / relative).resolve()
    if not file.is_relative_to(root) or not file.is_file():
        raise RuntimeError(f"Qwen-Image-2.1 shard is missing or outside the model folder: {relative}")
    return file


def _assert_materialized(model):
    missing = [name for name, tensor in list(model.named_parameters()) + list(model.named_buffers())
               if tensor.is_meta]
    if missing:
        raise RuntimeError("Qwen-Image-2.1 load left tensors without checkpoint data: " +
                           ", ".join(missing[:12]))


def load_local_diffusers(klass, native_loader, path, subfolder=None, dtype=None, **kwargs):
    root, mapping = _local_weight_map(path, subfolder)
    if mapping is None:
        return native_loader(path, subfolder=subfolder, dtype=dtype, **kwargs)
    pairs = []
    for name in mapping:
        if name.endswith(".img_mlp.gate_layer.weight"):
            prefix = name[:-len("gate_layer.weight")]
            other = prefix + "proj.weight"
            target = prefix + "gate_up.weight"
            if other not in mapping or target in mapping:
                raise RuntimeError(f"Ambiguous or incomplete Qwen-Image-2.1 FFN weights: {prefix}")
            pairs.append((name, other, target))
    if not pairs:
        result = native_loader(path, subfolder=subfolder, dtype=dtype, **kwargs)
        _assert_materialized(result[0] if isinstance(result, tuple) else result)
        return result
    if kwargs.get("device_map") is not None:
        raise RuntimeError("Qwen-Image-2.1 split FFN loading requires CPU loading before engine offloading")
    want_info = bool(kwargs.pop("output_loading_info", False))
    print(f"[Qwen-Image-2.1] Converting {len(pairs)} split FFN pairs from the local checkpoint", flush=True)
    model, info = native_loader(path, subfolder=subfolder, dtype=dtype,
                                output_loading_info=True, **kwargs)
    targets = {target for _, _, target in pairs}
    source_names = {name for left, right, _ in pairs for name in (left, right)}
    missing = set(info.get("missing_keys", ())) - targets
    unexpected = set(info.get("unexpected_keys", ())) - source_names
    if missing or unexpected or info.get("mismatched_keys") or info.get("error_msgs"):
        raise RuntimeError("Qwen-Image-2.1 checkpoint/config mismatch beyond split FFN weights: " +
                           str({"missing": sorted(missing)[:12], "unexpected": sorted(unexpected)[:12],
                                "mismatched": info.get("mismatched_keys"), "errors": info.get("error_msgs")}))
    import torch
    from safetensors import safe_open
    parameters = dict(model.named_parameters())
    # One pair at a time: no second full model and no rewritten large checkpoint.
    for left, right, target in pairs:
        if target not in parameters:
            raise RuntimeError(f"The selected Qwen-Image-2.1 engine has no fused FFN parameter: {target}")
        left_file = _checkpoint_file(root, mapping[left])
        right_file = _checkpoint_file(root, mapping[right])
        with safe_open(str(left_file), framework="pt", device="cpu") as handle:
            gate = handle.get_tensor(left)
        with safe_open(str(right_file), framework="pt", device="cpu") as handle:
            up = handle.get_tensor(right)
        if gate.ndim != 2 or gate.shape != up.shape or gate.dtype != up.dtype:
            raise RuntimeError(f"Qwen-Image-2.1 FFN pair has incompatible shape or dtype: {target}")
        expected = parameters[target]
        if (gate.shape[0] + up.shape[0], gate.shape[1]) != tuple(expected.shape):
            raise RuntimeError(f"Qwen-Image-2.1 fused FFN shape mismatch: {target}")
        value = torch.cat((gate, up), dim=0)
        if isinstance(dtype, torch.dtype):
            value = value.to(dtype=dtype)
        else:
            value = value.to(dtype=expected.dtype)
        model.load_state_dict({target: value}, strict=False, assign=True)
        del gate, up, value
    _assert_materialized(model)
    info["missing_keys"] = [name for name in info.get("missing_keys", ()) if name not in targets]
    info["unexpected_keys"] = [name for name in info.get("unexpected_keys", ()) if name not in source_names]
    print("[Qwen-Image-2.1] All checkpoint parameters are materialized; FP8 quantization can proceed", flush=True)
    return (model, info) if want_info else model


def patch_ready(engine_dir):
    root = Path(engine_dir).resolve()
    model = root / _MODEL_REL
    if not model.is_file():
        return False
    text = model.read_text(encoding="utf-8-sig")
    if "self.gate_up =" not in text:
        return True
    runtime = root / _RUNTIME_REL
    return (PATCH_MARKER in text and runtime.is_file() and
            "def load_local_diffusers(" in runtime.read_text(encoding="utf-8-sig"))


def patch_engine(engine_dir, logf=print):
    root = Path(engine_dir).resolve()
    target = root / _MODEL_REL
    if not target.is_file() or not target.resolve().is_relative_to(root):
        raise RuntimeError("Qwen-Image-2.1 transformer source is missing")
    original = target.read_bytes()
    text = original.decode("utf-8-sig")
    if "self.gate_up =" not in text:
        return True  # Older engines still use the native split FFN layout.
    tree = ast.parse(text)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "QwenImage21Transformer2DModel"]
    if len(classes) != 1:
        raise RuntimeError("Qwen-Image-2.1 transformer class changed; refusing an unsafe patch")
    existing = [node for node in classes[0].body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "aitk_from_pretrained"]
    if existing and PATCH_MARKER not in ast.get_source_segment(text, existing[0]):
        raise RuntimeError("Qwen-Image-2.1 already defines a custom directory loader; inspect it before patching")
    if not existing:
        method = next(node for node in classes[0].body if isinstance(node, ast.FunctionDef))
        line = min([method.lineno] + [node.lineno for node in method.decorator_list])
        hook = (
            "    @classmethod\n"
            "    def aitk_from_pretrained(cls, path, subfolder=None, dtype=None, **kwargs):\n"
            "        # " + PATCH_MARKER + "\n"
            "        from toolkit.models.v2.qwen21_diffusers_compat import load_local_diffusers\n"
            "        return load_local_diffusers(\n"
            "            cls, super().aitk_from_pretrained, path,\n"
            "            subfolder=subfolder, dtype=dtype, **kwargs)\n\n")
        lines = text.splitlines(keepends=True)
        lines.insert(line - 1, hook)
        text = "".join(lines)
        ast.parse(text)
    own = Path(__file__)
    if not own.is_file():
        # PyInstaller onedir keeps runtime helper source beside the executable.
        from kohya_core import KIT_DIR
        own = Path(KIT_DIR) / "kohya_core/qwen21_diffusers_compat.py"
    payload = own.read_bytes()
    ast.parse(payload.decode("utf-8-sig"))
    runtime = root / _RUNTIME_REL
    if not runtime.parent.is_dir():
        raise RuntimeError("AI Toolkit v2 module folder is missing")
    if not runtime.exists() or runtime.read_bytes() != payload:
        temporary = runtime.with_name(runtime.name + ".tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, runtime)
    if text != original.decode("utf-8-sig"):
        backup = target.with_name(target.name + ".kohya_backup_" + hashlib.sha256(original).hexdigest()[:12])
        if not backup.exists():
            backup.write_bytes(original)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(text, encoding="utf-8", newline="")
        os.replace(temporary, target)
        logf("[AI Toolkit] 已启用 Qwen-Image-2.1 Diffusers 分片 FFN 权重兼容转换")
    return patch_ready(root)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Patch Qwen-Image-2.1 directory loading")
    parser.add_argument("engine_dir")
    options = parser.parse_args()
    if not patch_engine(options.engine_dir):
        raise SystemExit("Qwen-Image-2.1 patch verification failed")
    print("Qwen-Image-2.1 folder loader patch is installed")
