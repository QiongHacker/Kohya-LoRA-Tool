"""Isolated Fizgig adapter for ordinary LoRA inspection, block baking and previews.

Invoked with -I in the installed engine environment. Never imports the GUI/core.
"""
from __future__ import annotations
import argparse
import copy
from contextlib import contextmanager
from types import SimpleNamespace
import hashlib
import json
import math
from pathlib import Path
import sys


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def phase(message, progress=None):
    print("WORKBENCH:" + json.dumps({"message": message, "progress": progress}, ensure_ascii=False), flush=True)


def inspect(path, family):
    from safetensors import safe_open
    from fizgig.families.registry import get, family_of_lora
    from fizgig.families.lorafile import lora_pairs
    path = Path(path)
    if path.suffix.lower() != ".safetensors" or not path.is_file():
        raise ValueError("请选择完整的 safetensors LoRA 文件。")
    digest = fingerprint(path)
    desc = get(family) if family != "auto" else family_of_lora(str(path))
    if desc is None:
        raise ValueError("未能识别模型架构，请手动选择；当前支持 Krea 2 / Anima。")
    if desc.key not in ("krea2", "anima"):
        raise ValueError("当前 LoRA 工作台仅支持 Krea 2 和 Anima。")
    driver = desc.load_driver()
    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        if any(any(token in key for token in (".lokr_", ".hada_", ".glora_", ".dora_scale", ".lora_magnitude_vector", ".lora_mid.", ".lora_bias")) for key in keys):
            raise ValueError("第一版仅支持普通 LoRA；LoKR、LoHa、DoRA 等变体暂不支持分层调整。")
        pairs = lora_pairs(desc, keys)
        if not pairs:
            raise ValueError("文件中没有完整的普通 LoRA 权重对。")
        if any(isinstance(up, tuple) for _, _, up, _ in pairs):
            raise ValueError("当前首版暂不调整融合权重，请使用普通线性 LoRA。")
        paired = {key for _, dk, uk, _ in pairs for key in (dk, uk[0] if isinstance(uk, tuple) else uk)}
        orphan = [key for key in keys if any(token in key for token in (".lora_A.", ".lora_B.", ".lora_down.", ".lora_up.", ".lora.down.", ".lora.up.")) and key not in paired]
        if orphan:
            raise ValueError("LoRA 存在缺失的上下投影权重，不能调整或导出。")
        counts, ranks, unknown = {}, set(), []
        for module, down, up, alpha in pairs:
            shape = f.get_slice(down).get_shape()
            us = f.get_slice(up[0] if isinstance(up, tuple) else up).get_shape()
            if len(shape) != 2 or len(us) != 2 or shape[0] <= 0 or shape[0] != us[1]:
                raise ValueError("首版支持二维线性 LoRA；当前权重维度不受支持或不一致。")
            if alpha and (not math.isfinite(float(f.get_tensor(alpha).item()))):
                raise ValueError("LoRA alpha 不是有限数值。")
            ranks.add(shape[0])
            block = driver.block_of(module) if module else None
            if block is None:
                unknown.append(down)
            else:
                counts[block] = counts.get(block, 0) + 1
        groups = [{"label": g.label, "blocks": [{"id": b.id, "label": b.label, "modules": counts[b.id]} for b in g.blocks if b.id in counts]} for g in driver.block_map()]
        groups = [g for g in groups if g["blocks"]]
        mapped = {b["id"] for g in groups for b in g["blocks"]}
        if set(counts) - mapped:
            raise ValueError("LoRA 的层数超出当前引擎结构，暂不能安全调整。")
        if not groups:
            raise ValueError("该文件与所选模型架构不匹配，或层级结构尚未支持。")
        extra = keys - paired - {ak for *_, ak in pairs if ak}
        warnings = []
        if unknown or extra:
            warnings.append("%d 个未映射权重对 / %d 个附加张量将原样保留；此文件暂不提供预览与压缩。" % (len(unknown), len(extra)))
        linear = all(not isinstance(up, tuple) and len(f.get_slice(dk).get_shape()) == 2 and len(f.get_slice(up).get_shape()) == 2 for _, dk, up, _ in pairs)
    from fizgig.families.block_profile import windows
    profile_windows = [{"id": wid, "label": label, "blocks": ids} for wid, label, ids in
                       windows(SimpleNamespace(primary_block_ids=set(counts), block_groups=driver.block_map))]
    return desc, pairs, {"profile_windows": profile_windows, "source": str(path.resolve()), "name": path.name, "fingerprint": digest, "family": desc.key,
        "family_label": desc.display_name, "bytes": path.stat().st_size, "ranks": sorted(ranks), "groups": groups,
        "pairs": len(pairs), "mapped_pairs": len(pairs) - len(unknown), "warnings": warnings,
        "can_preview": not unknown and not extra and desc.key != "minimax",
        "can_compress": not unknown and not extra and linear}


def factors(info, options):
    allowed = [b["id"] for g in info["groups"] for b in g["blocks"]]
    values = options.get("blocks") or {}
    if not isinstance(values, dict) or set(values) - set(allowed):
        raise ValueError("分层配置与当前 LoRA 不匹配，请重新读取文件。")
    out = {}
    for block in allowed:
        value = float(values.get(block, 1.0))
        if not math.isfinite(value) or not -3 <= value <= 3:
            raise ValueError("分层倍率应为 -3 到 3 的有限数值。")
        out[block] = value
    return out


def bake(desc, pairs, source, output, info, values):
    """Map native family blocks to the official Repair primitive, preserving file keys.

    v7.0.1's top-level save_repaired_lora uses legacy key regexes (missing
    Krea I/O and Anima's native block namespace). Reuse its actual contribution
    baker, with the generic family driver's mapping, rather than copying math.
    This adapter is deliberately tied to the installed, pinned v7.0.1 source.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from fizgig.repair_studio.bake import _bake_single_contribution
    sd = load_file(source, device="cpu")
    for tensor in sd.values():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError("LoRA 含有 NaN / Inf，不能导出。")
    driver = desc.load_driver()
    changed = set()
    for module, dk, uk, ak in pairs:
        block = driver.block_of(module) if module else None
        if block not in values or uk in changed:
            continue
        changed.add(uk)
        original = {"lora_down.weight": sd[dk], "lora_up.weight": sd[uk]}
        if ak: original["alpha"] = sd[ak]
        contribution = _bake_single_contribution(original, values[block])
        if contribution is None:
            raise ValueError("Fizgig Repair 不支持此权重对。")
        up, down, rank = contribution
        if not torch.isfinite(up).all():
            raise ValueError("调整后权重溢出，请降低该层倍率。")
        sd[uk], sd[dk] = up.contiguous(), down.contiguous()
        if ak:
            sd[ak] = torch.tensor(float(rank), dtype=sd[ak].dtype)
        else:
            suffix = next(tag for tag in (".lora_A.weight", ".lora_down.weight", ".lora.down.weight") if dk.endswith(tag))
            sd[dk[:-len(suffix)] + ".alpha"] = torch.tensor(float(rank))
    with safe_open(source, framework="pt") as f:
        metadata = dict(f.metadata() or {})
    for key in ("sshs_model_hash", "sshs_legacy_hash", "modelspec.hash_sha256"):
        metadata.pop(key, None)
    metadata["kohya_workbench"] = json.dumps({"schema": 1, "kernel": "Fizgig v7.0.1 Repair", "source_sha256": info["fingerprint"], "family": desc.key, "blocks": values}, sort_keys=True)
    metadata["ss_network_alpha"] = str(max(info["ranks"]))
    save_file(sd, output, metadata=metadata)


def compress(desc, pairs, source, output, rank):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from fizgig.families.extract import reduce_pair
    sd = load_file(source, device="cpu")
    with safe_open(source, framework="pt") as f:
        metadata = dict(f.metadata() or {})
    energies, ranks = [], []
    for i, (_, dk, uk, ak) in enumerate(pairs):
        A, B = sd[dk], sd[uk]
        if not torch.isfinite(A).all() or not torch.isfinite(B).all():
            raise ValueError("LoRA 含有 NaN / Inf，不能压缩。")
        phase("正在压缩权重 %d / %d" % (i + 1, len(pairs)), i / len(pairs))
        scale = float(sd[ak].item()) / A.shape[0] if ak else 1.0
        down, up, energy = reduce_pair(A, B, scale, rank)
        sd[dk], sd[uk] = down.to(A.dtype).contiguous(), up.to(B.dtype).contiguous()
        # Implicit alpha equals rank; explicit alpha must follow the new rank.
        if ak:
            sd[ak] = torch.tensor(float(down.shape[0]), dtype=sd[ak].dtype)
        energies.append(energy)
        ranks.append(down.shape[0])
    for key in ("sshs_model_hash", "sshs_legacy_hash", "modelspec.hash_sha256"):
        metadata.pop(key, None)
    metadata.update(ss_network_dim=str(max(ranks)), ss_network_alpha=str(max(ranks)), kohya_workbench_compression="SVD rank %d" % rank)
    save_file(sd, output, metadata=metadata)
    return {"rank": rank, "mean_energy": sum(energies) / len(energies)}


@contextmanager
def preview_engine(desc, pairs, source, info, params):
    """One pipeline and the same compatibility checks for A/B and Profiler."""
    import torch
    if not info["can_preview"]:
        raise ValueError("当前文件含未映射权重，暂不提供图像对照或 Profiler；可进行分层导出。")
    if not torch.cuda.is_available():
        raise ValueError("未检测到可用的 CUDA / ROCm 显卡；分层导出仍可使用。")
    if params["precision"] != "auto" and params["precision"] not in desc.precisions:
        raise ValueError("所选模型不支持该精度，请选择自动或受支持的精度。")
    if params["swap"] > desc.n_blocks or (desc.key == "anima" and params["swap"]):
        raise ValueError("当前模型的块交换数无效；Anima 请设为 0。")
    engine = desc.make_workbench_engine()
    if params.get("exact_attention"):
        engine.int8_attention = False
    try:
        phase("正在加载预览底模…")
        engine.preview_settings = {"steps": params["steps"], "cfg": params["cfg"], "negative": params["negative"], "turbo": 0}
        engine.ensure_pipeline(params["dit"], params["vae"], params["te"], device="cuda", precision=params["precision"], blocks_to_swap=params["swap"])
        engine.load_primary(source)
        expected = {module for module, *_ in pairs}
        loaded = {module for module, wrapper in engine.net.wrapped.items() if "primary" in wrapper.adapters}
        if expected != loaded:
            raise ValueError("部分 LoRA 权重与所选底模不匹配，无法提供完整对照。")
        block_ids = {b["id"] for g in info["groups"] for b in g["blocks"]}
        if block_ids != set(engine.primary_block_ids):
            raise ValueError("部分 LoRA 层未加载到所选底模，无法提供可靠对照。")
        if block_ids - set(engine.default_state().blocks):
            raise ValueError("所选底模与 LoRA 层级不一致。")
        yield engine
    finally:
        engine.reset()


def preview(desc, pairs, source, info, values, options, folder):
    params = options["preview"]
    modified = folder / "preview_modified.safetensors"
    try:
        bake(desc, pairs, source, str(modified), info, values)
        with preview_engine(desc, pairs, source, info, params) as engine:
            state = engine.default_state(params["width"], params["height"])
            state.prompt, state.seed, state.primary_scale = params["prompt"], params["seed"], params["strength"]
            images = []
            for index, (name, label) in enumerate((("original.png", "原版 LoRA"), ("adjusted.png", "调整版 LoRA"))):
                current = copy.deepcopy(state)
                if index and not engine.swap_primary_weights(str(modified)):
                    raise ValueError("调整后的 LoRA 无法完整加载，请检查文件与底模兼容性。")
                phase("正在生成" + label, index / 2)
                engine.on_step = lambda done, total: phase("正在生成" + label + " · %d / %d" % (done, total), (index + done / max(1, total)) / 2)
                image = engine.generate_preview(current, steps=params["steps"])
                image.save(folder / name)
                images.append({"name": name, "label": label})
            actual_steps, actual_cfg, _, _ = engine.sampling()
            return {"images": images, "recipe": dict(params, steps=actual_steps, cfg=actual_cfg), "blocks": values}
    finally:
        modified.unlink(missing_ok=True)



def explore(desc, pairs, source, info, options, folder):
    """Official Explorer mutations; render baked weights through the export path.

    Four candidates use the upstream GUI schedule: two structural, one random,
    one random with the most recently chosen blocks protected when possible.
    Frozen blocks are excluded from the official mutation call. No donor LoRA.
    """
    import random
    from fizgig.repair_studio.state import SliderState, BlockState
    params, config = options["preview"], options["explore"]
    if params["strength"] == 0:
        raise ValueError("变体探索的 LoRA 使用强度不能为 0。")
    values = factors(info, options)
    frozen, protected = set(config["frozen"]), set(config["last_pick"])
    if (frozen | protected) - set(values):
        raise ValueError("冻结或保护的层级与当前 LoRA 不匹配，请重新读取。")
    active = set(values) - frozen
    if not active:
        raise ValueError("所有层均已冻结，请先解冻至少一层。")
    # Preserve upstream enable flags between rounds: a disabled block still has
    # an underlying strength that the official mutation can restore later.
    saved_state = config.get("state", {})
    if set(saved_state) - set(values):
        raise ValueError("变体状态与当前 LoRA 层级不匹配，请重新读取。")
    states = {}
    for bid, value in values.items():
        saved = saved_state.get(bid)
        if saved and (saved["strength"] if saved["enabled"] else 0.0) == value:
            states[bid] = BlockState(primary_strength=saved["strength"], primary_enabled=saved["enabled"], donor_enabled=False)
        else:
            states[bid] = BlockState(primary_strength=value, donor_enabled=False)
    state = SliderState(blocks=states)
    anchor = desc.load_driver().block_map()[0].blocks[0].id
    variants = []
    rng = random.getstate()
    try:
        random.seed(config["mutation_seed"])
        for index in range(4):
            candidates = active
            if index == 3 and protected:
                available = active - protected
                if len(available) >= 2:
                    candidates = available
            mutant = state.mutate(candidates, num_mutations=config["mutations"],
                                  intensity=config["intensity"],
                                  structure=config["structure"] if index < 2 else 0.0,
                                  anchor=anchor)
            weights = {bid: bs.primary_strength if bs.primary_enabled else 0.0
                       for bid, bs in mutant.blocks.items()}
            # Validate the official result before allocating the GPU pipeline.
            weights = factors(info, {"blocks": weights})
            variants.append({"name": "explore_%04d.png" % (index + 1),
                             "label": "变体 %d" % (index + 1), "blocks": weights,
                             "state": {bid: {"strength": bs.primary_strength, "enabled": bs.primary_enabled}
                                       for bid, bs in mutant.blocks.items()},
                             "changed": state.diff_blocks(mutant)})
    finally:
        random.setstate(rng)
    renders = [{"name": "explore_0000.png", "label": "本轮基准", "blocks": values}] + variants
    temporary = folder / "explore_current.safetensors"
    try:
        with preview_engine(desc, pairs, source, info, params) as engine:
            render_state = engine.default_state(params["width"], params["height"])
            render_state.prompt, render_state.seed = params["prompt"], params["seed"]
            render_state.primary_scale = params["strength"]
            for index, item in enumerate(renders):
                bake(desc, pairs, source, str(temporary), info, item["blocks"])
                if not engine.swap_primary_weights(str(temporary)):
                    raise ValueError("变体 LoRA 无法完整加载，未生成有效对照。")
                phase("变体探索 %d / 5 · %s" % (index + 1, item["label"]), index / 5)
                engine.on_step = lambda done, total: phase(
                    "变体探索 %d / 5 · %d / %d" % (index + 1, done, total),
                    (index + done / max(1, total)) / 5)
                engine.generate_preview(copy.deepcopy(render_state), steps=params["steps"]).save(folder / item["name"])
            actual_steps, actual_cfg, _, _ = engine.sampling()
        return {"images": [{"name": item["name"], "label": item["label"]} for item in renders],
                "recipe": dict(params, steps=actual_steps, cfg=actual_cfg), "blocks": values,
                "exploration": {"config": config, "variants": variants,
                                "kernel": "Fizgig v7.0.1 SliderState.mutate"}}
    finally:
        temporary.unlink(missing_ok=True)


def profile(desc, pairs, source, info, options, folder):
    """Official quick rendered Profiler; this measures the original LoRA."""
    from fizgig.families.block_profile import run_ablation, conditions, verdicts, repair_settings
    params, config = options["preview"], options.get("profile") or {}
    if params["strength"] == 0:
        raise ValueError("Profiler 的 LoRA 使用强度不能为 0。")
    class_prompt = str(config.get("class_prompt") or "").strip()
    references = config.get("references") or []
    embed, baselines = None, []
    if references:
        if desc.key != "krea2":
            raise ValueError("Anima 首版使用图片对照，不提供人脸相似度评分。")
        # The upstream helper downloads on first use. Only enable it when its
        # detector and recognition files already exist; never install silently.
        face_dir = Path.home() / ".insightface" / "models" / "buffalo_l"
        if not all((face_dir / name).is_file() for name in ("det_10g.onnx", "w600k_r50.onnx")):
            raise ValueError("未安装 buffalo_l 人脸分析模型。可清空参考照片，仅分析画面变化；本任务不会自动下载。")
        from fizgig.lora_royale.likeness import available, embedding_for_path, embedding_for_image
        if not available():
            raise ValueError("当前 Fizgig 环境缺少人脸分析依赖，请清空参考照片后继续。")
        phase("正在分析参考照片（CPU）…")
        for path in references:
            vector = embedding_for_path(path)
            if vector is None:
                raise ValueError("参考照片未识别出可用人脸：" + Path(path).name + "。请选择清晰的单人照片，或清空参考照片。")
            baselines.append(vector)
        embed = embedding_for_image
    with preview_engine(desc, pairs, source, info, params) as engine:
        conds = conditions(engine, per_block=False)
        kinds = ["trigger"] + (["class"] if class_prompt else [])
        sequence = [(kind, cid, label) for kind in kinds for cid, label, _ in conds]
        total = len(sequence)
        manifest, names = [], {}
        generate = engine.generate_preview

        def render(state, **kwargs):
            index = len(manifest)
            kind, cid, _ = sequence[index]
            label = ("未含触发词 · " if kind == "class" else "") + (
                "原版 LoRA" if cid == "full" else "停用 LoRA" if cid == "none" else
                ("仅保留 " if cid.startswith("keep:") else "关闭 ") + next(l for w, l, _ in engine_windows if cid.split(":", 1)[1] == w))
            state.primary_scale = params["strength"]
            kwargs["steps"] = params["steps"]
            phase("Profiler %d / %d · %s" % (index + 1, total, label), index / total)
            engine.on_step = lambda done, steps: phase("Profiler %d / %d · %s · %d / %d" % (index + 1, total, label, done, steps), (index + done / max(1, steps)) / total)
            image = generate(state, **kwargs)
            name = "profile_%04d.png" % index
            image.save(folder / name)
            manifest.append({"name": name, "label": label})
            names[(kind, cid)] = name
            return image

        from fizgig.families.block_profile import windows
        engine_windows = windows(engine)
        engine.generate_preview = render
        try:
            abl = run_ablation(engine, prompt=params["prompt"], class_prompt=class_prompt,
                               seeds=(params["seed"],), width=params["width"], height=params["height"],
                               embed=embed, baselines=baselines, per_block=False,
                               on_progress=lambda done, count, *_: phase("Profiler 已完成 %d / %d 张" % (done, count), done / count))
        finally:
            engine.generate_preview = generate
        vd = verdicts(abl)
        suggested, _ = repair_settings(abl)
        values = {b["id"]: 1.0 for g in info["groups"] for b in g["blocks"]}
        for bid, (enabled, strength) in suggested.items():
            if bid in values:
                values[bid] = float(strength) if enabled else 0.0
        texts = {"quiet": "本次对照中影响较小，可尝试关闭", "bleed": "特征泄漏高于相似度贡献，建议优先减弱", "likeness": "对相似度有贡献，建议保留", "both": "相似度与特征泄漏并存，需看图确认", "look": "明显影响画面，需看图判断变化方向", "minor": "单独保留时影响较小"}
        condition_blocks = {cid: {bid: 1.0 if on is not None and bid in on else 0.0 for bid in values}
                            for cid, _, on in conds}
        def picture(kind, cid):
            return {"name": names[(kind, cid)], "thumbnail": "data:image/jpeg;base64," + abl["thumbs"][kind][cid],
                    "blocks": condition_blocks[cid]}
        results = []
        for wid, label, ids in abl["windows"]:
            alone = abl["alone"]["trigger"].get(wid, {})
            removed = abl["lost"]["trigger"].get("drop:" + wid, {})
            bleed = abl["alone"].get("class", {}).get(wid, {}).get("score")
            tag = vd[wid][0]
            results.append({"id": wid, "label": label, "blocks": ids, "tag": tag, "text": texts[tag],
                            "alone_change": alone.get("change"), "removed_change": removed.get("change"),
                            "likeness_share": alone.get("score"), "bleed_share": bleed,
                            "alone": picture("trigger", "keep:" + wid), "removed": picture("trigger", "drop:" + wid),
                            "class_alone": picture("class", "keep:" + wid) if class_prompt else None})
        warnings = ["结果仅针对本次提示词和种子；变化量不是画质评分，应用后请生成对照确认。"]
        if not references:
            warnings.append("未启用人脸评分，不判断角色相似度或特征泄漏。")
        elif any(score is None for kind in abl["scores"].values() for scores in kind.values() for score in scores):
            warnings.append("部分生成图未检测到人脸，相关相似度指标缺失。")
        if references and (abl["gain"].get("trigger") is None or abl["gain"]["trigger"] <= 0.03):
            warnings.append("本次 LoRA 的相似度增量不足，无法可靠归因到各组。")
        steps, cfg, _, _ = engine.sampling()
        return {"recipe": dict(params, steps=steps, cfg=cfg), "blocks": factors(info, options),
                "profile_images": manifest,
                "profile": {"mode": "quick", "scope": "original", "windows": results, "renders": total,
                            "baseline": picture("trigger", "full"), "off": picture("trigger", "none"),
                            "class_baseline": picture("class", "full") if class_prompt else None,
                            "class_off": picture("class", "none") if class_prompt else None,
                            "suggested_blocks": values, "suggestion_count": sum(v != 1 for v in values.values()),
                            "warnings": warnings, "class_prompt": class_prompt, "references": references}}


def main():
    # -I ignores PYTHONIOENCODING; configure streams explicitly on Windows.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.source) / "src"))
    job_path = Path(args.job)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    result_path = job_path.with_name("result.json")
    try:
        options = job["options"]
        phase("正在读取 LoRA 结构…")
        desc, pairs, info = inspect(options["source"], options.get("family", "auto"))
        if job["action"] != "inspect" and options.get("fingerprint") != info["fingerprint"]:
            raise ValueError("LoRA 文件已发生变化，请重新读取后再操作。")
        result = {"ok": True, "action": job["action"], "info": info}
        if job["action"] in ("export", "preview"):
            values = factors(info, options)
            if job["action"] == "export":
                phase("正在另存分层调整版…")
                bake(desc, pairs, options["source"], job["output"], info, values)
                result["blocks"] = values
            else:
                result.update(preview(desc, pairs, options["source"], info, values, options, job_path.parent))
        elif job["action"] == "explore":
            result.update(explore(desc, pairs, options["source"], info, options, job_path.parent))
        elif job["action"] == "profile":
            result.update(profile(desc, pairs, options["source"], info, options, job_path.parent))
        elif job["action"] == "analyze":
            from fizgig.families.block_profile import weight_stats
            phase("正在分析 Block 权重（CPU）…")
            stats = weight_stats(desc, options["source"])
            result["analysis"] = stats
        elif job["action"] == "compress":
            rank = int(options["rank"])
            if not info["can_compress"] or rank < 1 or rank >= max(info["ranks"]):
                raise ValueError("仅支持完整的线性 LoRA；目标 rank 必须低于原文件的最大 rank。")
            result["compression"] = compress(desc, pairs, options["source"], job["output"], rank)
        elif job["action"] != "inspect":
            raise ValueError("未知工作台操作。")
        # Catch a file being replaced while the worker read it.
        if fingerprint(options["source"]) != info["fingerprint"]:
            raise ValueError("操作期间源文件发生变化，结果未保存，请重新读取。")
        result_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        phase("处理完成", 1.0)
    except Exception as exc:
        result_path.write_text(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
