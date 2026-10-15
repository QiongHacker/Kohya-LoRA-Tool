"""Read native training counters and repair local trainer resume loops.

This module never loads model weights or starts a trainer. Accelerate/Fizgig
remain responsible for restoring weights, optimizer, scheduler and RNG.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re


def state_metadata(state_dir):
    """Use the trainer's native metadata; do not infer progress from LoRA files."""
    root = Path(state_dir)
    for name in ("tool_resume.json", "train_state.json", "training_state.json"):
        path = root / name
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(value, dict):
                raise ValueError("续训状态记录不是有效对象：" + str(path))
            return value
    return {}


def resume_position(state_dir, steps_per_epoch=None):
    meta = state_metadata(state_dir)
    for key in ("global_step", "current_step", "step"):
        if key in meta:
            step = int(meta[key])
            if step < 0:
                raise ValueError("续训步数不能为负数。")
            return step
    name = Path(state_dir).name
    step = re.search(r"-step(\d+)-state$", name)
    if step:
        return int(step.group(1))
    epoch = re.search(r"-(\d{6})-state$", name)
    if epoch and steps_per_epoch:
        return int(epoch.group(1)) * int(steps_per_epoch)
    return 0


def state_timestamp(state_dir):
    """Accelerate uses train_state.json (Kohya), not training_state.json."""
    root = Path(state_dir)
    stamps = [p.stat().st_mtime for p in root.iterdir() if p.is_file()]
    return max(stamps) if stamps else root.stat().st_mtime


def state_missing(state_dir, engine):
    root = Path(state_dir)
    if not root.is_dir():
        return ["快照目录"]
    if engine == "fizgig":
        required = ("lora.safetensors", "optimizer.pt", "training_state.json", "rng.pt")
        return [name for name in required if not (root / name).is_file() or (root / name).stat().st_size == 0]
    required = ("optimizer.bin", "scheduler.bin")
    missing = [name for name in required if not (root / name).is_file() or (root / name).stat().st_size == 0]
    if not any(p.is_file() and p.stat().st_size for pattern in ("model*.safetensors", "pytorch_model*.bin") for p in root.glob(pattern)):
        missing.append("LoRA 训练权重")
    if not any(p.is_file() and p.stat().st_size for p in root.glob("random_states*.pkl")):
        missing.append("随机数状态")
    return missing


def require_state(state_dir, engine, steps_per_epoch=None):
    missing = state_missing(state_dir, engine)
    if missing:
        raise ValueError("续训快照不完整，缺少：%s。请选择完整训练状态，不能用成品 LoRA 代替。" % "、".join(missing))
    step = resume_position(state_dir, steps_per_epoch)
    if step <= 0:
        raise ValueError("快照没有有效的已完成步数，已停止续训。请提供原始完整快照。")
    return step


def find_accelerate_state(output_dir, output_name, engine):
    root = Path(output_dir)
    pattern = re.compile(re.escape(output_name) + (r"-(?:step\d+|\d{6})-state$" if engine == "musubi" else r"-step\d+-state$"))
    final = root / (output_name + ".safetensors")
    final_stamp = final.stat().st_mtime if final.is_file() else 0
    candidates = []
    for directory in (root, root / "snapshots"):
        if not directory.is_dir():
            continue
        for state in directory.iterdir():
            if not state.is_dir() or not pattern.fullmatch(state.name):
                continue
            try:
                if state_missing(state, engine):
                    continue
                stamp = state_timestamp(state)
                if stamp <= final_stamp:
                    continue
                step = resume_position(state)
                # Legacy Musubi states encode completed epochs in their name.
                order = step or int(re.search(r"-(\d{6})-state$", state.name).group(1))
                candidates.append((stamp, order, str(state)))
            except (OSError, ValueError, TypeError, AttributeError):
                continue
    return max(candidates)[2] if candidates else None


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("训练循环结构已变化，无法安全恢复断点；请更新引擎。")
    return source.replace(old, new, 1)


def _write_patch(path, source):
    compile(source, str(path), "exec")
    backup = path.with_name(path.name + ".kohya-resume.bak")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    staged = path.with_name(path.name + ".kohya-resume.tmp")
    staged.write_bytes(source.encode("utf-8"))
    os.replace(staged, path)


def _kohya_counter_metadata(source):
    marker = "KOHYA_TOOL_PATCH: kohya resume batching"
    if marker in source:
        return source
    dump = '                json.dump({"current_epoch": current_epoch.value, "current_step": current_step.value + 1}, f)'
    extended = '                # KOHYA_TOOL_PATCH: kohya resume batching\n                json.dump({"current_epoch": current_epoch.value, "current_step": current_step.value + 1,\n                           "steps_per_epoch": num_update_steps_per_epoch,\n                           "batches_per_epoch": len(train_dataloader),\n                           "gradient_accumulation_steps": args.gradient_accumulation_steps}, f)'
    source = _replace_once(source, dump, extended)
    read_step = '                steps_from_state = data["current_step"]'
    checked = '                if ("batches_per_epoch" in data and\n                        (data["batches_per_epoch"] != len(train_dataloader) or\n                         data["gradient_accumulation_steps"] != args.gradient_accumulation_steps)):\n                    raise RuntimeError("续训的数据批次设置已变化，请使用原图集、重复次数和批大小。")\n                steps_from_state = data["current_step"]'
    return _replace_once(source, read_step, checked)


def patch_kohya_resume(kdir, logf=print):
    """Keep absolute optimizer steps and skip consumed batches on a resume."""
    path = Path(kdir) / "sd-scripts/train_network.py"
    marker = "KOHYA_TOOL_PATCH: kohya resume absolute v2"
    source = path.read_text(encoding="utf-8-sig")
    if marker in source:
        extended = _kohya_counter_metadata(source)
        if extended != source:
            _write_patch(path, extended)
        return
    begin = "        # calculate steps to skip when resuming or starting from a specific step\n"
    end = "        noise_scheduler = self.get_noise_scheduler(args, accelerator.device)"
    if source.count(begin) != 1 or source.count(end) != 1:
        raise ValueError("当前 Kohya 训练循环不支持安全恢复绝对步数，请更新第一引擎。")
    start = source.index(begin)
    finish = source.index(end, start)
    original = source[start:finish].rstrip("\n")
    if "steps_from_state" not in original or "global_step = 0" not in original:
        raise ValueError("Kohya 续训计数接口已变化，请更新第一引擎。")
    native = "\n".join("    " + line if line else "" for line in original.splitlines())
    resumed = '''        # KOHYA_TOOL_PATCH: kohya resume absolute v2
        if args.resume:
            restored_step = args.initial_step if args.initial_step is not None else steps_from_state
            if restored_step is None or not 0 < int(restored_step) < args.max_train_steps:
                raise RuntimeError("续训快照没有有效步数，或已达到目标步数；没有启动从头训练。")
            global_step = int(restored_step)
            epoch_to_start, consumed_updates = divmod(global_step, num_update_steps_per_epoch)
            initial_step = consumed_updates * args.gradient_accumulation_steps
            current_step.value = global_step
            logger.info("[resume] restored global_step=%s; next epoch=%s; skip batches=%s",
                        global_step, epoch_to_start + 1, initial_step)
        else:
'''
    source = source[:start] + resumed + native + "\n\n" + source[finish:]
    source = _replace_once(source, "        if initial_step > 0:  # only if skip_until_initial_step is specified",
                           "        if initial_step > 0 and not args.resume:  # native fresh-run initial step")
    progress = '''        progress_bar = tqdm(
            range(args.max_train_steps - initial_step), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps"
        )'''
    source = _replace_once(source, progress, '''        if args.resume:
            progress_bar = tqdm(total=args.max_train_steps, initial=global_step, smoothing=0,
                                disable=not accelerator.is_local_main_process, desc="steps")
        else:
''' + "\n".join("    " + line for line in progress.splitlines()))
    first_sample = "        self.sample_images(accelerator, args, 0, global_step, accelerator.device, vae, tokenizers, text_encoder, unet)"
    source = _replace_once(source, first_sample, "        if not args.resume:\n    " + first_sample)
    source = _kohya_counter_metadata(source)
    _write_patch(path, source)
    logf("[续训] 已修复 Kohya 的绝对步数、轮内跳过和保存编号。")


_MUSUBI_POSITION = '''        # KOHYA_TOOL_PATCH: musubi resume epoch v2
        epoch_to_start = 0
        global_step = 0
        resume_batches = 0
        if args.resume:
            import re as _kohya_resume_re
            _meta_path = os.path.join(args.resume, "tool_resume.json")
            if os.path.isfile(_meta_path):
                with open(_meta_path, encoding="utf-8") as _stream:
                    _saved = json.load(_stream)
                if (_saved["steps_per_epoch"] != num_update_steps_per_epoch or
                        _saved["batches_per_epoch"] != len(train_dataloader) or
                        _saved["gradient_accumulation_steps"] != args.gradient_accumulation_steps):
                    raise RuntimeError("续训的数据批次设置已变化，请使用原图集、重复次数和批大小。")
                global_step = int(_saved["global_step"])
            else:
                _resume_name = os.path.basename(os.path.normpath(args.resume))
                _step_match = _kohya_resume_re.search(r"-step(\\d+)-state$", _resume_name)
                _epoch_match = _kohya_resume_re.search(r"-(\\d{6})-state$", _resume_name)
                if _step_match:
                    global_step = int(_step_match.group(1))
                elif _epoch_match:
                    global_step = int(_epoch_match.group(1)) * num_update_steps_per_epoch
                else:
                    raise RuntimeError("续训快照没有已保存的步数或轮次；没有启动从头训练。")
            if not 0 < global_step < args.max_train_steps:
                raise RuntimeError("快照已达到目标步数；继续训练请提高目标总轮数。")
            epoch_to_start, _consumed_updates = divmod(global_step, num_update_steps_per_epoch)
            resume_batches = _consumed_updates * args.gradient_accumulation_steps
            logger.info("[resume] restored global_step=%s; next epoch=%s; skip batches=%s",
                        global_step, epoch_to_start + 1, resume_batches)
        progress_bar = tqdm(total=args.max_train_steps, initial=global_step, smoothing=0,
                            disable=not accelerator.is_local_main_process, desc="steps")

        def _kohya_save_position(state_dir):
            # Commit counters after Accelerate has saved all model/optimizer/RNG files.
            _record = {"global_step": global_step, "steps_per_epoch": num_update_steps_per_epoch,
                       "batches_per_epoch": len(train_dataloader),
                       "gradient_accumulation_steps": args.gradient_accumulation_steps}
            _target = os.path.join(state_dir, "tool_resume.json")
            with open(_target + ".tmp", "w", encoding="utf-8") as _stream:
                json.dump(_record, _stream)
            os.replace(_target + ".tmp", _target)

'''


def patch_musubi_resume(kdir, logf=print):
    path = Path(kdir) / "musubi-tuner/src/musubi_tuner/training/trainer_base.py"
    source = path.read_text(encoding="utf-8-sig")
    marker = "KOHYA_TOOL_PATCH: musubi resume epoch v2"
    if marker in source:
        return
    old_marker = "        # KOHYA_TOOL_PATCH: musubi resume epoch\n"
    native_marker = "        # TODO skip until initial step\n"
    begin = old_marker if old_marker in source else native_marker
    end = "        noise_scheduler = FlowMatchDiscreteScheduler(shift=args.discrete_flow_shift, reverse=True, solver=\"euler\")"
    if source.count(begin) != 1:
        raise ValueError("当前 Musubi 训练循环不支持安全恢复轮次，请更新第二引擎。")
    start = source.index(begin)
    finish = source.find(end, start)
    if finish <= start:
        raise ValueError("Musubi 训练循环结构已变化。")
    source = source[:start] + _MUSUBI_POSITION + source[finish:]
    loop = "            for step, batch in enumerate(train_dataloader):"
    source = _replace_once(source, loop, '''            _resume_skip = resume_batches if epoch == epoch_to_start else 0
            _epoch_loader = accelerator.skip_first_batches(train_dataloader, _resume_skip) if _resume_skip else train_dataloader
            for step, batch in enumerate(_epoch_loader, start=_resume_skip):''')
    for call, state_name in (
        ("train_utils.save_and_remove_state_stepwise(args, accelerator, global_step)", 'f"{args.output_name}-step{global_step:08d}-state"'),
        ("train_utils.save_and_remove_state_on_epoch_end(args, accelerator, epoch + 1)", 'f"{args.output_name}-{epoch + 1:06d}-state"'),
        ("train_utils.save_state_on_train_end(args, accelerator)", 'f"{args.output_name}-state"'),
    ):
        pattern = re.compile(r"^( +)" + re.escape(call) + r"$", re.MULTILINE)
        if len(pattern.findall(source)) != 1:
            raise ValueError("Musubi 状态保存接口已变化，请更新第二引擎。")
        source = pattern.sub(lambda m: m.group(0) + "\n" + m.group(1) +
                            "_kohya_save_position(os.path.join(args.output_dir, " + state_name + "))", source)
    _write_patch(path, source)
    logf("[续训] 已修复 Musubi 的轮次、步数和状态保存；旧 epoch 快照仍可恢复。")
