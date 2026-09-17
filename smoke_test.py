# -*- coding: utf-8 -*-
"""冒烟测试：每次改动/打包前运行，验证核心配置完整性与基本链路。

用法：python smoke_test.py
返回码 0=通过；1=失败（会打印具体问题）。
覆盖：
  1. 语法检查（Kohya一键工具.py / kohya_gui.py / preprocess.py / video_caption.py）
  2. 配置完整性：所有模式的 PRESETS / GUIDE_STEPS / OUTPUT_NAMES / MIN_IMAGES / DATASET_TIPS
  3. AI 图像模型配置（AT_IMAGE_MODELS：arch/模型/显存提示）
  4. yaml 生成可解析（Qwen/Z-Image/H3）
  5. 导入验证（Kohya一键工具）
"""
import io
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
# Windows 控制台默认 GBK，打印 ✔/✘/中文会 UnicodeEncodeError，统一转 UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

FAILED = []


def check(name, fn):
    try:
        fn()
        print("  [ok] %s" % name)
    except Exception as e:
        FAILED.append(name)
        print("  [FAIL] %s: %s" % (name, e))
        traceback.print_exc()


def test_syntax():
    import py_compile
    files = ["Kohya一键工具.py", "kohya_gui.py", "preprocess.py", "video_caption.py",
             "gui/__init__.py", "gui/tag_tools.py",
             "kohya_core/tagging/__init__.py", "kohya_core/tagging/dictionary.py",
             "kohya_core/tagging/normalize.py", "kohya_core/tagging/translate.py",
             "kohya_core/tagging/complete.py", "kohya_core/tagging/test_tagging.py",
             "kohya_core/anima_ckpt.py", "test_anima_ckpt.py"]
    for f in files:
        py_compile.compile(os.path.join(ROOT, f), doraise=True)


def test_import_core():
    import Kohya一键工具 as core
    if not hasattr(core, "MODE_KEYS"):
        raise AssertionError("MODE_KEYS 缺失")
    return core


def test_config_completeness():
    core = test_import_core()
    for mode in core.MODE_KEYS:
        # PRESETS 每个 base_type 齐全
        presets = core.PRESETS.get(mode)
        if not presets:
            raise AssertionError("PRESETS 缺 %s" % mode)
        for bt in ("sd15", "sdxl", "flux", "anima"):
            if bt not in presets:
                raise AssertionError("PRESETS[%s] 缺 %s" % (mode, bt))
        # GUIDE_STEPS
        if mode not in core.GUIDE_STEPS:
            raise AssertionError("GUIDE_STEPS 缺 %s" % mode)
        # OUTPUT_NAMES
        if mode not in core.OUTPUT_NAMES:
            raise AssertionError("OUTPUT_NAMES 缺 %s" % mode)
        # MIN_IMAGES
        if mode not in core.MIN_IMAGES:
            raise AssertionError("MIN_IMAGES 缺 %s" % mode)
        # DATASET_TIPS
        if mode not in core.DATASET_TIPS:
            raise AssertionError("DATASET_TIPS 缺 %s" % mode)
        # 引导步骤 id 唯一
        ids = [s["id"] for s in core.GUIDE_STEPS[mode]]
        if len(ids) != len(set(ids)):
            raise AssertionError("GUIDE_STEPS[%s] 步骤 id 重复" % mode)
        for s in core.GUIDE_STEPS[mode]:
            for k in ("id", "label", "btn", "check", "act", "tip"):
                if k not in s:
                    raise AssertionError("GUIDE_STEPS[%s] 步骤缺字段 %s" % (mode, k))


def test_at_image_models():
    core = test_import_core()
    for mode in ("qwen_image", "zimage"):
        info = core.AT_IMAGE_MODELS.get(mode)
        if not info:
            raise AssertionError("AT_IMAGE_MODELS 缺 %s" % mode)
        for k in ("label", "arch", "model_id", "min_vram", "rec_vram", "size", "hint"):
            if k not in info:
                raise AssertionError("AT_IMAGE_MODELS[%s] 缺 %s" % (mode, k))


def test_download_models():
    core = test_import_core()
    # FLUX 四件套
    for k in ("dit", "clip_l", "t5xxl", "ae"):
        if k not in core.FLUX_MODEL_LINKS:
            raise AssertionError("FLUX_MODEL_LINKS 缺 %s" % k)
        v = core.FLUX_MODEL_LINKS[k]
        if len(v) != 3 or not str(v[2]).startswith("http"):
            raise AssertionError("FLUX_MODEL_LINKS[%s] 格式错误" % k)
    if not callable(core.flux_missing_models):
        raise AssertionError("flux_missing_models 缺失")
    # Anima DiT 底模可应用内下载
    anima = core.get_download_models("anima")
    if not anima or not str(anima[0].get("url", "")).startswith("http"):
        raise AssertionError("DOWNLOAD_MODELS 缺 anima 应用内下载")
    # Krea2 文件齐全（含可选 turbo）
    if len(core.KREA2_MODEL_LINKS) < 4:
        raise AssertionError("KREA2_MODEL_LINKS 不完整")
    # FLUX.2 三件套（DiT / Qwen3 文本编码器 / VAE）
    for k in ("dit", "te", "vae"):
        if k not in core.FLUX2_MODEL_LINKS:
            raise AssertionError("FLUX2_MODEL_LINKS 缺 %s" % k)
        v = core.FLUX2_MODEL_LINKS[k]
        if len(v) != 3 or not str(v[2]).startswith("http"):
            raise AssertionError("FLUX2_MODEL_LINKS[%s] 格式错误" % k)
    if not callable(core.flux2_missing_models):
        raise AssertionError("flux2_missing_models 缺失")


def test_yaml():
    import tempfile
    import yaml
    core = test_import_core()
    tmp = tempfile.mkdtemp()
    vd = os.path.join(tmp, "img")
    os.makedirs(vd, exist_ok=True)
    params = {"project": "冒烟", "rank": "16", "alpha": "16", "unet_lr": "1e-4",
              "video_steps": "2000", "trigger": "myoc", "resolution": "1024"}
    # AI 图像 yaml
    for mode in ("qwen_image", "zimage"):
        cfg = os.path.join(tmp, mode + ".yaml")
        core.write_at_image_yaml(params, core.AT_IMAGE_MODELS[mode], vd, tmp, cfg)
        d = yaml.safe_load(open(cfg, encoding="utf-8"))
        if d["config"]["process"][0]["model"]["arch"] != core.AT_IMAGE_MODELS[mode]["arch"]:
            raise AssertionError("%s yaml arch 不符" % mode)
    # Z-Image 8G 快跑档（2026-09-06）：分辨率钳到 512 + 关采样 + 量化 TE + weighted（官方 zimage 预设）
    cfg = os.path.join(tmp, "zimage_8g.yaml")
    core.write_at_image_yaml(dict(params, resolution="1024"), core.AT_IMAGE_MODELS["zimage"], vd, tmp, cfg, vram_gb=8)
    d = yaml.safe_load(open(cfg, encoding="utf-8"))
    p0 = d["config"]["process"][0]
    _ram8 = core.detect_ram_gb() or 0
    _exp8 = 384 if _ram8 < 32 else 512
    if p0["datasets"][0]["resolution"] != [_exp8, _exp8]:
        raise AssertionError("Z-Image 8G 分辨率钳制不符: %s (ram %sG)" % (p0["datasets"][0]["resolution"], _ram8))
    if p0["train"].get("disable_sampling") is not True:
        raise AssertionError("Z-Image 8G 未关闭采样")
    if p0["train"].get("timestep_type") != "weighted":
        raise AssertionError("Z-Image 8G timestep 非 weighted")
    if p0["model"].get("quantize_te") is not True or p0["model"].get("qtype_te") != "qfloat8":
        raise AssertionError("Z-Image 8G TE 未量化")
    if p0["model"].get("layer_offloading") is not True or p0["model"].get("layer_offloading_transformer_percent") != 0.6:
        raise AssertionError("Z-Image 8G 未开层交换")
    # 16G 不启用快跑档（保持原行为，避免误伤现有配置）
    cfg = os.path.join(tmp, "zimage_16g.yaml")
    core.write_at_image_yaml(dict(params, resolution="1024"), core.AT_IMAGE_MODELS["zimage"], vd, tmp, cfg, vram_gb=16)
    d = yaml.safe_load(open(cfg, encoding="utf-8"))
    p0 = d["config"]["process"][0]
    if p0["datasets"][0]["resolution"] != [1024, 1024] or p0["train"].get("disable_sampling") is True or p0["model"].get("layer_offloading") is True:
        raise AssertionError("Z-Image 16G 误启用快跑档")
    # 手动开关：16G 强制开（fast_tier=on）→ 快跑档生效；8G 强制关（fast_tier=off）→ 完全常规
    cfg = os.path.join(tmp, "zimage_16g_on.yaml")
    core.write_at_image_yaml(dict(params, resolution="1024", fast_tier="on"), core.AT_IMAGE_MODELS["zimage"], vd, tmp, cfg, vram_gb=16)
    d = yaml.safe_load(open(cfg, encoding="utf-8"))
    p0 = d["config"]["process"][0]
    if p0["train"].get("disable_sampling") is not True or p0["model"].get("layer_offloading") is not True:
        raise AssertionError("Z-Image 16G fast_tier=on 未生效")
    if p0["datasets"][0]["resolution"] != [_exp8, _exp8]:
        raise AssertionError("Z-Image 16G fast_tier=on 分辨率钳制不符")
    cfg = os.path.join(tmp, "zimage_8g_off.yaml")
    core.write_at_image_yaml(dict(params, resolution="1024", fast_tier="off"), core.AT_IMAGE_MODELS["zimage"], vd, tmp, cfg, vram_gb=8)
    d = yaml.safe_load(open(cfg, encoding="utf-8"))
    p0 = d["config"]["process"][0]
    if p0["train"].get("disable_sampling") is True or p0["model"].get("layer_offloading") is True:
        raise AssertionError("Z-Image 8G fast_tier=off 误启用快跑档")
    if p0["datasets"][0]["resolution"] != [1024, 1024]:
        raise AssertionError("Z-Image 8G fast_tier=off 分辨率被误钳制")
    # Qwen-Image fast_tier=on：强制快跑档生效 + 分辨率 512
    cfg = os.path.join(tmp, "qwen_16g_on.yaml")
    core.write_at_image_yaml(dict(params, resolution="1024", fast_tier="on"), core.AT_IMAGE_MODELS["qwen_image"], vd, tmp, cfg, vram_gb=16)
    d = yaml.safe_load(open(cfg, encoding="utf-8"))
    p0 = d["config"]["process"][0]
    if p0["train"].get("disable_sampling") is not True or p0["model"].get("layer_offloading") is not True:
        raise AssertionError("Qwen fast_tier=on 未生效")
    if p0["datasets"][0]["resolution"] != [512, 512]:
        raise AssertionError("Qwen fast_tier=on 分辨率钳制不符")
    # H3 yaml
    cfg = os.path.join(tmp, "h3.yaml")
    core.write_h3_train_yaml(params, vd, tmp, cfg)
    d = yaml.safe_load(open(cfg, encoding="utf-8"))
    if d["config"]["process"][0]["model"]["arch"] != "minimax_h3":
        raise AssertionError("H3 yaml arch 不符")
    # Krea2（AI-Toolkit 引擎）yaml：16G → qint8+768+low_vram+关采样；24G → qfloat8+1024
    import tempfile as _tf
    _td = _tf.mkdtemp(prefix="k2at_")
    try:
        _raw = os.path.join(_td, "raw.safetensors")
        open(_raw, "wb").write(b"x" * 1024)
        _old_files = core.krea2_model_files
        core.krea2_model_files = lambda: {"raw": _raw, "vae": None, "te": None, "turbo": None}
        _old_count = core.count_images
        core.count_images = lambda *a, **k: 3
        try:
            cfg = os.path.join(tmp, "krea2_at16.yaml")
            core.write_krea2_at_yaml(dict(params, resolution="1024", sample_preview=False), vd, tmp, cfg, vram_gb=16)
            d = yaml.safe_load(open(cfg, encoding="utf-8"))
            p0 = d["config"]["process"][0]
            if p0["model"]["arch"] != "krea2" or p0["model"]["qtype"] != "qint8":
                raise AssertionError("Krea2(AT) 16G yaml 档位不符")
            if p0["datasets"][0]["resolution"] != [512, 512]:
                raise AssertionError("Krea2(AT) 16G 未按快档压到 512")
            if p0["train"].get("disable_sampling") is not True or "sample" not in p0:
                raise AssertionError("Krea2(AT) 16G 需保留 sample 段 + disable_sampling（引擎 cache_sample_prompts 会崩）")
            if p0.get("sample", {}).get("negative_prompt") != "lowres, bad anatomy, worst quality, low quality, blurry, jpeg artifacts, signature, watermark":
                raise AssertionError("Krea2(AT) 负向提示词不应为空(空串会被引擎当 bool 崩)")

            if p0["model"].get("quantize_te") is not True:
                raise AssertionError("Krea2(AT) 16G 应量化文本编码器（0.13 快档配方）")
            # 死区回归：16.0x / 17~23G 也必须拿到省显存组合。
            # 历史 bug：判定写 vram_gb <= 16，16 卡若报成 16.01 就退化成
            # qfloat8 + 1024 + 无 layer_offloading（16G 必 OOM，且连重试兜底都拿不到）。
            for _v in (16.01, 16.6, 17, 18.9):
                cfg = os.path.join(tmp, "krea2_at_%s.yaml" % _v)
                core.write_krea2_at_yaml(dict(params, resolution="1024", sample_preview=False),
                                         vd, tmp, cfg, vram_gb=_v)
                d = yaml.safe_load(open(cfg, encoding="utf-8"))
                p0 = d["config"]["process"][0]
                if p0["model"]["qtype"] != "qint8" or p0["datasets"][0]["resolution"] != [512, 512]:
                    raise AssertionError("Krea2(AT) %sG 未走省显存档（阈值悬崖）" % _v)
                if p0["model"].get("layer_offloading") is not True:
                    raise AssertionError("Krea2(AT) %sG 缺少 layer_offloading（死区）" % _v)
                if p0["model"].get("layer_offloading_transformer_percent") is None:
                    raise AssertionError("Krea2(AT) %sG 分层交换比例缺失" % _v)
            cfg = os.path.join(tmp, "krea2_at24.yaml")
            core.write_krea2_at_yaml(dict(params, resolution="1024", sample_preview=True), vd, tmp, cfg, vram_gb=24)
            d = yaml.safe_load(open(cfg, encoding="utf-8"))
            p0 = d["config"]["process"][0]
            if p0["model"]["qtype"] != "qfloat8" or p0["datasets"][0]["resolution"] != [1024, 1024]:
                raise AssertionError("Krea2(AT) 24G yaml 档位不符")
            if p0["model"].get("quantize_te") is not True:
                raise AssertionError("Krea2(AT) 24G 应量化文本编码器")
        finally:
            core.krea2_model_files = _old_files
            core.count_images = _old_count
    finally:
        import shutil as _sh
        _sh.rmtree(_td, ignore_errors=True)

def test_tagging():
    """标签管理 v1 · 离线中英词典核心链路（加载/翻译/补全/中文反查/GUI 模块可导入）。"""
    from kohya_core.tagging import TagDict
    from kohya_core.tagging import normalize, translate
    d = TagDict()
    if not d.available():
        raise AssertionError("缺少离线词典文件: installers/tag_dict/danbooru_zh.tsv")
    if len(d) < 150000:
        raise AssertionError("离线词条过少: %d" % len(d))
    if d.to_zh("hatsune_miku") != "初音未来":
        raise AssertionError("英→中 翻译错误: hatsune_miku")
    if d.to_zh("blue hair") != d.to_zh("blue_hair"):
        raise AssertionError("空格写法未规范化命中")
    if normalize.norm_en("Blue Hair") != "blue_hair":
        raise AssertionError("normalize 错误")
    if not d.zh_candidates("初音") or d.zh_candidates("初音")[0][0] != "hatsune_miku":
        raise AssertionError("中→英 反查错误: 初音")
    if not any(r[0] == "blue_hair" for r in d.complete_en("blue", limit=50)):
        raise AssertionError("英文补全缺 blue_hair")
    import gui.tag_tools  # GUI 辅助模块可正常导入

def test_anima_ckpt():
    """Anima 合并包剥离工具：识别/剥离/缓存（合成 safetensors，零依赖）。"""
    import json, os, struct, tempfile
    from kohya_core import anima_ckpt as ac

    def _w(path, keys):
        h = {"__metadata__": {}}
        off, payload = 0, b""
        for i, k in enumerate(keys):
            h[k] = {"dtype": "F32", "shape": [1], "data_offsets": [off, off + 4]}
            payload += struct.pack("<f", float(i + 1))
            off += 4
        blob = json.dumps(h, separators=(",", ":")).encode("utf-8")
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(blob)) + blob + payload)

    d = tempfile.mkdtemp()
    pure = os.path.join(d, "pure.safetensors")
    merged = os.path.join(d, "merged.safetensors")
    _w(pure, ["net.0.weight", "net.1.weight"])
    _w(merged, ["net.0.weight", "net.1.weight", "cond_stage_model.qwen3_06b.w"])
    assert ac.checkpoint_kind(pure) == "pure"
    assert ac.checkpoint_kind(merged) == "merged"
    out = os.path.join(d, "out.safetensors")
    nkeep, ndrop = ac.strip_to_dit(merged, out, ref_path=pure, logf=lambda *a: None)
    assert (nkeep, ndrop) == (2, 1), (nkeep, ndrop)
    assert ac.checkpoint_kind(out) == "pure"

def test_monitor_sampling_and_fizgig_resume():
    """采样预览进度不污染训练监控（防看门狗误杀）+ Fizgig 断点查找。"""
    import Kohya一键工具 as core
    mon = core.TrainMonitor()
    mon.start(total=3800)
    mon.on_line("steps:   8%| | 304/3800 [00:10<02:00, 4.17s/it, avr_loss=0.1]")
    s1 = mon.snapshot()
    assert s1.get("step") == 304 and s1.get("total") == 3800, s1
    mon.on_line("sampling:  62%| | 5/8 [01:25<00:59, 19.67s/it]")
    s2 = mon.snapshot()
    assert s2.get("step") == 304 and s2.get("total") == 3800, "采样行污染了监控: %s" % s2
    mon.on_line("rendering previews (epoch 2) on the fp8 Turbo...")
    s3 = mon.snapshot()
    assert s3.get("step") == 304, s3
    # 基线/收尾采样（Generating Samples: 0/1 …）同样不能污染步数，且必须推进 last_activity
    #（否则训练 100% 后的收尾采样超过 grace 会被"卡死看门狗"误杀，2026-09 AMD 用户复现）
    import time as _t
    t0 = s3.get("last_activity") or 0
    _t.sleep(0.05)
    mon.on_line("Generating baseline samples before training (step 304)")
    mon.on_line("Generating Samples:   0%|          | 0/1 [00:00<?, ?it/s]")
    s4 = mon.snapshot()
    assert s4.get("step") == 304 and s4.get("total") == 3800, "Generating Samples 污染监控: %s" % s4
    assert (s4.get("last_activity") or 0) > t0, "采样行未记为进程活动（看门狗会误杀）"
    # 断点续训：引擎按“剩余步数”从 0 重数（sd-scripts: range(max-initial)）→ 映射回绝对步
    mon2 = core.TrainMonitor()
    mon2.start(total=2000)
    mon2.set_step(800)
    mon2.on_line("steps:   6%| | 120/1200 [00:20<04:00, 4.17s/it, loss=0.12]")
    s5 = mon2.snapshot()
    assert s5.get("step") == 920 and s5.get("total") == 2000, "续训监控未映射回绝对步: %s" % s5
    mon2.on_line("steps:  50%| | 600/1200 [00:20<04:00, 4.17s/it, loss=0.11]")
    s6 = mon2.snapshot()
    assert s6.get("step") == 1400 and s6.get("total") == 2000, s6
    # Fizgig 断点查找：{name}-NNNNNN-state；有最终 LoRA 视为跑完不提示
    import tempfile, os as _os, json
    d = tempfile.mkdtemp()
    for ep in ("000001", "000002"):
        st = _os.path.join(d, "krea2_fizgig_lora-%s-state" % ep)
        _os.makedirs(st, exist_ok=True)
        with open(_os.path.join(st, "training_state.json"), "w", encoding="utf-8") as f:
            json.dump({"epoch": int(ep), "global_step": int(ep) * 152}, f)
    found = core.find_fizgig_state(d, "krea2_fizgig_lora")
    assert found and found.endswith("krea2_fizgig_lora-000002-state"), found
    open(_os.path.join(d, "krea2_fizgig_lora.safetensors"), "wb").write(b"x")
    assert core.find_fizgig_state(d, "krea2_fizgig_lora") is None, "跑完仍提示续训"
    # 旧成品 + 更新的新断点 → 仍应提示（修 find_fizgig_state 一刀切 bug，2026-09-08）
    st3 = _os.path.join(d, "krea2_fizgig_lora-000003-state")
    _os.makedirs(st3, exist_ok=True)
    with open(_os.path.join(st3, "training_state.json"), "w", encoding="utf-8") as f:
        json.dump({"epoch": 3, "global_step": 456}, f)
    _os.utime(st3, (9000, 9000))
    _os.utime(_os.path.join(d, "krea2_fizgig_lora.safetensors"), (1000, 1000))
    assert core.find_fizgig_state(d, "krea2_fizgig_lora") is not None, "旧成品+新断点未提示续训"
    # kohya/musubi 断点：有 -step…-state 且无成品 → 提示续训；成品已生成 → 不提示
    import tempfile as _tf2, os as _os2
    d2 = _tf2.mkdtemp()
    st2 = _os2.path.join(d2, "character_lora-step00000400-state")
    _os2.makedirs(st2, exist_ok=True)
    assert core.find_latest_state(d2, "character_lora") is not None, "中断点应提示续训"
    _os2.utime(st2, (2000, 2000))
    with open(_os2.path.join(d2, "character_lora.safetensors"), "wb") as f:
        f.write(b"x")
    _os2.utime(_os2.path.join(d2, "character_lora.safetensors"), (3000, 3000))
    assert core.find_latest_state(d2, "character_lora") is None, "跑完仍提示续训(kohya)"
    # 旧成品 + 更新的新中断 → 仍应提示（只看最终成品 vs 断点 mtime）
    _os2.utime(st2, (9000, 9000))
    assert core.find_latest_state(d2, "character_lora") is not None, "旧成品+新中断未提示续训(kohya)"


def test_lora_naming():
    """训练完成按项目名导出成品：挑最新成品、复制为 <项目名>.safetensors、原文件保留。"""
    import tempfile
    from kohya_core import lora_naming as ln

    d = tempfile.mkdtemp()
    for name, m in (("krea2_lora-000006.safetensors", 1000), ("krea2_lora-000008.safetensors", 2000)):
        p = os.path.join(d, name)
        with open(p, "wb") as f:
            f.write(b"\0" * 16)
        os.utime(p, (m, m))
    logs = []
    got = ln.export_project_named_lora("krea2", "测试项目", logf=logs.append, out_dir=d)
    assert got == os.path.join(d, "测试项目.safetensors") and os.path.isfile(got), logs
    assert os.path.isfile(os.path.join(d, "krea2_lora-000008.safetensors"))  # 原文件保留（续训/已完成检测仍认它）


def test_project_data_cleanup():
    """删除项目要能一并清掉图集数据（含打标文件），并能清理已经遗留的孤儿数据。

    2026-09-15 用户反馈：删了项目，打标好的文件还留在磁盘上一直占空间。
    根因：delete_project() 只删 projects/<名>.json，data/dataset/<项目名>/
    （预处理图片 + .txt 打标 + 各引擎缓存）原样留下 —— 而项目一删，
    这批数据再没有任何界面入口能找到它。
    """
    import tempfile
    import shutil
    import Kohya一键工具 as core
    from kohya_core import paths as _P

    tmp = tempfile.mkdtemp()
    _real_data_dir = _P.data_dir
    # ⚠️ 必须改 kohya_core.paths 里的引用：data_sub/projects_dir 等查的是
    # 本模块 globals，改 Kohya一键工具.data_dir 对它们无效（会写进真实数据目录）。
    _P.data_dir = lambda: tmp
    try:
        core.save_project("测试项目", {"name": "测试项目", "mode": "style"})
        ds = core.project_data_dir("测试项目")
        os.makedirs(os.path.join(ds, "train_character"), exist_ok=True)
        for i in range(3):
            open(os.path.join(ds, "train_character", "a%d.png" % i), "wb").write(b"x" * 100)
            open(os.path.join(ds, "train_character", "a%d.txt" % i), "w", encoding="utf-8").write("1girl")
        os.makedirs(os.path.join(ds, "krea2_cache"), exist_ok=True)
        open(os.path.join(ds, "krea2_cache", "c.bin"), "wb").write(b"y" * 50)
        assert core.dir_stats(ds) == (7, 365), core.dir_stats(ds)

        # 复现原问题：删项目后图集数据仍在（既有行为，正是用户踩到的）
        core.delete_project("测试项目")
        assert os.path.isdir(ds), "前置条件不符：删项目后图集数据应仍在"

        # 新函数：清图集数据；output 训练产物不归它管，必须原样保留
        out_d = core.project_output_dir("测试项目")
        os.makedirs(out_d, exist_ok=True)
        open(os.path.join(out_d, "成品.safetensors"), "wb").write(b"z")
        ok, n, b = core.delete_project_data("测试项目")
        assert ok and (n, b) == (7, 365), (ok, n, b)
        assert not os.path.isdir(ds), "图集数据未删净"
        assert os.path.isdir(out_d), "delete_project_data 不该动训练产物目录"
        assert core.delete_project_data("测试项目")[0] is True, "重复删除应幂等"
        assert core.delete_project_data("")[0] is True, "空项目名必须安全返回"

        # ---- 孤儿（已删项目遗留）检测 ----
        os.makedirs(os.path.join(tmp, "dataset", "train_character"), exist_ok=True)   # 旧版共享目录
        os.makedirs(os.path.join(tmp, "dataset", "已删项目", "train"), exist_ok=True)
        open(os.path.join(tmp, "dataset", "已删项目", "train", "x.txt"), "w", encoding="utf-8").write("t")
        core.save_project("活项目", {"name": "活项目"})
        os.makedirs(os.path.join(tmp, "dataset", "活项目", "train_character"), exist_ok=True)
        names = [x[0] for x in core.find_orphan_project_dirs()]
        assert "已删项目" in names, names
        assert "活项目" not in names, "有对应项目的目录被误判成孤儿：%s" % names
        assert "train_character" not in names, "旧版共享目录被误判成孤儿：%s" % names

        ok_n, files, size, failed = core.delete_orphan_project_dirs()
        assert (ok_n, files, size, failed) == (1, 1, 1, []), (ok_n, files, size, failed)
        assert not os.path.isdir(os.path.join(tmp, "dataset", "已删项目"))
        assert os.path.isdir(os.path.join(tmp, "dataset", "活项目")), "误删了活项目的数据"
        assert os.path.isdir(os.path.join(tmp, "dataset", "train_character")), "误删了旧版共享目录"
    finally:
        _P.data_dir = _real_data_dir
        shutil.rmtree(tmp, ignore_errors=True)


def test_label_editor_safety():
    """标签批量操作的两道闸：预演（绝不写盘）+ 快照/撤销（字节级还原）。

    2026-09-15 用户反馈：多选标签时忘了按 Ctrl，把前面选择要删的标签也删了。
    根因是批量删除/替换直接覆写全部 .txt，既无确认也无撤销。
    """
    import tempfile
    import shutil
    from pathlib import Path
    import Kohya一键工具 as core
    from kohya_core import paths as _P

    tmp = tempfile.mkdtemp()
    _real_data_dir = _P.data_dir
    _P.data_dir = lambda: tmp          # 与 paths 内部保持一致，别写进真实数据目录
    try:
        ds = os.path.join(tmp, "dataset", "proj", "train_character")
        os.makedirs(ds)
        # ⚠️ list_dataset_images **以图片为驱动**（只遍历图片再配对同名 .txt），
        # 所以只有 .txt 没有配对图片的文件根本不会被列出 —— 每个用例都要有同名图片。
        raw = {                                   # stem -> 原始内容
            "a": "1girl, solo, blue_hair\n",      # 带换行
            "b": "1girl, solo",                   # 无换行
            "c": "solo, blue_hair \n",            # 带尾随空格
        }
        for stem, txt in raw.items():
            open(os.path.join(ds, stem + ".txt"), "w", encoding="utf-8", newline="").write(txt)
            open(os.path.join(ds, stem + ".png"), "wb").write(b"x")
        open(os.path.join(ds, "d.png"), "wb").write(b"x")   # 没有 txt 的图不该被算进去

        def _tp(stem):
            return os.path.join(ds, stem + ".txt")

        def _read(stem):
            return open(_tp(stem), encoding="utf-8", newline="").read()

        # ---- ① dry_run 只统计，绝不写盘 ----
        assert core.batch_remove_tags(ds, "solo", dry_run=True) == (3, 3)
        for stem, txt in raw.items():
            assert _read(stem) == txt, "dry_run 改写了文件 %s" % stem
        assert core.batch_remove_tags(ds, "不存在的标签", dry_run=True) == (0, 0)

        # ---- ② snapshot 记的是原始内容，同时正常写盘 ----
        snap = {}
        assert core.batch_remove_tags(ds, "solo", snapshot=snap) == (3, 3)
        for stem, txt in raw.items():
            assert snap[_tp(stem)] == txt, "快照不是原始内容：%r" % snap.get(_tp(stem))
            assert "solo" not in _read(stem), "没删干净 %s" % stem

        # ---- ③ 撤销：字节级还原（含无换行 / 尾随空格两种形态）----
        assert core.restore_captions(snap) == 3
        for stem, txt in raw.items():
            assert _read(stem) == txt, "撤销未还原 %s -> %r（应为 %r）" % (stem, _read(stem), txt)

        # ---- ④ 替换同样支持预演 + 快照 + 撤销 ----
        assert core.batch_replace_tags(ds, "blue_hair", "aqua_hair", dry_run=True) == 2
        assert _read("a") == raw["a"], "替换的 dry_run 改写了文件"
        snap2 = {}
        assert core.batch_replace_tags(ds, "blue_hair", "aqua_hair", snapshot=snap2) == 2
        assert "aqua_hair" in _read("a")
        assert core.restore_captions(snap2) == 2
        assert _read("a") == raw["a"], "替换的撤销未还原"

        # ---- ⑤ 工具自己写的文件（Windows 上 save_caption 写 CRLF）也必须字节级还原 ----
        # 这条是防回归：_read_caption_raw 若用默认 newline 读，CRLF 会被归一成 LF，
        # 撤销就把整个数据集的行尾悄悄改掉了。
        open(os.path.join(ds, "e.png"), "wb").write(b"x")
        p_e = os.path.join(ds, "e.txt")
        core.save_caption(p_e, "1girl, crlfmarker")
        e_raw = open(p_e, encoding="utf-8", newline="").read()
        snap3 = {}
        assert core.batch_remove_tags(ds, "crlfmarker", snapshot=snap3) == (1, 1)
        assert core.restore_captions(snap3) == 1
        assert open(p_e, encoding="utf-8", newline="").read() == e_raw, \
            "工具自写文件未字节级还原（行尾被改写）"

        # ---- ⑥ 源码接线：GUI 必须真的用了这两道闸 ----
        gsrc = Path(os.path.join(os.path.dirname(core.__file__), "kohya_gui.py")).read_text(encoding="utf-8-sig")
        csrc = Path(core.__file__).read_text(encoding="utf-8-sig")
        assert "dry_run=True" in gsrc, "批量删除/替换没有预演步骤"
        assert gsrc.count("snapshot=snapshot") >= 3, "不是所有不可逆批量操作都留了撤销快照"
        assert "def _push_undo" in gsrc and "def _do_undo" in gsrc, "缺少撤销实现"
        assert "def restore_captions" in csrc, "缺少还原实现"
        # 缩略图不再写死尺寸（用户反馈「太小、下面明明有不少空间」）
        assert "im.thumbnail((220, 140))" not in gsrc, "缩略图仍是硬编码 220x140"
        assert "def _on_right_resize" in gsrc, "缩略图未接自适应"
        # 多选开关：图片列表与标签统计窗都要有，且偏好持久化
        assert "label_editor_multi" in gsrc, "多选偏好未持久化"
        # 两处开关的文案都是纯「多选模式」（用户要求不要多余的括号说明）
        assert gsrc.count('text="多选模式"') >= 2, "图片列表/标签统计窗未都加多选开关"

        # ---- ⑦ 缩略图自适应的三条防回归（2026-09-15 实测踩过的坑，都很隐蔽）----
        import re as _re
        # ① 绑定必须 add="+"：CTkFrame 内部用 <Configure> 维护自己的 canvas/圆角，
        #    直接 bind 会把它顶掉，控件自身就画不出来。
        assert 'self._on_right_resize, add="+"' in gsrc, \
            "自适应绑定没用 add='+'，会覆盖 CTk 控件的内部 <Configure> 处理器"
        # ② _on_right_resize 里绝不许量其它控件的高度 —— 那会成环：
        #    缩略图高度 → body 的请求高度 → 底部工具条能否分到 pack 空间 → 工具条高度
        #    → 预留量 → 缩略图高度…… 实测 40 轮 update 触发回调 566 次（死循环），
        #    布局整体崩坏、工具条 unmapped，表现为「标签编辑器功能组件全没了」。
        _m = _re.search(r"    def _on_right_resize\(self.*?\n(?=    def )", gsrc, _re.S)
        assert _m, "找不到 _on_right_resize 实现"
        _body = _m.group(0)
        # 唯一允许的输入是**窗口自身**的尺寸；把窗口测量摘掉后，不应再有别的尺寸测量
        _stripped = (_body.replace("self.win.winfo_width()", "")
                          .replace("self.win.winfo_height()", ""))
        for _bad in ("winfo_width()", "winfo_height()", "winfo_reqheight()"):
            assert _bad not in _stripped, \
                "_on_right_resize 里量了其它控件的 %s —— 会形成布局死循环，绝不能加回来" % _bad
        # ③ 缩略图容器必须是原生 tk.Frame：CTkFrame 是复合控件，
        #    pack_propagate(False) 转发不到内部 canvas，实测设 1236x700 仍被撑到 1248x1017。
        assert "self.prev_box = tk.Frame(" in gsrc, "缩略图容器不是原生 tk.Frame（propgate 不生效）"
        assert "self.prev_box.pack_propagate(False)" in gsrc, "缩略图容器未禁止尺寸传播"
    finally:
        _P.data_dir = _real_data_dir
        shutil.rmtree(tmp, ignore_errors=True)


def test_home_output_button():
    """主页要有直达「输出目录」的入口，且打开目录的路径必须健壮。

    2026-09-16 用户反馈：想看训练出来的 LoRA，必须先打开某个项目，
    或者自己去 %APPDATA%\\KohyaLoraTool\\output 里翻，很费劲。

    两个关键点：
      ① 主页入口给的是**输出根目录**（不是某个项目）—— 一次看到全部项目，
         这才真正免除「先点开之前的项目」；
      ② 打开目录不能裸调 os.startfile —— 从没训练过时目录根本不存在，会静默失手。
    """
    from pathlib import Path
    import Kohya一键工具 as core
    gsrc = Path(os.path.join(os.path.dirname(core.__file__), "kohya_gui.py")).read_text(encoding="utf-8-sig")

    # 1) 主页头部有这个按钮，绑到输出根目录入口，并登记进 _home_widgets（否则切页会残留）
    i = gsrc.find("def _build_home(self)")
    j = gsrc.find("def _show_home(self)")
    assert i != -1 and j > i, "找不到 _build_home / _show_home"
    head = gsrc[i:j]
    assert "self.btn_output_dir = ctk.CTkButton(" in head, "主页头部缺少「输出目录」按钮"
    assert "command=self.cmd_open_output_root" in head, "主页按钮未绑定到输出根目录入口"
    assert "self._home_widgets.append(self.btn_output_dir)" in head, "按钮未登记进主页组件"

    # 2) 打开目录必须「先建目录 + 失败有提示」
    k = gsrc.find("def _open_dir_or_warn(self")
    assert k != -1, "缺少健壮的打开目录封装 _open_dir_or_warn"
    body = gsrc[k:k + 900]
    assert "os.makedirs(d, exist_ok=True)" in body, "打开前未确保目录存在（从没训练过会失手）"
    assert "messagebox.showerror(" in body, "打开失败未提示用户"

    # 3) 两个入口语义必须不同：主页 = 输出根目录；工作区 = 当前项目
    a = gsrc.find("def cmd_open_output_root(self")
    assert a != -1, "缺少 cmd_open_output_root"
    assert 'core.data_sub("output")' in gsrc[a:a + 800], "主页入口未指向输出根目录"
    b = gsrc.find("def cmd_open_output(self")
    assert b != -1, "缺少 cmd_open_output"
    assert 'core.data_sub("output", self.current_project)' in gsrc[b:b + 800], \
        "工作区入口未指向当前项目"

    # 4) 旧的裸 startfile 必须已消除（没 try/except，失败就静默）
    assert 'os.startfile(core.data_sub("output"))' not in gsrc, "仍残留裸 startfile"
    print("HOME_OUTPUT_BUTTON_OK")


def test_preprocess_progress():
    """预处理要有真实进度可看（WD14 打标是预处理里最长的一段）。

    2026-09-16 用户反馈：预处理期间界面没有进度反馈，总怀疑「是不是卡住了」。
    预处理跑在**子进程**（`preprocess.py`）里，父进程拿不到任何回调 ——
    唯一的信息通道是它的 stdout，所以解析 `[WD14] 内置打标进度：done/total`。

    三条必须守住的边界（写错了比不做更糟）：
      ① 别的阶段的日志（训练步数 / 下载字节 / 缓存 tqdm）**绝不能被当成预处理进度**
         —— 否则进度条会乱跳；
      ② 官方打标脚本走 tqdm（`\\r` 原地刷新、行里没有 N/M）时**不能假装 0% 进度**，
         只能报「阶段已开始 + 已运行多久」；
      ③ 两处调用点都要接 —— 单独「数据预处理」和「一键开始训练」里的自动预处理，
         不接的话「一键」反而更让人干等。
    """
    from pathlib import Path
    import Kohya一键工具 as core

    # ---- ① 解析真实进度行（preprocess.py 逐字） ----
    m = core.PreprocessMonitor()
    assert m.feed("[WD14] 内置打标进度：120/450（已写 120 张）") is True
    s = m.snapshot()
    assert (s["done"], s["total"]) == (120, 450), s
    assert s["stage"] == "内置打标", s["stage"]
    assert s["running"] is True
    m2 = core.PreprocessMonitor()          # 半角冒号也要认
    assert m2.feed("[WD14] 内置打标进度: 7/9") is True
    assert m2.snapshot()["total"] == 9
    m2b = core.PreprocessMonitor()          # 整块传入（兼容 chunk）
    assert m2b.feed("[预处理] x\n[WD14] 内置打标进度：5/80（已写 5 张）") is True
    assert m2b.snapshot()["total"] == 80

    # ---- ② 无关日志绝不能驱动（否则进度条乱跳） ----
    m3 = core.PreprocessMonitor()
    for junk in ("[训练] steps: 5%| 51/1024 [01:00<19:00, 2.10s/it]",
                 "[下载] 12.3 MB / 45.6 MB",
                 "caching latents: 20/20",
                 "[OK] 预处理完成"):
        assert m3.feed(junk) is False, "无关日志被当成预处理进度：%s" % junk
    s3 = m3.snapshot()
    assert s3["running"] is False and (s3["done"], s3["total"]) == (0, 0), s3
    assert m3.feed("") is False and m3.feed(None) is False

    # ---- ③ 官方脚本路径：在动、但无数字（不许假装 0%） ----
    m4 = core.PreprocessMonitor()
    assert m4.feed("[WD14] 使用官方打标脚本: /a/b/wd14.py") is True
    s4 = m4.snapshot()
    assert s4["running"] is True and s4["total"] == 0, s4

    # ---- ④ 完成行把进度补满（末批不足 5 张时不会打最后一条进度行） ----
    m5 = core.PreprocessMonitor()
    m5.feed("[WD14] 内置打标进度：9/12（已写 9 张）")
    m5.feed("[WD14] 内置打标完成：为 12 张图片生成标签")
    assert m5.snapshot()["done"] == 12, m5.snapshot()

    # ---- ⑤ GUI 接线 ----
    g = Path(os.path.join(os.path.dirname(core.__file__), "kohya_gui.py")).read_text(encoding="utf-8-sig")
    assert "def _begin_preprocess_progress" in g and "def _end_preprocess_progress" in g, "缺少预处理进度开关"
    assert "def _render_preprocess_progress" in g, "界面没有渲染预处理进度"
    assert g.count("_pp_log = self._begin_preprocess_progress()") >= 2, \
        "预处理进度只接了一处（单独预处理 / 一键训练里的自动预处理都要有）"
    assert g.count("self._end_preprocess_progress()") >= 2, "预处理结束后没有撤掉进度监控"
    tw = g[g.find("def _train_worker"):]
    assert "self._pp_mon = None" in tw[:700], "训练开始未清掉预处理监控（会闪旧进度）"
    assert 'text="📊 训练监控"' in tw[:1000], "训练开始未把面板标题拨回训练语义"
    print("PREPROCESS_PROGRESS_OK")


def test_python_env_source_guard():
    """建训练环境必须校验 Python 来源；状态徽章必须如实显示实际版本。

    2026-09-16 qiansui 用户实测链条：
      · 机器上只有 Anaconda 自带的 Python 3.11.7；
      · 工具用它建了训练环境（版本在允许范围内 → 直接采用，从不看来源）；
      · 打标 / 训练里的原生库接连 0xC0000005（conda 的 native DLL 污染）；
      · 他按别的建议「屏蔽 Anaconda」→ 训练环境立刻报「损坏、找不到 python 路径」
        （因为 venv 的基座就是 conda）；
      · 最后装官方 Python + 重下训练内核才好 —— 正是工具日志里早就写着的那条路。
    """
    from pathlib import Path
    import Kohya一键工具 as core

    # ① 来源识别
    for p in (r"C:\Software\anaconda3\python.EXE", r"D:\miniconda3\python.exe",
              r"C:\Users\a\Miniforge3\python.exe", r"C:\ProgramData\Anaconda3\python.exe"):
        assert core._python_is_conda(p) is True, "没认出 conda 来源：%s" % p
    for p in (r"C:\Python312\python.exe",
              r"C:\Users\a\AppData\Local\Programs\Python\Python312\python.exe"):
        assert core._python_is_conda(p) is False, "误判为 conda：%s" % p
    assert core._python_is_conda(None) is False
    assert core._python_is_conda("") is False

    # ② 建环境时不能「默默采用」conda 的解释器
    _src = Path(core.__file__).read_text(encoding="utf-8-sig")
    _k = _src.index("def install_python(")
    _body = _src[_k:_src.index("\ndef ", _k + 10)]
    assert "_python_is_conda(py)" in _body, "install_python 未校验 Python 来源"
    assert "and not _conda" in _body, "install_python 仍会直接采用 conda 的解释器"
    assert "Anaconda" in _body, "缺少对 Anaconda 的说明文案"

    # ③ 徽章必须显示**实际**版本，不能写死 3.12
    st = core.system_status(force=True)
    assert "python_conda" in st, "system_status 未暴露 python 来源"
    assert "python_path" in st, "system_status 未暴露 python 路径"
    _g = Path(os.path.join(os.path.dirname(core.__file__), "kohya_gui.py")).read_text(encoding="utf-8-sig")
    assert '"● Python 3.12"' not in _g, \
        "徽章又写死成 Python 3.12 了（用户会误以为自己环境符合要求）"
    assert "python_conda" in _g and "Anaconda ⚠" in _g, "徽章未如实显示 conda 来源"
    print("PYTHON_ENV_SOURCE_GUARD_OK")


def test_native_crash_diagnosis():
    """训练侧必须能识别 native 崩溃（0xC0000005），并给出可执行的排查步骤。

    2026-09-16 qiansui 用户的训练失败：`anima_train_network.py` 原生崩溃、**零输出**，
    accelerate 把它转成退出码 1 → 工具只报「训练结束，退出码 1，请查看上方日志」。
    而 `3221225477` 在全代码库 **0 命中** —— 工具根本不认识这个错误码。
    """
    from pathlib import Path
    import Kohya一键工具 as core

    # ① 真实的失败文本（accelerate traceback 片段）必须命中
    tail = ["The following values were not passed to `accelerate launch` and had defaults used instead:",
            "\t`--mixed_precision` was set to a value of 'no'",
            "Traceback (most recent call last):",
            "subprocess.CalledProcessError: Command '[...anima_train_network.py...]'"
            " returned non-zero exit status 3221225477."]
    out = []
    assert core._diagnose_native_crash("\n".join(tail), logf=out.append) is True, "未识别 native 崩溃"
    blob = "\n".join(out)
    for kw in ("0xC0000005", "Anaconda", "vc_redist", "重复训练不会变好"):
        assert kw in blob, "诊断缺少关键信息 %s：%s" % (kw, blob)
    assert core._diagnose_native_crash("exit code 0xC0000005", logf=lambda s: None) is True

    # ② 无关日志不得误报（OOM/普通报错不是 native 崩溃，修法完全不同）
    for junk in ("[训练] steps: 5%| 51/1024 [01:00<19:00, 2.10s/it]",
                 "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB",
                 "ImportError: No module named 'cv2'", ""):
        assert core._diagnose_native_crash(junk, logf=lambda s: None) is False, \
            "无关日志被误判为 native 崩溃：%r" % junk

    # ③ 必须接在所有训练失败点的公共入口上（否则六条路径里只有一条有诊断）
    _src = Path(core.__file__).read_text(encoding="utf-8-sig")
    _k = _src.index("def _diagnose_optimizer_failure(")
    assert "_diagnose_native_crash(log_text, logf)" in _src[_k:_k + 900], \
        "native 崩溃诊断没接在 _diagnose_optimizer_failure 上"

    # ④ 换源重试不能用「失败」二字（用户会以为整体失败了），且结束要有结论
    assert "当前镜像下载失败" not in _src, "换源提示仍写「失败」（会误导用户以为整体失败）"
    assert "[Anima] ✓ 文本编码器已就绪" in _src, "Anima 下载完没有明确结论"
    print("NATIVE_CRASH_DIAGNOSIS_OK")


def test_high_coverage_tag_lock():
    """标签统计：高覆盖率特征要能提醒 + 手动锁进固定前缀。

    2026-09-16 用户实测：他的角色特征 `green hair` 是 20/21（95%）、`witch hat` 是
    17/21（81%），全都够不着强绑定的 **100%** 门槛 → 自动锁定永远拿不到它们 →
    单写触发词唤不出角色（他自己手动补上 `green hair` 就"非常像"）。

    所以标签统计里要有两样：**覆盖率提醒** + 一个把选中标签**锁进固定前缀**的入口。
    """
    from pathlib import Path
    import Kohya一键工具 as core

    _base = os.path.dirname(core.__file__)
    g = Path(os.path.join(_base, "kohya_gui.py")).read_text(encoding="utf-8-sig")
    # ① 覆盖率：必须真的去数图片总数
    assert "core.count_images(self.train_dir)" in g, "统计窗未取图片总数（算不出覆盖率）"
    assert "未锁定" in g, "统计窗未标出「高覆盖但未锁定」的标签"
    assert "def _stats_lock_to_prefix" in g, "缺手动锁定入口"
    assert "_btn_lock" in g, "锁定按钮未挂到窗口上（不便验证与扩展）"
    # ② 必须说清代价：锁了以后固定出现 + 要重跑预处理才生效
    _k = g.index("def _stats_lock_to_prefix")
    _body = g[_k:_k + 2600]
    assert "固定出现" in _body, "确认框没说「锁进去的特征会固定出现」（锁了帽子就脱不掉）"
    assert "重跑" in _body, "确认框没说要重跑预处理才生效"
    assert "_push_undo" in _body, "手动锁定没有留撤销快照（写坏了没法还原）"

    # ③ 核心函数：走 ||| 手动固定区（复用已有且已有测试覆盖的机制）
    _c = Path(core.__file__).read_text(encoding="utf-8-sig")
    assert "def lock_tags_to_prefix(" in _c, "缺核心锁定函数"
    assert "_FIXED_SEP" in _c, "未使用 ||| 手动固定区分隔符"

    # ④ 强绑定没锁到特征时不能静默（否则用户只能猜"是不是不生效"）
    _p = Path(os.path.join(_base, "preprocess.py")).read_text(encoding="utf-8")
    assert "没有可锁定的特征" in _p, "强绑定没锁到特征时又静默了（用户无从查证）"
    assert "★ 锁进固定前缀" in _p, "强绑定的提示没有指向界面入口"
    # ⑤ 覆盖率不足的告警不能说成"数据集不一致" —— 真实原因常是**视角遮挡让自动打标漏标**
    #    （2026-09-16 用户实测：21 张里有背面/侧身图，双马尾 twintails 只有 20/21，
    #     而它确实是角色的固定特征。旧文案"人物一致性不足，建议统一训练集特征"
    #     会把人引去改数据集甚至删掉侧身图，而且根本修不了"打标器看不到"这件事。）
    #    注：这里用**正向断言** —— 代码注释里会引用旧文案做历史说明，负向断言会误报。
    assert "视角遮挡" in _p and "打标漏标" in _p, "缺「视角遮挡导致自动打标漏标」的说明"
    assert "打开「标签统计」" in _p, "告警没有给出可执行的下一步"
    # 统计窗文案用的是「侧身 / 背面图容易让自动打标漏标」——断言跟着实际用词走
    assert "漏标" in g and "侧身" in g, "统计窗说明未解释「侧身/背面图会被漏标」"
    print("HIGH_COVERAGE_TAG_LOCK_OK")


def test_anima_component_picker():
    """Anima 的文本编码器 / VAE 要能「指定已有文件」，且查找优先用它。

    2026-09-16 用户反馈：他本机已经有 Anima 的模型，但工具只在 3 个固定 APPDATA 目录里
    按**精确目录名**找（Qwen3-0.6B / Anima_vae），找不到就直接下载
    （Qwen3 1.2GB + VAE 0.3GB，国内约 1.3MB/s ≈ 20 分钟）。
    日志实证确实走了下载（`[Anima] 从魔搭下载 Qwen3-0.6B/…`）——
    文件他早就有了，白等一场。
    """
    import tempfile
    from pathlib import Path
    import Kohya一键工具 as core

    td = tempfile.mkdtemp(prefix="kk_anima_t_")
    q3 = os.path.join(td, "Qwen3-0.6B")
    os.makedirs(q3, exist_ok=True)
    open(os.path.join(q3, "config.json"), "w", encoding="utf-8").write("{}")
    open(os.path.join(q3, "model.safetensors"), "wb").write(b"\x00" * 16)
    bad = os.path.join(td, "model.safetensors (1).safetensors")
    open(bad, "wb").write(b"\x00" * 16)
    nodir = os.path.join(td, "empty")
    os.makedirs(nodir, exist_ok=True)
    vae = os.path.join(td, "qwen_image_vae.pth")
    open(vae, "wb").write(b"\x00" * 16)

    # ① 校验必须前置（选的时候就拦住，而不是等训练时炸）
    assert core._anima_component_ok("qwen3", q3)[0] is True
    ok, why = core._anima_component_ok("qwen3", bad)
    assert ok is False and "标准名" in why, why
    assert core._anima_component_ok("qwen3", nodir)[0] is False
    assert core._anima_component_ok("vae", td)[0] is False        # VAE 必须是文件
    assert core._anima_component_ok("vae", vae)[0] is True       # .pth 合法
    assert core._anima_component_ok("vae", os.path.join(td, "没有这个文件"))[0] is False

    # ② 查找优先级：手动指定 > 目录扫描（这是整件事的关键，写反了等于没做）
    _src = Path(core.__file__).read_text(encoding="utf-8-sig")
    _k = _src.index("def _anima_find_qwen3_any(")
    assert '_manual = anima_get_component("qwen3")' in _src[_k:_k + 700], \
        "Qwen3 查找未优先用手动指定的路径"
    _v = _src.index("vae_dir = os.path.join(base, \"Anima_vae\")")
    assert 'vae_file = anima_get_component("vae")' in _src[_v:_v + 500], \
        "VAE 查找未优先用手动指定的路径"

    # ③ 顺带修的：VAE 完整性校验必须限定 .safetensors（否则合法 .pth 被误报"损坏"）
    assert 'if vae_file.lower().endswith(".safetensors") and not _safetensors_complete(vae_file):' in _src, \
        "VAE 完整性校验未限定 .safetensors（.pth 会被误报成损坏）"

    # ④ 界面：Anima 分支要开真对话框（messagebox 放不下按钮），且带选择入口
    g = Path(os.path.join(os.path.dirname(core.__file__), "kohya_gui.py")).read_text(encoding="utf-8-sig")
    assert 'if bt == "anima":' in g and "self._show_anima_components()" in g, \
        "Anima 指引未改走组件对话框"
    # 注意：不能写 "def _show_anima_components(self)"（带右括号）——
    # 该方法现在签名是 (self, wait=False)，带括号的串根本不存在。
    assert "def _show_anima_components(self" in g, "缺少 Anima 组件对话框"
    _d = g[g.index("def _show_anima_components(self"):]
    _d = _d[:_d.index("\n    def ", 10)]
    assert "core.anima_set_component" in _d, "对话框未写回指定的路径"
    assert "选文件夹" in _d and "选文件" in _d, "对话框缺少选择入口"
    assert "_refresh_anima" in _d, "对话框没有状态刷新（指定后看不到变化）"

    # ⑤ 入口必须够得着：原先只挂在「没有模型？点这里下载」里，
    #    已有底模的用户**根本不会点那里** ✗ —— 所以要在训练前拦一道。
    assert "def _anima_components_preflight(self" in g, "缺少训练前预检"
    _p = g[g.index("def _anima_components_preflight(self"):]
    _p = _p[:_p.index("\n    def ", 10)]
    assert "if _q and _v:" in _p and "_modal(" in _p, \
        "预检未做到「两个都齐时静默通过、否则先问一句」"
    # 训练必须先等用户选完再开跑（少这一句 = 用户还没选，训练已经开始下载了）
    assert "_show_anima_components(wait=True)" in _p, "预检未阻塞等待用户选择"
    assert "w.wait_window()" in g, "对话框不支持阻塞等待"
    assert g.count("self._anima_components_preflight(params)") >= 2, \
        "预检只接了一处（一键开始训练 / 开始训练 都要有）"
    print("ANIMA_COMPONENT_PICKER_OK")


def test_label_undo_stack():
    """撤销必须能**连退多步**。

    2026-09-16 用户反馈：「这个撤销只能按一次啊，锁定两个之后第一个就改不了了」——
    原实现是**单槽快照**（self._undo = None）：第二次批量操作直接把第一次的快照覆盖掉，
    于是只能退一步，用户连锁两个特征后第一个再也回不去 ✗

    栈语义：每项存的是「**该次操作前**这些 .txt 的原文」，从栈顶往回退即可逐步还原。
    这里既查实现（必须是栈 + 退完一步还留着按钮可用），也在核心层跑一遍逆序还原，
    确认「连撤两步 == 回到原文」这个真正被用户感知的性质成立。
    """
    import tempfile
    import shutil
    from pathlib import Path
    import Kohya一键工具 as core
    from kohya_core import paths as _P

    g = Path(os.path.join(os.path.dirname(core.__file__), "kohya_gui.py")).read_text(encoding="utf-8-sig")
    assert "self._undo = []" in g, "撤销仍是单槽（第二次操作会覆盖第一次）"
    assert "_UNDO_MAX" in g, "撤销栈没有上限（连续大批量操作会一直堆积）"
    _p = g[g.index("def _push_undo(self"):]
    _p = _p[:_p.index("\n    def ", 10)]
    assert "self._undo.append(" in _p, "新快照没有入栈"
    assert "del self._undo[0]" in _p, "超上限时没有丢最旧的"
    _u = g[g.index("def _do_undo(self"):]
    _u = _u[:_u.index("\n    def ", 10)]
    assert "self._undo[-1]" in _u and "self._undo.pop()" in _u, "撤销没按栈顶弹出"
    assert 'state=("normal" if self._undo else "disabled")' in _u, \
        "撤销后一律禁用按钮（退一步就点不动了 —— 正是用户报的现象）"

    # 行为：两次操作各留快照，按逆序还原必须回到原文
    tmp = tempfile.mkdtemp()
    _real = _P.data_dir
    _P.data_dir = lambda: tmp
    try:
        ds = os.path.join(tmp, "dataset", "proj", "train_character")
        os.makedirs(ds)
        orig = {"a": "1girl, solo, blue_hair\n", "b": "1girl, solo, blue_hair\n"}
        for s, t in orig.items():
            open(os.path.join(ds, s + ".txt"), "w", encoding="utf-8", newline="").write(t)
            open(os.path.join(ds, s + ".png"), "wb").write(b"x")   # 列表以图片为驱动
        s1 = {}
        core.batch_remove_tags(ds, "solo", snapshot=s1)            # 第 1 步
        s2 = {}
        core.batch_remove_tags(ds, "blue_hair", snapshot=s2)       # 第 2 步
        _p1 = os.path.join(ds, "a.txt")
        assert open(_p1, encoding="utf-8").read().strip() == "1girl"
        core.restore_captions(s2)                                  # 撤销第 2 步
        assert open(_p1, encoding="utf-8").read().strip() == "1girl, blue_hair"
        core.restore_captions(s1)                                  # 撤销第 1 步
        got = open(_p1, encoding="utf-8").read()
        assert got == orig["a"], repr(got)
    finally:
        _P.data_dir = _real
        shutil.rmtree(tmp, ignore_errors=True)
    print("LABEL_UNDO_STACK_OK")


def test_krea2_warmup_notice():
    """Krea2(Fizgig) 必须提前说明「前 2 个 epoch 是预热期」。

    2026-09-17 多用户反馈「Krea2 训练速度降低」（4090：一秒多/步 → 3 秒；5070 Ti：5s → 8.8s）。
    复盘两份用户日志后确认的机制：**引擎自己就打印了预热说明** ——
      INFO:fizgig.krea2.trainer:[warm-up] Warm-up phase — the first two epochs start slowly
      while the GPU plans kernels and fills its caches.
    实测步速在预热期是 7.38 → 7.74 → 7.87 → 7.95 → 8.00 → 8.06 **逐步爬升**；
    而工具此前只写「5s/it 左右步速正常」✗ → 用户在预热期看到 7~8s/it，必然误判成「变慢了」✗；
    且日志显示用户两次都在 33 步（第一个存档点之前）就停 → **重开又回到预热** ✗，
    怎么试看到的都是慢的那一段 ✓ 必须提前说清楚。
    """
    from pathlib import Path
    import Kohya一键工具 as core      # noqa: E402
    src = Path(core.__file__).read_text(encoding="utf-8-sig")
    assert "前 2 个 epoch" in src and "预热阶段" in src, "缺少 Krea2 预热期说明"
    _i = src.index("前 2 个 epoch 是**预热阶段**")
    assert "warm-up" in src[max(0, _i - 700):_i + 700].lower(), \
        "预热说明没有引用引擎原文 —— 属于凭猜，不是证据"
    assert "重新预热" in src, "没说明「中途停止重开会重新预热」（用户就是栽在这里）"
    # ★ 不得写"未经证实的应然速度"（例如「稳态 5s/it 正常」）——
    # 工具里原有的「5s/it 左右步速正常」就是这么来的 ✗，而它**当初怎么测出来的已不可考** ✗，
    # 结果用户拿它当标尺、看到 7~8s 就以为坏了 ✗ 这种数字没验证过就不该写进日志。
    _seg = src[src.index("[Krea2(Fizgig)] ⚠ 前 2 个 epoch"):]
    _seg = _seg[:_seg.index('")')]
    assert "s/it" not in _seg, "预热提示里写了具体 s/it 数字 —— 未经验证的应然速度会误导用户"
    # 内存提示不能再说「已自动降低块交换数」：该函数体里**只有这条日志、没有任何调整动作** ✗
    # 注意只看**函数体**：注释里为了说明来龙去脉仍会引用这句旧文案，
    # 直接对全文断言会误判（本轮就又踩了一次「断言太字面」✗）。
    _w = src.index("def _warn_low_ram(")
    _body = src[_w:_w + src[_w:].index("\n\n\n")]
    # 只看**代码行**（去掉注释行）：注释里会引用旧文案来说明来龙去脉 ✓
    _code = "\n".join(_l for _l in _body.splitlines() if not _l.strip().startswith("#"))
    assert "已自动降低块交换数" not in _code, \
        "内存提示又在说假动作（并未真的调整参数）—— 会让人把变慢归因到没发生的事"
    assert "请把「块交换(blocks_to_swap)」调小" in _body, "内存提示没给出可执行建议"
    print("KREA2_WARMUP_NOTICE_OK")


def test_krea2_auto_quant_is_int8():
    """Krea2 的 auto 档必须走 int8 —— fp8 在 K2 上是**灾难档** ✗。

    2026-09-17 用户汇总实测（512px）：
        · 4090 24G：fp8 7 s/步 → int8 **1 s/步**（7×）
        · 16G 卡  ：fp8 50~100 s/步 → int8 **2.2 s/步**（25~45×）
    根因：K2 的 fp8 路径**没用上 scaled_mm**（per-channel 量化与它不兼容，强开会 raise，
    见 `_patch_musubi_fp8_scaled_mm`），每次前向要反量化回 bf16；块交换越多越惨。
    官方数据同向：3090 上 fp8 7.1 vs convrot_int8 5.3；Blackwell 上 bf16 2.0 快过 fp8 2.3。
    """
    import Kohya一键工具 as core      # noqa: E402

    # ① 两个引擎的 auto 档，只要不是低显存，都必须走 int8
    for _v in (12, 16, 24, 47.48):
        _q, _d = core._resolve_quant_mode(None, lambda s: None, _v, requested="auto")
        assert _q == "int8", "%.0fG 的 auto 档没走 int8（得到 %s）" % (_v, _q)
        _f, _s, _dd = core._fizgig_quant_swap(_v, "auto")
        assert _f == ["--quant_int8", "bf16"], \
            "Fizgig %.0fG 的 auto 档没走 int8（得到 %s）" % (_v, _f)
        # ⚠️ swap 必须**沿用该档位原值**，不能顺手改 0：16G 档靠块交换才跑得起来，
        # 改成 0 会 OOM。这里就是最初写错、被 engine 套件拦下的地方 ✓
        _expect_swap = 0 if _v >= 32 else (12 if _v >= 24 else (20 if _v >= 16 else 26))
        assert _s == _expect_swap, \
            "%.0fG 的 swap 应保持档位原值 %s，得到 %s" % (_v, _expect_swap, _s)

    # ② 低显存仍以显存优先：<10G 走 NF4（不能因为提速把兜底弄丢）
    assert core._fizgig_quant_swap(8, "auto")[0] == ["--quantize_4bit"], "8G 档的 NF4 兜底被破坏"

    # ③ 显式选择必须仍被尊重 —— 老项目存档里可能就存着 fp8，不能静默改掉
    assert core._fizgig_quant_swap(24, "fp8")[0] == [], "显式 fp8 被静默改成了别的档"
    assert core._resolve_quant_mode(None, lambda s: None, 24, requested="fp8")[0] == "fp8", \
        "musubi 的显式 fp8 被静默改掉"
    assert core._resolve_quant_mode(None, lambda s: None, 24, requested="int8")[0] == "int8"

    # ④ 显式 fp8 必须带实测代价 —— 否则老用户不知道自己还踩在慢档上
    assert "慢" in core._fizgig_quant_swap(24, "fp8")[2], "Fizgig 显式 fp8 没给出实测代价"
    assert "慢" in core._resolve_quant_mode(None, lambda s: None, 24, requested="fp8")[1], \
        "musubi 显式 fp8 没给出实测代价"

    # ⑤ 底模已预量化时不做工具侧量化（原有行为不能丢）
    assert core._resolve_quant_mode(None, lambda s: None, 24, requested="auto",
                                    prequantized=True)[0] == "none", "预量化底模被重复量化"
    print("KREA2_AUTO_QUANT_OK")


def test_wd14_model_selectable():
    """WD14 打标模型必须「可选 + 缺失时静默回默认」—— 这就是老用户无缝、新用户无感的关键。

    2026-09-17 变更：模型不再内置（内含 311MB onnx，发布包 562MB / 安装包 488MB，
    分发吃力）→ 改为首次使用时下载（魔搭优先 → hf-mirror 兜底）。
    社区主流是 swinv2-v3（月下载约 70 万，是旧版 moat-v2 的数千倍，标签库更新到 2024-02）。
    """
    import preprocess as P      # noqa: E402

    # ① 缺失 / 空 / 垃圾值 → 一律静默落默认，**绝不抛异常**
    #    （老用户升级后项目里没有这个键，走的就是这条路 ✓）
    for _bad in (None, "", "   ", "不存在的模型", "Auto"):
        _k, _r = P.resolve_wd14_model(_bad)
        assert _k == P.WD14_DEFAULT_MODEL, "%r 没落到默认模型（得到 %s）" % (_bad, _k)
        assert _r == P.WD14_MODELS[P.WD14_DEFAULT_MODEL], "默认模型 repo 不对"
    # ② 用户显式选择必须被尊重（含切回旧模型）
    assert P.resolve_wd14_model("moat-v2")[0] == "moat-v2", "显式选旧模型被改掉"
    assert P.resolve_wd14_model("swinv2-v3")[0] == "swinv2-v3"
    # ③ 默认必须是社区主流的 swinv2-v3
    assert P.WD14_DEFAULT_MODEL == "swinv2-v3", "默认模型被改动"
    # ④ 两个模型的目录名互不相同 → 天然共存，互不覆盖
    _dirs = {P._wd14_repo_dirname(r) for r in P.WD14_MODELS.values()}
    assert len(_dirs) == len(P.WD14_MODELS), "两个模型的目录名冲突，会互相覆盖"
    # ⑤ 下载源策略：魔搭优先（hf 作兜底）—— 魔搭 URL 必须指向本项目仓库
    assert P.WD14_MS_REPO in P.WD14_MS_BASE, "魔搭下载源没指向本项目仓库"
    assert "wd14_models" in P.WD14_MS_BASE, "魔搭路径与上传位置不一致"
    # ⑥ 命令行参数必须存在且**不用 choices**（未知值要静默回默认，而不是 argparse 报错退出）
    _src = open(os.path.join(ROOT, "preprocess.py"), encoding="utf-8").read()
    assert '"--wd14-model"' in _src, "缺少 --wd14-model 参数"
    _line = [l for l in _src.splitlines() if '"--wd14-model"' in l][0]
    assert "choices=" not in _line, "用了 choices → 未知值会让 argparse 直接报错退出（破坏无缝）"
    print("WD14_MODEL_SELECTABLE_OK")


def test_close_confirm_while_running():
    """训练/安装进行中关闭窗口必须**先确认**——误点关闭会直接终止任务、白跑几小时。

    2026-09-17 用户反馈：「软件没有关闭提醒，如果在训练不小心误关，会直接停掉」。
    以前 `_on_close` 是「保存配置 → 直接 destroy()」✗，训练在跑也照关不误。

    这里**真跑行为**（用假 self 直接调 `_on_close`），而不是查源码里有没有某句文案 ✗。
    """
    import kohya_gui as G      # noqa: E402
    import Kohya一键工具 as core      # noqa: E402
    from tkinter import messagebox as MB

    class _Root:
        def __init__(self):
            self.destroyed = False

        def destroy(self):
            self.destroyed = True

    def _make(busy):
        class _Fake:
            def __init__(self):
                self.busy = busy
                self._task_title = "一键开始训练" if busy else ""
                self.current_project = None
                self.ui_proc = None
                self.root = _Root()

            def _autosave(self):
                pass

            def _task_running(self):
                # 直接用真实实现的语义（busy 或底层有活跃子进程），但底层探测在本测试里
                # 不依赖真实进程 —— 单测不该去问系统
                return bool(self.busy)

        return _Fake()

    _real_ask = MB.askyesno
    _real_stop = core.stop_active_process
    _stops = []
    try:
        # ⚠️ 必须把 stop_active_process 打桩：它会 set 全局 _STOP_EVENT，
        # 真跑一次会把**同一个测试进程里后续的 run_stream 全部带停** ✗
        core.stop_active_process = lambda: _stops.append(1)
        # ① 有任务 + 用户选「否（继续跑）」→ **窗口不能关**，也不能停任务
        _f = _make(True)
        MB.askyesno = lambda *a, **k: False
        G.App._on_close(_f)
        assert _f.root.destroyed is False, "选了「不关」，窗口却关了"
        assert not _stops, "选了「不关」，任务却被停了"
        # ② 有任务 + 用户选「是（仍要关）」→ 关窗 **且主动停任务**
        _f2 = _make(True)
        MB.askyesno = lambda *a, **k: True
        G.App._on_close(_f2)
        assert _f2.root.destroyed is True, "确认关闭后窗口没关"
        assert _stops, "确认关闭时没有主动停止任务（只 destroy 会让子进程管道断裂）"
        # ③ 没有任务 → **不打扰**，直接关（不能给每次正常退出都弹框）
        _called = {"n": 0}

        def _count(*a, **k):
            _called["n"] += 1
            return True

        _f3 = _make(False)
        MB.askyesno = _count
        G.App._on_close(_f3)
        assert _f3.root.destroyed is True, "空闲时关不掉"
        assert _called["n"] == 0, "空闲退出也弹了确认框（打扰）"
    finally:
        MB.askyesno = _real_ask
        core.stop_active_process = _real_stop
        try:
            core.reset_stop()          # 保险：清掉可能被置位的停止事件
        except Exception:
            pass

    # ④ 源码层确认：确认框必须**默认选中「否」**（误点的代价是继续跑，不是白跑）
    _gsrc = open(os.path.join(ROOT, "kohya_gui.py"), encoding="utf-8").read()
    _i = _gsrc.index("def _on_close(self)")
    _seg = _gsrc[_i:_i + 2000]
    assert "default=" in _seg and "no" in _seg.lower(), "关闭确认框没设默认值为「否」"
    print("CLOSE_CONFIRM_WHILE_RUNNING_OK")


def test_wd14_selector_visible():
    """打标模型的选择必须**在折叠区之外**——用户要能直接看到，而不是去翻高级参数。

    2026-09-17 教训：我第一版把它放进了「高级参数」折叠区（`adv_collapsed = True` 默认收起），
    用户反馈「打标模型选择组件在哪里？我没有看到啊」✗ —— 位置选错了：
    它是常规选择，不是"老手参数" ✓

    这里**真构造界面**并断言控件确实 mapped（折叠区里的子控件 `winfo_ismapped()` 为 False，
    所以这条断言正好能抓住"又被塞进折叠区"的回归 ✓）。
    """
    import kohya_gui as G      # noqa: E402

    _app = G.App()
    try:
        _app._build_main_cards()
        _app.root.update_idletasks()
        _m = getattr(_app, "wd14_model_menu", None)
        assert _m is not None, "打标模型下拉不存在"
        # ⚠️ 判据不能用 winfo_ismapped()：_build_main_cards() 之后界面可能还停在主页，
        # 祖先不可见 → 所有子控件都是 not mapped，那会误报（第一次就踩了 ✗）。
        # 真正要保证的是语义：**它不在折叠容器 adv_body 里**（那才是"用户看不到"的原因）。
        _adv = getattr(_app, "adv_body", None)
        _p, _in_adv = _m, False
        while _p is not None:
            if _p is _adv:
                _in_adv = True
                break
            _p = getattr(_p, "master", None)
        assert not _in_adv, "打标模型控件在「高级参数」折叠区里 —— 用户翻不到 ✗"
        # 也不该藏在"高级参数"卡片里（哪怕折叠区之外）—— 它属于「① 准备图片数据」
        _p2, _in_card3 = _m, False
        while _p2 is not None:
            if _p2 is getattr(_app, "btn_toggle_adv", None):
                _in_card3 = True
            _p2 = getattr(_p2, "master", None)
        assert not _in_card3, "打标模型还在「高级参数」卡片里，应放在①准备图片数据 ✓"
        assert _app.wd14_model_var.get().startswith("swinv2-v3"), \
            "默认值不是新模型（%s）" % _app.wd14_model_var.get()
    finally:
        try:
            _app.root.destroy()
        except Exception:
            pass
    print("WD14_SELECTOR_VISIBLE_OK")


def main():
    print("== Kohya-LoRA 工具 · 冒烟测试 ==")
    check("语法检查", test_syntax)
    check("导入 + 配置完整性（全部模式）", test_config_completeness)
    check("AI 图像模型配置", test_at_image_models)
    check("yaml 生成可解析", test_yaml)
    check("下载模型配置（FLUX/Anima/Krea2）", test_download_models)
    check("标签管理 v1 · 离线词典核心链路", test_tagging)
    check("Anima 合并包识别/剥离/缓存", test_anima_ckpt)
    check("采样预览不污染监控 + Fizgig 断点查找", test_monitor_sampling_and_fizgig_resume)
    check("训练完成按项目名导出成品", test_lora_naming)
    check("删项目清理图集数据 + 遗留数据清理", test_project_data_cleanup)
    check("标签批量操作：预演 + 快照撤销", test_label_editor_safety)
    check("主页输出目录入口", test_home_output_button)
    check("预处理进度（WD14 打标）", test_preprocess_progress)
    check("Anima 组件指定已有文件", test_anima_component_picker)
    check("Krea2 预热期说明（防「更新后变慢」误判）", test_krea2_warmup_notice)
    check("Krea2 auto 量化必须是 int8（fp8 是灾难档）", test_krea2_auto_quant_is_int8)
    check("WD14 打标模型可选 + 缺失静默回默认", test_wd14_model_selectable)
    check("任务进行中关闭窗口必须先确认", test_close_confirm_while_running)
    check("打标模型选择必须可见（不在折叠区）", test_wd14_selector_visible)
    check("标签撤销可连退多步", test_label_undo_stack)
    check("Python 环境来源校验 + 徽章如实显示", test_python_env_source_guard)
    check("训练 native 崩溃诊断", test_native_crash_diagnosis)
    check("高覆盖特征锁进固定前缀", test_high_coverage_tag_lock)
    print("-" * 40)
    if FAILED:
        print("✘ 失败 %d 项: %s" % (len(FAILED), "、".join(FAILED)))
        return 1
    print("✔ 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
