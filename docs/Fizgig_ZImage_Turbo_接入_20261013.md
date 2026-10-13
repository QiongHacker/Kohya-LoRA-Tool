# Fizgig v7.3.1 / Z-Image Turbo 接入记录

日期：2026-10-13；训练器版本：v0.19.10。

## 上游核对

- Fizgig 官方 v7.3.1 发布于 2026-10-12 23:19:27 UTC（北京时间 2026-10-13 07:19:27），固定源码提交 `7ee4986dae28fe35cc4ef636b984729624b32b42`。
- 来源：<https://github.com/shootthesound/Fizgig/releases/tag/v7.3.1>；核对该提交下 `src/fizgig/families/zimage.py`、`zimage/driver.py`、`zimage/model.py`、`zimage/pipeline.py` 与 requirements。
- 上游家族入口为 **Z-Image Turbo**，标记实验性。它不是 Z-Image Base，不能把同架构当作同一权重版本。
- 普通 LoRA 在 Turbo 本体上训练；官方训练适配器在训练时固定使用，采样时关闭，不写入导出的 LoRA。
- 上游提供 BF16 / INT8 / NF4，以及块交换。工具保留显式精度选择，默认由引擎规划。此版本接入普通 LoRA；未新增该模型的滑块、LoKr 或全参数微调入口。

## 固定资源

| 资源 | 固定身份 | 检查 |
| --- | --- | --- |
| Fizgig 源码 ZIP | 12,666,311 bytes；SHA256 `ed0dfe599006492d5dfb04ddbc66aae0a4237b257d05eab0424aa4280de1b3bc` | 官方下载与魔搭完整下载一致 |
| Turbo tokenizer / VAE config | `Tongyi-MAI/Z-Image-Turbo`，提交 `f332072aa78be7aecdf3ee76d5c247082da564a6` | 5 个文件，共 15,881,877 bytes；逐项大小及 SHA256 见 `kohya_core/fizgig_helpers.json` |
| Fizgig Turbo training adapter | `ShootTheSound/Fizgig-Z-Image-Turbo-Training-Adapter`，提交 `22a6a059ee04deeebc8d0d4c6d2c8d6e74c943bf` | 70,117,384 bytes；SHA256 `2da9920c1b27e07cbf4f9959a7d35af335c12ed73fa615b8b88beb2b64e835bf`；30 层 LoRA A/B 结构 |
| Turbo 底模 / Qwen3-4B / FLUX.1 VAE | `Comfy-Org/z_image_turbo` 的 `split_files` | 魔搭地址响应检查；未完整下载这三份大权重，也未宣称其哈希经过本轮验证 |

配套小文件与适配器已放入工具的魔搭仓库。工具下载后校验固定资源；大权重在选取或下载完成后核对 safetensors 长度、数据类型和模型结构。结构检查可以判别架构，不能可靠区分结构相同的 Base / Turbo 或所有社区衍生权重。

## 软件入口

新建项目 → Z-Image → **Z-Image Turbo** → **Fizgig v7.3.1**。底模、Qwen3-4B、FLUX.1 VAE、训练适配器均可选择已有本地文件，也可按需下载；不要求复制或改名。原始路径随项目保存。训练适配器使用当前固定官方版本。

默认 rank / alpha 16，固定学习率 1e-4，30 轮，768px，采样每 5 轮。建议 12GB 及以上显存；首次用 512px 小规模试训，采样 8 步、CFG 1。文本编码器只用于缓存后卸载。额外负向提示词不参与该采样路径，界面隐藏该选项。

## 旧项目与 AMD 修复

- Base 的旧模式键 `zimage` 与 AI Toolkit 入口保留。新模式键为 `zimage_fz`，明确表示 Turbo / Fizgig。
- 新项目默认 v7.3.1；旧 v7.0.1 / v6.5.0 项目保留所存版本和旧环境。新版源码按版本隔离，仅当依赖完全一致时复用已有 Python 环境；修复和安装依赖使用新版本独立环境。
- 缓存身份包含实际源码提交；续训快照包含实际版本、底模、rank、优化器等。不同版本的快照不会混用。滑块导出与重新导入也保留原引擎版本。
- ROCm 补齐 torch / torchvision / SDK 声明的显卡组件，并核对版本和安装文件；安装命令结束后仍需完整验证才登记就绪。
- 通过目标 Python 的真实 HIP 枚举选择独显。显式可见设备设置继续生效；选卡失败阻止 GPU 自检。安装验证、状态与训练共用设备环境。
- 具体 GRE 日志证据和群友补救脚本范围见 `AMD_Fizgig_RX7900GRE_诊断_20261013.md`。

## 检查范围

已完成：修改 Python / JSON 的语法检查；固定上游源码部署适配、补丁锚点与清单检查；前端类型检查和生产构建；源码差异格式检查；公开配套资源下载与身份核对。源码与安装包隐私检查通过；已核对安装包中的页面清单、8 个冻结模块、模型目录和辅助资源。没有执行自动化测试套件。

本机为 NVIDIA。**未执行 AMD 依赖安装或实机计算，未执行完整 GPU 训练和生成质量验收。** 不能把这些安装与选卡问题的修复表述为所有 AMD 驱动、算子或显卡都已根治。

## v0.19.10 构建产物

- Setup.exe：227,198,688 bytes；SHA256 `216082ab3d6f4e564c33b96bac6de5c35247270c051bb0a66963e6dbbd30b9c9`。
- 便携 ZIP：240,471,786 bytes；SHA256 `33cd4e05df659833a225729dd7aa86453cce9de7d165d99cbdffe981bed37fae`。
- 发布使用现有 release.py，发布完成后回读魔搭更新信息及安装包完整下载哈希、GitHub Release 资产与两端 Git 提交。
