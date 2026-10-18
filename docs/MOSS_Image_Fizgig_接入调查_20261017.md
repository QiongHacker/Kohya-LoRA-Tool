# MOSS-Image 接入 Fizgig：前置调查与实施门槛

日期：2026-10-17

## 当前状态

- 已使用维护者账号 `l1934332574-maker` 在指定分类发布 Discussion #194，并通过公开页面回读核对标题、正文和分类： https://github.com/shootthesound/Fizgig/discussions/194
- 当前等待上游对范围和技术路线的反馈；尚未编写驱动或提交 PR。

- 已读取 Fizgig 驱动规范、静态图适配规范和交付检查表。
- 已核对上游 master `4847537845aa7fb6ca7961a0d0546cb2befcdcbb` 的完整文件树，未发现名含 moss 的路径；当前公开 Discussion 列表未发现 MOSS-Image 适配主题。这不排除未公开开发或其他分支的工作。
- 按官方要求，先发 Discussion 并约定范围，再编写 family 驱动；当前不修改本软件模型入口、不安装依赖、不下载大模型。
- 英文提案见同目录 `MOSS_Image_Fizgig_上游提案_20261017.md`。

## 已确认的技术约束

| 项目 | 官方证据与适配要点 |
| --- | --- |
| 生图主干 | Base transformer/config.json：42 层、dim=3072、text_dim=4096、latent_channels=128。是自定义架构，不能复用 Flux 模型名直接接入。 |
| 文本编码 | model_index.json 为 Qwen3_5Model；官方 ComfyUI 插件标识 qwen35_9b。pipeline 使用 chat template、最后隐藏层和 mask；官方用户指南要求冻结 Text Encoder。 |
| 潜空间 | VAE config 的 latent_channels=32、patch_size=[2,2]，DiT 输入为128通道；训练编码、打包与归一化需要逐项对照官方实现。 |
| 采样 | 自定义 MOSSImageScheduler，shift 后对齐训练网格；官方 pipeline 默认50步、CFG3、shift5、uncond_skip_layers=[22]。推理默认值不能当训练配方。 |
| 导出 | standalone 与 ComfyUI 的投影实现存在命名和融合布局差异，需要验证 fused/split LoRA 转换；不能只看训练正常退出。 |
| 训练配方 | 检查的公开指南提供冻结编码器和标注建议，尚未找到完整 LoRA loss/timestep/LR/Rank 配方；提请上游协助确认。 |
| 显存 | 5B仅指生图主干，9B编码器缓存阶段单独测量；不声明8GB可训、AMD兼容或默认量化方案。 |

## 阶段计划

1. 发起 New models & fine-tuning Discussion，询问是否重复开发、实现方式和训练配方，等待范围约定。
2. 范围约定后，独立开发隐藏的 moss_image family，首先对齐官方文本条件、单步预测、sigma和VAE。
3. 运行短程真实 LoRA 训练，确认采样随训练变化、导出键全部加载且效果一致、状态续训、离线加载及小显存规划。
4. 达到上游检查表后提交 PR；再接本软件的模型组件选择、配置保存、版本门槛、预览与续训入口。
5. 尚未验证的全量微调、滑块、布局训练、SFT兼容和AMD路径，分别记录，不能在界面宣称已支持。

## 来源与固定版本

- Fizgig drivers: https://github.com/shootthesound/Fizgig/blob/4847537845aa7fb6ca7961a0d0546cb2befcdcbb/docs/drivers/README.md
- Fizgig stills: https://github.com/shootthesound/Fizgig/blob/4847537845aa7fb6ca7961a0d0546cb2befcdcbb/docs/drivers/STILLS.md
- Fizgig checklist: https://github.com/shootthesound/Fizgig/blob/4847537845aa7fb6ca7961a0d0546cb2befcdcbb/docs/drivers/CHECKLIST.md
- MOSS code and guide: https://github.com/OpenMOSS/MOSS-Image/tree/c1355e6b1986dfe8fba3c0455783a05899212342
- Base configs, pipeline and model card: https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/tree/1ceede80badddd6a7812069619edaa5633587192
- Official ModelScope: https://modelscope.cn/models/openmoss/MOSS-Image-Base

## 验证边界

本轮仅完成文档、源码和配置阅读，以及讨论提交；没有运行模型、训练、GPU质量或显存测试。已有用户工作 `kohya_core/h3_dataset.py` 保持不动。
