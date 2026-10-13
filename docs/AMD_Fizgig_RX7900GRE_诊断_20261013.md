# RX 7900 GRE：Fizgig 安装失败诊断

日期：2026-10-13。对应训练器 v0.19.7、Fizgig v7.0.1、Windows ROCm 7.15 固定环境。

## 日志证据

- 界面识别 RX 7900 GRE（gfx1100），系统同时有 AMD Radeon(TM) Graphics 核显。
- Fizgig 安装后的 GPU 卷积自检实际运行在核显上：`dev=AMD Radeon(TM) Graphics`、`cap=10.3`。
- 同一份日志中的 AI Toolkit 诊断枚举到两张 GPU，默认设备为核显 gfx1036。
- Fizgig 仅安装通用 torch / torchvision / ROCm 包，自检前输出“未发现按架构分包的包”。
- 失败为 `device kernel image is invalid`。该日志尚不能证明 GRE 上的驱动或计算故障。

## 源码处理

1. 使用目标引擎的 ROCm 实际枚举索引选择独显；不把 DXGI/WMI 索引作为 HIP 索引。
2. 显式显卡可见设置仍由用户控制，记录其实际首张可见设备。
3. 安装完整的 `torch[device-ARCH]`、`torchvision[device-ARCH]`、`rocm[libraries,devel,device-ARCH]`。
4. 根据包元数据核对架构组件版本和安装文件；损坏组件单独重新安装。
5. 安装验证、就绪状态和训练使用同一显卡环境；选卡失败时阻止计算自检。
6. 导出诊断包含尚未激活的新版本 Fizgig 环境，避免遗漏失败安装目录。

## 群友补救脚本

桌面文件：`修复_Fizgig_RX7900GRE.ps1`。

脚本针对该日志已有的 v7.0.1 环境。先退出训练器，再运行脚本。流程为基础版本核对、补齐组件、选择真实 GRE、FP16 卷积前向和反向检查、源码与 CLI 检查、登记引擎。仅通过全部检查后显示成功。

成功后在用户桌面生成 `启动训练器_RX7900GRE.cmd`。该启动器每次重新检测 GRE 的 HIP 编号，通过进程环境为旧版训练器选卡。修复脚本需保留在原位置。失败保留完整步骤输出和修复日志。

## 已完成的检查范围

- 修改源码和脚本内嵌 Python 的语法检查。
- Windows PowerShell 5.1 语法检查。
- 内嵌模块与源码内容、SHA-256 一致性检查。
- 本机固定版 Fizgig 源码清单和 CLI 导入路径检查。
- Git 差异格式检查。

本机为 NVIDIA，未执行 AMD 包安装、AMD 硬件计算或完整训练；这些结果等待实际用户运行后确认。修复合入训练器 v0.19.10，并加入固定 Fizgig v7.3.1 的 Z-Image Turbo 普通 LoRA 入口。发布构建与镜像回读记录见对应接入说明；AMD 实机结果仍需用户反馈。
