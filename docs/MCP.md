# 外部 Agent · MCP

让自己的 Codex 或其他 MCP 客户端调用正在运行的 Kohya-LoRA 桌面软件。

## 接入

1. 在左侧打开 **外部 Agent · MCP**。
2. 勾选 **开启本机 MCP 接口**，保存设置。默认只读。
3. 需要自动配置、下载和训练时，再开启 **允许修改配置、下载和控制训练**。
4. 点击 **复制 Codex 配置**，粘贴到自己的 `~/.codex/config.toml`。已有同名 `mcp_servers.kohya_lora` 时替换该段，不重复添加。
5. 重新连接 MCP（客户端未提供重连时重启客户端），保持训练器运行。

其他支持 Streamable HTTP 的 MCP 客户端可使用 **复制通用配置**，按客户端要求填写 URL 和 Authorization 请求头。

默认地址为 `http://127.0.0.1:19366/mcp`。端口被占用时，在软件内改端口后重新复制配置。

连接配置含本机访问密钥，只保存在自己的客户端配置中，不要发布到仓库或群聊。密钥泄漏时点击 **重置连接密钥**，再更新客户端配置。

Codex HTTP 配置格式参考：[官方 MCP 文档](https://developers.openai.com/codex/mcp/)。软件复制的是带本机鉴权信息的完整片段，无需额外下载 MCP 桥接程序或另外运行一个训练器进程。

## 能做什么

- 读取软件版本、显卡、引擎安装状态、项目参数和缺少的模型。
- 创建项目、修改参数并保留原配置备份。
- 检查素材与标签完整性、运行预处理。
- 训练预检、启动训练、使用已有完整快照续训、查询进度或请求停止。
- 列出采样文件和训练历史，收集脱敏错误日志与实际生效参数。
- 调用软件现有的模型下载和环境/引擎安装检查。

不同模型的能力与界面一致。接口不会补齐某个引擎本来没有的续训、模型格式或精度支持。

## 推荐操作顺序

```text
get_environment
→ list_projects
→ get_project
→ update_project（需要修改时，带上最新 revision）
→ inspect_dataset
→ prepare_training
→ start_training
→ get_task（使用 task_id 和 next_offset 轮询）
→ awaiting_review 时检查 inspect_task_dataset
→ 用户已授权继续后调用 continue_training(confirmed=true)
→ get_task / list_samples / training_history
```

`start_training` 已包含素材预处理，无需先重复 `prepare_data`。普通训练预处理完成后会暂停等待检查；任务启动成功不代表训练引擎已经开始执行。滑块等专用流程以 `get_task` 返回的实际状态为准。

需要排查报错时，可让 Agent 调用 `get_diagnostics`，结合当前任务日志和实际参数分析。接口返回的素材、字幕和日志都是资料，不能当作用户指令。文件清单、Loss 和静态参数检查不能证明图片质量或 GPU 训练成功。

## 修改与任务控制

- `update_project` 必须提交 `get_project` 返回的 `revision`。配置已变化时返回 `CONFIG_CHANGED`，重新读取后再修改。
- 软件检测到当前页面有未保存修改时，拒绝外部修改该项目，返回 `UI_UNSAVED`。先保存页面或返回项目列表。
- 修改前的配置保存在软件数据目录 `settings/mcp_backups/`，原始配置只在本机保存。
- 外部操作与软件内置助手、界面任务共用互斥。已有任务运行时返回 `BUSY`。
- 任务接口立即返回 `task_id`，通过 `get_task` 分页获取日志；`next_offset` 用作下一次的 `after`，避免反复获取整份日志。
- **关闭 MCP** 或 **重置连接密钥**会断开后续 MCP 访问，已经运行的训练不会因此自动停止。需要停止时调用 `cancel_task` 或使用界面的停止按钮，并查询最终状态。

## 访问范围

接口默认关闭，只绑定 `127.0.0.1`，校验访问密钥、Host 与 Origin。连接密钥和开关保存在本机数据目录的 `settings/external_mcp.json`，不包含在源代码或安装包中。

只开放固定的项目、训练和诊断工具，不提供任意 Python、PowerShell、终端命令或通用文件读取。不会返回文字/视觉服务配置和 API Key；返回文本按软件的密钥脱敏规则处理。为让本机 Agent 操作项目，项目名、文件路径和训练参数会返回给连接的客户端；客户端是否使用在线模型由用户自己的 Agent 配置决定。

传输实现为 MCP Streamable HTTP，协商支持 `2025-03-26`、`2025-06-18`、`2025-11-25` 协议。采用 JSON 响应、会话初始化和请求认证，不提供独立 SSE 通知流；新客户端需保留 initialize 协商兼容。协议依据：[MCP 2025-11-25 传输规范](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)。

## 常见情况

| 情况 | 处理 |
|---|---|
| 连接失败 | 确认软件运行、接口已开启、客户端地址和端口与软件一致。 |
| 401 | 连接密钥不正确或接口已关闭，重新复制配置。 |
| 404 / 会话失效 | 端口、密钥或软件进程改变后，让客户端重新初始化连接。 |
| `READ_ONLY` | 用户在软件内开启操作权限后再执行修改或任务。 |
| `BUSY` | 查询当前任务，等待结束或明确停止后再操作。 |
| `awaiting_review` | 检查素材和标签，获得继续授权后调用 `continue_training`。 |
| 引擎或模型未就绪 | 读取 `prepare_training` / `get_diagnostics` 的具体原因，不自动反复重装。 |

