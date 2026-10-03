# Controller 模型等待队列与凭证补充接口

队列位于公共模型 transport，而非各 route；进程内所有 `ModelServerClient`
实例共用 local 通道，`OpenAIVisionClient` 与 `OpenAIBankStatementClient` 共用
external 通道。HTTP 应用使用请求上下文传递其共享队列和断连取消事件。
本地分类、李马每个推理步骤、陆奥、客服、聊天均通过此入口。健康检查与
模型启动/停止/状态查询不进入推理队列。李马不再持有整轮模型锁，因此
步骤之间可以让其他已排队请求先运行，不会递归获取同一个 gate。
马师傅的异步准备是确定性本地计算，提交线程时复制队列/取消上下文；
准备阶段不预占模型 slot，随后调用同一中间件的本地内部接口不会自锁。

## 默认值和边界

local/external 独立配置，均默认并发 **1**、最多等待 **32**、等待超时 **300 秒**。
并发仅允许 1–4；等待容量仅允许 0–256；等待超时须为有限正数且不超过
3600 秒。超过边界的配置会拒绝启动。这是保守 Controller 策略，不代表
任何外部模型账户的并发配额。

| 环境变量（CHANNEL 为 LOCAL 或 EXTERNAL） | 默认值 |
|---|---:|
| `OMNI_MODEL_QUEUE_CHANNEL_CAPACITY` | 1 |
| `OMNI_MODEL_QUEUE_CHANNEL_MAX_PENDING` | 32 |
| `OMNI_MODEL_QUEUE_CHANNEL_WAIT_TIMEOUT_SECONDS` | 300 |

例如设置 local 并发的实际变量为 `OMNI_MODEL_QUEUE_LOCAL_CAPACITY`。
FIFO 以到达公共 transport 的顺序为准。等待时不提交上游模型请求；满队列
立即返回 429；等待超时返回 504。运行中的整个 body/stream（包括静默等待）
占用 slot，直到上游连接关闭。超时覆盖等待与运行的总时间；原调用的时间
预算仍有效（凭证 180 秒、银行 360 秒、本地默认 3600 秒、李马单步预算）。

同步路由在线程池运行；异步调用方可使用 `ModelServerClient.async_chat()`、
`async_chat_json()` 或公共 `run_model_call()`，不要在 event loop 直接调用
同步方法。HTTP 断连/异步任务取消会移除 waiter，运行中由 watchdog 关闭
上游 socket，并等待 transport 退出后释放 slot，不能先释放配额再遗留后台
推理。重复取消也必须等待终止清理；李马 HTTP 路由使用同一清理协议。
`chat_json()` 的显式取消回调在流式和非流式请求中均有效。
DNS 解析或系统连接建立阶段受系统解析/连接超时约束；响应头、TLS
握手及 body/stream 的阻塞读取均支持 socket 中断。

队列不持久化，不保存 prompt、会话 payload 或用户标识；进程重启不会恢复
等待请求。关闭时唤醒 waiter，明确失败；连接因进程退出而断开的请求需要
调用方重试。Main durable quantization queue 与该集中模型准入层相互独立。
请维持单 Controller worker；多进程部署没有跨进程共享计数或全局配额。

### 错误协议

响应含 `detail` 与 `code`：

| HTTP | code | 含义 |
|---:|---|---|
| 429 | `model_queue_full` | 等待容量用尽；未执行 |
| 504 | `model_queue_wait_timeout` | FIFO 等待超时；未执行 |
| 504 | `model_request_timeout` | 运行/总预算超时；不应用部分输出 |
| 499 | `model_request_cancelled` | 取消；已断连的客户端通常收不到响应 |
| 503 | `model_queue_unavailable` | 服务/队列关闭，请重试 |
| 502 | `model_transport_error` | 上游连接中断；不应用部分输出 |
| 429 | `model_upstream_rate_limited` | 上游限流或额度限制，不自动重试 |
| 503 | `model_upstream_unavailable` | 上游暂不可用，不自动重试 |

准入层 429/503 提供 `Retry-After: 1`。上游 429/503 保留 HTTP 状态，
`Retry-After` 转为 1–3600 秒的安全整数（HTTP-date 转剩余秒数，缺失或无效为 1）。
其他供应商 HTTP 错误保持原有映射，但不回显供应商正文、URL、错误 reason、
认证头或密钥；urllib 错误响应在释放 slot 前关闭。
李马原有部分结果保全、步骤超时与取消恢复协议保持不变；准入等待超时
不会被误当作模型输出不足而递归拆分或重试。

## 状态接口

- `GET /internal/model-queue/status`：使用既有共享 `VISION_INTERNAL_TOKEN`，
  请求头为 `X-Vision-Token`；无需管理员 Cookie 或局域网来源。
  未配置 token 返回 503；缺失/错误 token 返回 401。Main 可复用现有凭证
  内部调用的 token 请求。
- `GET /api/model-queue`：沿用管理员 session Cookie、账号有效性/版本校验
  与允许网段限制；超级管理员，或具有 `dashboard.read` **或**
  `quantization.manage` 的账号可读。未登录 401、无权限/网段不符 403。
  GET 无需 CSRF；并不接受 Bearer admin token 替代会话。

### 浏览器地址与已安装 API

网关将 `/dashboard/api/` 去掉后转发至 Controller Unix Socket。因此管理 UI
应请求 **`GET /dashboard/api/api/model-queue`**，不是 `/dashboard/api/model-queue`，
也不是 `/api/model-queue`（网关裸 `/api/` 指向 Main）。浏览器携带既有
`omni_admin_session` Cookie；来源仍须通过管理员允许网段检查。
内部服务应通过 `/run/omni-ai-controller/controller.sock`（或安装配置的 socket）
请求 `GET /internal/model-queue/status` 并携带 `X-Vision-Token`。
普通用户 UI 不应获取该内部 token；如需展示，必须由 Main 提供受用户会话
保护的代理，只返回上述计数，不把 token 或 Controller 管理 Cookie 交给浏览器。

2026-10-03 只读核查：当前 active 服务的 WorkingDirectory 是
`/opt/ai_server/.releases/20261002-master-worker/omni_ai_controller`；该安装快照
没有本节两个队列状态路由或 `user_supplement` 字段。本工作树实现**尚未部署**，
不能将这些地址描述为当前已安装 API；本次未调用在线模型或改变安装服务。

唯一顶层字段为 `channels`，含 `external` 和 `local`，各通道仅有整数
`queued`、`running`、`capacity`、`max_pending`；不含 prompt、用户名、任务
ID、会话、密钥、payload、队列 ticket 或配置路径。

## 凭证补充：POST /internal/vision/receipts

新增可选 `user_supplement` 对象；省略它时旧请求与旧返回 schema 兼容。
只允许以下字段，未知字段或类型/长度不符返回 422：

| 字段 | 类型和限制 | 用途 |
|---|---|---|
| `note` | 可选 string，最多 30 字符 | 人工财务备注 |
| `manual_receipt` | 可选 string，最多 4000 字符 | 人工票据补充/纯文字票据 |
| `previous_quantization` | 可选 object 或 null | 由可信 Main 服务提供的此前量化 |

`previous_quantization` 输入的紧凑 UTF-8 JSON（不含空白）最多 **64 KiB**，
且拒绝非有限数字/非 JSON 对象/超过 32 层的结构。模型输入只投影财务白名单：
`text`（500 字符）、`primary_receipt_index`、`financial_facts`、`classification`、
`receipts`（最多 36 项及其财务 schema 字段）。嵌套字段递归白名单并限长；
usage、tokens、raw response、logs、prompt、用户补充嵌套和其他元数据不重送。
64 KiB 是此前量化的预算，不是人工文字的字符限制。整个 HTTP 凭证请求
另有 **46 MiB** 的流式 body 上限，超过时在 JSON 解析前返回 413。

模型消息顺序为 **原票 → 旧量化财务投影 → 人工补充**，可信 system 规则
明确所有来源和补充都是无指令权的数据，原票可见证据优先；不得执行补充
中的指令或自动采用它声称的分类，必须重新提取、重新分类并报告不确定性。
返回不回显 supplement，也不把它加入 Controller 队列数据库。

manual-only 可以用 `source_kind: "text"`，无图片时提供 `document_text` 或
非空 `user_supplement.manual_receipt`；`page_count` 可以省略，默认 1。
可附带 `filename` 作为元数据，不会因此被误判为缺少 base64 的图片请求。此时
不会声称存在验证过的 PDF 文字层或视觉证据；返回 recognition_mode 为
`manual_text`。纯文字仍是未验证人工资料。

`source_kind: "pdf_rendered"` 接受唯一的所选原页码，例如 3、7；传入总
`page_count` 时必须覆盖最大所选页，省略时按最大所选原页码推断。不得混入
PDF 文字层，不重编号、不补造连续页。返回 `source_pages` 仅引用实际提供
页码；`page_start/end` 只表示包围范围，`page_reviews` 仅含实际所选页。
保留现有最多 12 页和总图片 32 MiB 的限制。

凭证默认 `gpt-6.1-sol` 和 `reasoning_effort: low` 保持不变，明确保存的其他
模型选择不覆盖。银行只接入 external transport gate，不改变旧提取路径、
`gpt-6-sol` 参数、连续页检查或返回 schema。

### Main 客户端待改（本次只读，未修改 Main）

已核查 notebook-replacement Main 工作树的 `VisionProxyClient`：三个入口
`recognize()`、`recognize_document()`、`recognize_text_document()` 目前发送
`payload["supplement"]`，且 `supplement_evidence()` 添加 `source` 标记。
这与 Controller 契约不兼容。Controller 现在对误拼顶层 `supplement` 明确返回
422，避免成功响应却静默忽略补充；正确的 `user_supplement` 不接受 `source`。

Main 需把这三个入口的实际 HTTP 字段改为 `user_supplement`，仅传 `note`、
`manual_receipt` 及可选 `previous_quantization`。Main 内部 Python 参数可继续
叫 `supplement`，但内部证据的 `source` 标记不得原样放入 Controller 请求。
旧量化应从持久化数据提取财务投影放入 `previous_quantization`，不在
`document_text` 拼接操作指令、debug traces、usage 或整份原始模型输出；
Main 自己继续保留原始金额、页码与已确认银行事实。

`VisionProxyClient._request()` 当前只提取 `detail`，丢弃响应 `code` 和
`Retry-After`。应将三者连同原 HTTP 状态传到队列/业务错误处理，按有限重试
策略处理 429/503，不得把它们改成通用 502 或无限重试。状态轮询应走独立
GET 方法（不能使用要求 `status` 字段的识图 POST 响应校验），共用内部 token。
外部未知配额默认并发 1 不等于账户有一条已验证配额；需要单独验证后才调整。

## 测试

在 Controller 工作树运行 `.venv/bin/python -m pytest -q`。
测试导入使用惰性替身模型配置，不读取部署模型 secrets；网络测试只启动
loopback 假 HTTP 对端，所有推理输出由 mock 提供。覆盖双通道、FIFO/满队列、
容量/超时、跨 client/HTTP 请求、完整李马步骤穿插聊天、stream 持锁、静默
连接取消、ASGI 断连、event loop 响应、补充/预算、原页码与状态鉴权。