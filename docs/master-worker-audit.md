# 李规划/审批，马证据工作的跨仓契约

基线为 Controller `5f21684`；不改 DB、候选生成内核、视觉/token-lean、银行解析。
全局 `AUDIT_SKILL_LOCK` 仍覆盖本轮所有模型调用和工作线程退出；不同案件不交错执行。
2400 秒整体 deadline、240 秒单步、0.88 审批阈值、输出限额重试、按交易拆片和 partial 保全保留。

## 执行与并发边界

李初始规划串行完成。任务必须带 `operation`（`organize` / `search_amount` /
`compare_details` / `review_anomaly`）、`transaction_ids`、`receipt_ids`、`amount`（可 null）、
`depends_on`、策略和优先级。旧任务仅含 strategy/objective/priority 或 prio 时，由服务端
按策略确定候选范围及默认操作。高优先级先执行，但依赖先于优先级；未知依赖/循环依赖
blocked，不伪造成功。相同操作、金额和范围只分配一次。未分配候选最后异常复核。
初始 scope_catalog 只是有令牌预算的 ID 预览，省略量显式记录；不会把审核容量降到64。
任务空范围由服务端扩为该策略的完整可用范围，不只限预览；最终审批只送当前片的候选。

单个 `ThreadPoolExecutor(max_workers=1)`，最多一个下一批预处理 future。
马对下一批实际执行分类、日期排序、金额索引查询、重复候选查询和按需明细缓存时，
李可以同时通过模型审批上一批；不是提示词假并发。
**本版本所有模型调用仍串行**，最多一个本地模型请求，没有同时跑两个模型请求。
这种保守实现不要求修改本地模型并发配置。步骤/usage 编号及聚合发布有锁，笔记/任务
更新和 progress 发布按锁序串行化。event barrier fake 测试验证实际重叠，不依赖 sleep。

取消/deadline 后不提交下一请求/future；已提交的短 CPU 工作退出并保存已完成内容，
线程退出前不释放全局案件锁。只有李完成审批的 decisions 可应用。最终输出截断拆片
复用父片已完成 observations，过滤为子范围，不再次调用马。员工报销组不切成单票凑单。
本轮李已经确认及前轮 Main 实际采纳的交易/凭证不再下发模型；尚未确认时的有界预取可保留缓存，但
不得凭缓存自动确认。收入扣费/加成允许基于内核差额和公司/日期支持作披露性推断，
不强求票面手续费字样。工资差额仅 suspected，员工无证据报销最后留疑。

## Worker wire 与唯一决策方

马的新输出只有 `observations`、`task_results`、`cache_notes`、`summary`、`risks`，
不请求 decisions、recommendation、confidence。observation 提供 transaction_id、
receipt_upload_ids、group_id（非组 null）、finding、evidence、unresolved。
历史 worker.decisions 仅以只读适配转为候选证据，不继续进入新 wire 或公共 worker steps。

李最终输入是本片任务、内核候选、精选 observations 和案件摘要，不是全本笔记，
也不再将完整 plan 反复附给每个 worker 片。所有关系建议（包括 suggest/leave_unmatched）
必须是内核允许且 observation 覆盖的完整关系；只有已知内核交易可以作不含凭证的
未解决说明。group_id 对应的凭证组必须完整，不能取子集或混合多组/单票凑出新组。
重复交易、重复凭证和跨案未知 ID 拒绝。match 还须李 confidence >= 0.88、存在非空
证据和 finding、没有 unresolved。马不再先给一套置信度让李重复裁决。

## agent_state / 双 8MiB 笔记

公共结果和 progress 都用 `agent_state`；内部续轮输入用 `context['_agent_state']`。
固定键为 version=1、audit_id、run_id、notebooks、task_lists；两个 owner 都固定为
`audit_planner` / `evidence_worker`。notebook 包含 limit_bytes、used_bytes、entries
（key/kind/content/source_fingerprint）、evicted_entries、overflow。
task_lists 是各自真实任务数组，包含 task_id/operation/status/objective/depends_on/
transaction_ids/receipt_ids/amount/priority/parent_task_id；状态改变即发布 progress。

每本完整 notebook 的 canonical JSON（ensure_ascii=False，紧凑分隔符）UTF-8
字节数 <= `8*1024*1024`；used_bytes 包括 envelope、自身计数字段及 entries。
超限只整条淘汰可重建派生缓存，记录 evicted_entries；单条过大拒绝并记 overflow，
绝不剪坏 JSON。原资料只在原输入保存，已审批 decisions 只在权威 steps/result 保存，
笔记仅保存摘要和步骤引用，故笔记淘汰不会删除原始资料/已确认决定。
每位最多 512 项任务，已结束任务可整条淘汰计 task_evictions；全为活动任务时明确报错。

仅 audit_id 与 run_id 同时相等才续用缓存；source_fingerprint 为完整 compact 源对象
的 SHA-256，内容变化使基本/详细投影失效。基础缓存保留金额、币种、方向、类型、
日期与角色及原支付分项；按需要取 payer/payee/party、账户、reference、taxes。
索引从指纹已验证的基本事实构建，实际用于 exact amount（含1200）和同金额重复候选
检索；同金额消歧才自动升级到明细。按 income/expense/refund/payroll 分类、实际
payment/sales/settlement 日期排序，不将 invoice 创建/到期日当付款日。
查询仍限于内核给定候选，金额索引不能自行制造新候选关系。

stats 展示 cache_hits/cache_misses/invalidations、amount_searches/exact_amount_hits/
duplicate_candidates、evidence_reuses 和任务溢出/淘汰。缓存假设不会变成会计规则；
planner 的假设笔记不作为模型输入规则。state 和 source_inventory 不计入 4m 源字符预算
（正常/partial 共用同一计数函数），不附在每个 step.result，
不把8MiB全本发给模型。正常/partial 返回 state；格式等硬失败仍抛 AuditSkillError，
但 context/progress 保留 state，HTTP 502 保留原字符串 detail 并增添 agent_state/steps/usage。

## 全量库存初始化与 Main 配合

Main 开始渐进审核即 bootstrap 双 owner state 并写当前 agent-state，即使全 exact 或
完全无候选也保存双本本、完成的确定性整理任务及原会计报告，零 Controller/模型调用。
每轮 source_inventory 仍含全部流水和独立子票，包括 confirmed、excluded、needs_review；
只有 counts/source_inventory_summary 进入 planner，worker 仍仅接受内核残差候选。

两本都按源写 source_basic，并另存 128 项一片的 source_index、独立 summary、共同
strategy_order/objectives 的 planner_seed。不把全库存塞成一条会被整体驱逐的 entry。
完整 compact 源 canonical JSON（sort_keys=True、ensure_ascii=False、紧凑分隔符、禁止 NaN）
UTF-8 bytes 的 SHA-256 是两仓相同 source_fingerprint；基础投影/索引按原符号、类型、
退款及日期角色登记，非 actual 的 payment_due/document_issue 不当事件日期。

bootstrap 的 inventory:base 与 Li plan 的 task-1 等 ID 分离。Li 的显式 organize 命令
实际派生 inventory:plan:<iteration>:<command task_id>，马登记全量源/索引后状态 completed，
记录 origin=li_organize_command、derived_from_task_id、status_source=deterministic 和全源计数；
原 candidate scoped organize/search_amount/compare_details 命令与依赖继续执行。
重复登记只命中已有相同指纹缓存，不清本、不断言模型已整理，不擦除已审批决定/既有笔记。
基础身份信息按需取；原 scoped OCR 摘录仅随 compare_details 进入该片，不改变全源指纹。

## Main / Interface 其余配合边界

- Main 为一次审核稳定传 `_run_id`，续轮把公共 agent_state 放回 `_agent_state`，并继续
  传 `_prior_steps`、`_deadline`、iteration，以及已确认交易/凭证 ID；换案件或换 run 不续本。
- 保留全部 compact receipt 的原财务事实字段，特别是 financial_facts.amount_decimal、
  transaction_time_iso/role、document_kind、amount_components、payer/payee、reference_numbers、
  account_numbers、taxes；不得把分项总额合并成假净额。跨轮排除只认 Main 的实际 confirmed IDs，
  不从未经 Main 采纳的 _prior_steps/confirmed_decisions 重新推断确认。
- 持久化/透传正常、partial 和 502 的 agent_state、steps、usage；只应用李 completed
  final_assessment 的已校验 decisions，不应用马 observations 或缓存假设。
- Interface 双任务列表绑定两个固定 owner；双小本本预览 entries.content、used_bytes/
  limit_bytes、evicted_entries/overflow 和 stats，不展示一个空备注冒充笔记；大 state 不
  作为聊天 prompt。进度接口返回真实任务状态。Controller 仍保留最近32个内存进度快照，
  跨进程/重启保存由 Main 负责。
- DB029 去掉64上限保持不动；此改动不迁移DB、不恢复64上限、不重置资料，不 commit/push/deploy。