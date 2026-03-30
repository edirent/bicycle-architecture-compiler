# Two-Table Compiler Prototype (Full M/N/P0/P2/P3)

这个目录是独立 Python prototype（Module I），核心语义保持不变：

- `M` = reachable data-only tails
- `N` = native logical measurements `(Q, P)`
- 内部状态 `State(hidden_class, tail)`
- `P0/P2` 只做 source，`P3` 只做 edge，Dijkstra 做 shortest-path
- 不包含 mapping / scheduling / ISA lowering
- runtime 不 import Rust compiler

## 1. LocalTable/JointTable Schema

### 1.1 LocalTable

`LocalTable` 只存 native inventory，不存 target->answer。

```python
NativeEntry(
    native_id: str,
    frame_id: str,
    head: str,      # I/X/Y/Z
    tail: Tail,     # 11q
    scope: str,     # intra_block_native / cross_logical_block_native / inter_module_bell
    support: Tuple[str, ...],
    metadata: Tuple[Tuple[str, str], ...],
)
```

说明：

- `scope` 用于区分成本模型与 remote-count
- `support/metadata` 用于标注 logical sheets/support family
- `X1P⊗X1P'` 这类同 code-block 跨 primed/unprimed 的 native joint measurement，作为 `cross_logical_block_native` 放进 `LocalTable`（`P0`），不是 `P2`

### 1.2 JointTable

`JointTable` 只存 protocol rules：

- `P2Rule`: 组合 source 规则
  - `lhs_head / rhs_head / require_same_head / require_nonidentity_heads`
  - `required_relation` (`commute` / `anticommute` / `any`)
  - `tail_op`（当前 `tail_product`）
  - `total_cost`（P2 source 总成本，不是边增量）
  - `allowed_input_scopes`（仅允许 `intra_block_native` / `cross_logical_block_native`）
- `P3Rule`: transition 规则

## 2. Asset Generation / Caching

缓存目录：

```text
prototype/two_table_compiler/generated/
  gross_local_table.json
  joint_table.json
  random_corpus_seed7.json
```

运行默认优先加载缓存；缺失时可自动生成或手动刷新。

```bash
python -m prototype.two_table_compiler.build_assets --refresh-all
python -m prototype.two_table_compiler.build_assets --refresh-local
python -m prototype.two_table_compiler.build_assets --refresh-joint
python -m prototype.two_table_compiler.build_assets --refresh-random --seed 7 --count 100
```

## 3. LocalTable Data Source

`gross_local_table.json` 来自两部分：

1. Gross 原生表导出（主来源）
   - `results/native_11q_real.csv`
   - exporter: `crates/bicycle_cliffords/src/bin/export_native_11q.rs`
   - definitions: `crates/bicycle_cliffords/src/native_measurement.rs` + `crates/bicycle_cliffords/src/measurement.rs`
2. 补充 seed（当前最小可用 cross-logical-native）
   - `prototype/two_table_compiler/seeds/cross_logical_native_seed.json`
   - 包含 `X1P⊗X1P'` family 的 native entry（scope=`cross_logical_block_native`）

Python runtime 仅加载 JSON，不运行 Rust 编译器路径。

## 4. Target Classification / Canonicalization

搜索入口先做 target canonicalization：

1. primed/unprimed 逻辑表达式规范化（例如 `X1P⊗X1P'`）
2. 在当前 frame 做 native family lookup（基于 LocalTable metadata）
3. 命中 native family 后进入当前 frame 搜索；`P0`/`P2` source 均在该 frame 动态构造
4. 逻辑 target 未命中时返回“当前 frame 不可达”（不自动 Bell fallback）

然后才进入常规 `P2/P3` 搜索。

## 5. Cost Model / Remote Count

- `intra_block_native`: native base cost
- `cross_logical_block_native`: native cost（可由 metadata 配置 small shift）
- `inter_module_bell`: 在对应规则成本上加 Bell penalty

`remote_inter_module_count` 只统计 `scope == inter_module_bell` 的步骤。
`cross_logical_block_native` 不进入 remote/Bell 统计。

语义边界：

- `P0` = 当前 frame 直接 native measurement（`intra_block_native` + `cross_logical_block_native`）
- `P2` = 两个 native measurements 按 JointTable 协议组合出的 source
- `inter_module_bell` 既不是 `P0` 也不是 `P2` 输入
- frame miss 返回 frame-unreachable，不自动 Bell fallback

## 5.1 Search Status Contract

对 reachability / budget 相关语义，内部接口显式区分四种状态：

- `FOUND_OPTIMAL`
- `FOUND_REACHABLE_UPPER_BOUND`
- `SEARCH_TRUNCATED`
- `RULE_UNREACHABLE`

其中：

- `FOUND_OPTIMAL`
  - reachable = True
  - optimal = True
- `FOUND_REACHABLE_UPPER_BOUND`
  - reachable = True
  - optimal = False
  - 含义是“已有 witness path，但预算下尚未证明最优”
- `SEARCH_TRUNCATED`
  - reachable = False
  - truncated = True
  - 绝不能再被当成 `RULE_UNREACHABLE`
- `RULE_UNREACHABLE`
  - reachable = False
  - 只有这个状态才允许写成 frame-unreachable / rule-unreachable

额外 contract：

- no-budget（`max_popped_states=None`，CLI 上等价于 `--max-popped-states 0`）时：
  - 不应出现 `FOUND_REACHABLE_UPPER_BOUND`
  - 不应出现 `SEARCH_TRUNCATED`
- benchmark / report / story 层统一以 `synthesize_search()` 为准，不能再从 `plan is None` 推 unreachable
- `synthesize()` 只保留给旧测试/兼容层；新的 benchmark / report / story 语义判断禁止依赖它
- `target_tail` 查询只是对全量 `P2` source 的 post-filter，不改变搜索图本身
- `target-aware P2` 不能重新退化成缩图搜索
- 若理论上规则完备，则 `RULE_UNREACHABLE` 应优先视为执行层 bug 信号，而不是默认可接受现象
- 旧四态修复之前的 benchmark / report artifact 视为 stale，不应直接和当前结果对比

## 6. Random Testing Corpus

随机语料沿用 gross 侧规则并固定 seed 导出为 JSON：

- `crates/bicycle_benchmark/src/random.rs`
- `crates/bicycle_compiler/src/random_check.rs`

规则：每 qubit 均匀采样 `{I,X,Y,Z}`，拒绝 all-identity。

## 7. CLI / Tests

```bash
python -m prototype.two_table_compiler.demo --target XIIIIIIXIII
python -m prototype.two_table_compiler.demo --target "X1P⊗X1P'"
python -m unittest discover prototype/two_table_compiler
```

## 8. Boundary

这个 prototype 继续保持 Module-I 语义与独立运行，不回退到旧 gross compiler runtime wrapper。
