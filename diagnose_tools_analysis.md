# diagnose_tools.py 代码使用情况分析报告

## 执行流程追踪

**主流程**: `main.py` → `DBAgent.run()` → `ActionManager.step()` → `diagnose_tools.base_information_collect()`

**辅助流程**: `prompt.py` → `SQLRuleMatcher.get_rule_descriptions()`
**辅助流程**: `action_manager.py` → `SQLRuleMatcher.match_rules()`, `SQLRuleMatcher._rule_used_in_rewrite()`, `SQLRuleMatcher.get_rule_info()`

---

## ✅ 被调用的代码（约 450 行，27%）

### 1. **base_information_collect()** (521-536行，16行)
- **调用位置**: 
  - `action_manager.py:106`
  - `simple_db_agent.py:157`
- **用途**: 收集数据库基本信息
- **状态**: ✅ **被使用**

### 2. **SQLRuleMatcher 类** (36-400行，约 365行)
- **调用位置**:
  - `prompt.py:283` - 创建实例并调用 `get_rule_descriptions()`
  - `action_manager.py:517, 526` - 创建实例并调用 `match_rules()`, `_rule_used_in_rewrite()`, `get_rule_info()`

#### 被调用的方法：
- ✅ `__init__()` - 初始化
- ✅ `match_rules()` - 匹配规则
- ✅ `get_rule_descriptions()` - 获取规则描述
- ✅ `get_rule_info()` - 获取规则信息
- ✅ `_rule_used_in_rewrite()` - 检查规则是否在重写中使用
- ✅ `_load_rules()` - 内部调用
- ✅ `_has_union_all()`, `_has_order_by()`, `_has_window_functions()` 等所有 `_has_*` 方法 - 内部调用
- ✅ `_get_rules_by_pattern()` - 内部调用
- ✅ 所有 `_check_*` 方法（18个）- 被 `_rule_used_in_rewrite()` 内部调用

### 3. **extract_tables()** (408-453行，46行)
- **调用位置**: 
  - 被 `find_related_views()` 调用（但 `find_related_views()` 未被使用）
  - 被多个未使用的 MCP 函数调用
- **状态**: ⚠️ **间接使用**（通过未使用的函数）

### 4. **extract_columns()** (456-518行，63行)
- **调用位置**: 
  - 被 `find_related_views()` 调用（但 `find_related_views()` 未被使用）
  - 被多个未使用的 MCP 函数调用
- **状态**: ⚠️ **间接使用**（通过未使用的函数）

### 5. **clean_sql()** (402-405行，4行)
- **调用位置**: 
  - 被多个未使用的函数内部调用
  - 被 `find_related_views()` 调用
- **状态**: ⚠️ **间接使用**（通过未使用的函数）

---

## ❌ 未被调用的代码（约 1200 行，73%）

### 1. **MCP 工具函数**（10个函数，约 400+ 行）

这些函数都使用 `@mcp.tool()` 装饰器，原本设计为通过 MCP 服务器调用，但由于 MCP 服务器功能已被禁用，这些函数从未被调用：

#### 信息收集类（5个）：
- ❌ `indexes_info_collect()` (662-679行，18行)
- ❌ `query_info_collect()` (683-699行，17行) 
  - 内部调用了：`find_redundant_where_conditions_recursive()`, `count_nested_selects()`, `find_duplicate_subqueries()`, `find_related_views()`
- ❌ `system_knobs_info_collect()` (703-712行，10行)
- ❌ `query_knobs_info_collect()` (716-725行，10行)
- ❌ `plan_info_collect()` (729-822行，94行)

#### 动作空间定义类（5个）：
- ❌ `indexes_action_space()` (827-838行，12行)
- ❌ `system_knobs_action_space()` (842-876行，35行)
- ❌ `rewrite_action_space()` (880-981行，102行)
- ❌ `query_knobs_action_space()` (985-1017行，33行)
- ❌ `plan_action_space()` (1021-1067行，47行)

### 2. **未使用的优化函数**（6个函数，约 500+ 行）

这些函数在代码注释中标记为 "no use" 或 "action generate (no use)"：

- ❌ `update_statistics()` (1071-1143行，73行) - 注释：`# action generate (no use)`
- ❌ `join_order_optimization()` (1146-1312行，167行)
- ❌ `optimize_index()` (1315-1391行，77行)
- ❌ `optimize_repeatedly_subqueries()` (1394-1473行，80行)
- ❌ `optimize_query_knob()` (1476-1568行，93行)
- ❌ `rewrite_query()` (1571-1650行，80行)

### 3. **仅被未使用函数调用的辅助函数**（约 200+ 行）

这些函数只被上述未使用的 MCP 函数或优化函数调用：

- ❌ `is_redundant_comparison()` (539-552行，14行) - 只被 `query_info_collect()` 调用
- ❌ `find_redundant_where_conditions_recursive()` (555-573行，19行) - 只被 `query_info_collect()` 调用
- ❌ `count_nested_selects()` (576-589行，14行) - 只被 `query_info_collect()` 调用
- ❌ `extract_subqueries()` (592-606行，15行) - 只被 `find_duplicate_subqueries()` 调用
- ❌ `find_duplicate_subqueries()` (609-622行，14行) - 只被 `query_info_collect()` 调用
- ❌ `find_related_views()` (625-657行，33行) - 只被 `query_info_collect()` 调用

### 4. **MCP 相关代码**（约 10 行）

- ❌ `FastMCP` 导入和初始化 (10, 33行)
- ❌ `mcp = FastMCP("db_diagnosis")` (33行)
- ❌ `if __name__ == "__main__": mcp.run(transport="stdio")` (1653-1655行)

---

## 📊 统计总结

| 类别 | 行数 | 占比 | 状态 |
|------|------|------|------|
| **被直接使用的代码** | ~450行 | 27% | ✅ |
| **未被使用的代码** | ~1200行 | 73% | ❌ |
| **总计** | ~1656行 | 100% | |

### 详细分类：

1. **SQLRuleMatcher 类**: ~365行 ✅ **被使用**
2. **base_information_collect**: ~16行 ✅ **被使用**
3. **MCP 工具函数**: ~400行 ❌ **未使用**
4. **优化函数**: ~500行 ❌ **未使用**
5. **辅助函数（仅被未使用函数调用）**: ~200行 ❌ **未使用**
6. **MCP 相关代码**: ~10行 ❌ **未使用**
7. **extract_tables/columns（间接）**: ~109行 ⚠️ **间接使用**

---

## 🔍 结论

在 `main.py` 的执行流程中：

1. **实际被使用的代码约 27%**（主要是 `SQLRuleMatcher` 类和 `base_information_collect` 函数）
2. **未被使用的代码约 73%**，包括：
   - 所有 MCP 工具函数（因为 MCP 服务器功能被禁用）
   - 所有优化函数（标记为 "no use"）
   - 仅被未使用函数调用的辅助函数

### 建议：

1. **可以安全删除的代码**：
   - 所有 `@mcp.tool()` 装饰的函数
   - 所有标记为 "no use" 的优化函数
   - 仅被未使用函数调用的辅助函数
   - MCP 相关的导入和初始化代码

2. **需要保留的代码**：
   - `SQLRuleMatcher` 类（完整保留）
   - `base_information_collect()` 函数
   - `extract_tables()` 和 `extract_columns()`（虽然目前只被未使用的函数调用，但可能在未来有用）

3. **如果删除未使用的代码，可以减少约 1200 行代码（73%）**
