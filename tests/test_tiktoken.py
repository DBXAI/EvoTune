import tiktoken

def count_tokens(text: str, model: str = "gpt-4"):
    """
    计算文本在指定模型下的 token 数量
    :param text: 输入的字符串
    :param model: 模型名称 (如 "gpt-3.5-turbo", "gpt-4", "gpt-4o")
    :return: token 数量
    """
    # 获取对应模型的编码器
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

if __name__ == "__main__":
    # 在这里粘贴你要统计的文本
    sample_text = """            As an SQL optimization expert, your task is to diagnose and fix the following slow SQL query.
            You have the context of slow SQL and the root cause of the performance degradation as below:

            ## Context
            1. Query Information:
            • SQL: select l_returnflag, l_linestatus, sum(l_quantity) as sum_qty, sum(l_extendedprice) as sum_base_price, sum(l_extendedprice * (1 - l_discount)) as sum_disc_price, sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge, avg(l_quantity) as avg_qty, avg(l_extendedprice) as avg_price, avg(l_discount) as avg_disc, count(*) as count_order from lineitem where l_shipdate <= date '1998-12-01' - interval '80' day group by l_returnflag, l_linestatus order by l_returnflag, l_linestatus;
            • Execution Plan: [[{"Plan": {"Node Type": "Aggregate", "Strategy": "Sorted", "Partial Mode": "Finalize", "Parallel Aware": false, "Async Capable": false, "Startup Cost": 2303247.43, "Total Cost": 2303249.39, "Plan Rows": 6, "Plan Width": 236, "Group Key": ["l_returnflag", "l_linestatus"], "Plans": [{"Node Type": "Gather Merge", "Parent Relationship": "Outer", "Parallel Aware": false, "Async Capable": false, "Startup Cost": 2303247.43, "Total Cost": 2303248.83, "Plan Rows": 12, "Plan Width": 236, "Workers Planned": 2, "Plans": [{"Node Type": "Sort", "Parent Relationship": "Outer", "Parallel Aware": false, "Async Capable": false, "Startup Cost": 2302247.41, "Total Cost": 2302247.42, "Plan Rows": 6, "Plan Width": 236, "Sort Key": ["l_returnflag", "l_linestatus"], "Plans": [{"Node Type": "Aggregate", "Strategy": "Hashed", "Partial Mode": "Partial", "Parent Relationship": "Outer", "Parallel Aware": false, "Async Capable": false, "Startup Cost": 2302247.19, "Total Cost": 2302247.33, "Plan Rows": 6, "Plan Width": 236, "Group Key": ["l_returnflag", "l_linestatus"], "Planned Partitions": 0, "Plans": [{"Node Type": "Seq Scan", "Parent Relationship": "Outer", "Parallel Aware": true, "Async Capable": false, "Relation Name": "lineitem", "Alias": "lineitem", "Startup Cost": 0.0, "Total Cost": 1436969.35, "Plan Rows": 24722224, "Plan Width": 25, "Filter": "(l_shipdate <= '1998-09-12 00:00:00'::timestamp without time zone)"}]}]}]}]}, "JIT": {"Functions": 9, "Options": {"Inlining": true, "Optimization": true, "Expressions": true, "Deforming": true}}}]]
            • Execution Runtime: 24.332892894744873 s
            • Internal Metrics: Tuples Returned:59989212, Blocks Hit:115950, Blocks Read:1010599, Tuples Fetched:1367, Index Tuples Fetched:0, Sequential Tuples Read:59986052, Sequential Scans:3, Index Scans:0, Heap Blocks Hit:113943, Heap Blocks Read:1010599, Index Blocks Hit:1, Index Blocks Read:0
            
            2. Database Environment Information:
            • Database Type: postgresql
            • Workload Type: OLAP
            • Database Size: ['15 GB']
            • Database Schema: 
            • CPU: [98.9, 99.9, 99.15, 99.9, 99.95, 99.9, 100.425, 99.775, 0.0]
            • Read I/O: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            • Write I/O: [0.69140625, 0.7812500000000004, 1.0029296875, 0.29882812499999967, 0.0703125, 0.31054687499999944, 12.91796875, 25.80224609375005, 0.0]
            • Virtual Memory: [4.311836242675781, 4.311836242675781, 4.311836242675781, 4.311836242675781, 4.311836242675781, 4.311836242675781, 4.311836242675781, 4.311836242675781, 4.311836242675781]
            • Physical Memory: [0.4140777587890625, 0.6732177734375, 0.9137496948242188, 1.1422309875488281, 1.363983154296875, 1.566680908203125, 1.7573623657226562, 1.9385337829589844, 2.1070709228515625]
            • Net Received: [0.021902084350585938, 0.006246328353881835, 0.0024514198303222656, 0.0021886825561523455, 0.00030422210693359375, 0.0016045570373535139, 0.0023427009582519523, 0.0021333694458007834, 0.010493278503417969]
            • Net Sent: [0.017873764038085938, 0.004295825958251953, 0.0024514198303222656, 0.0021886825561523455, 0.00030422210693359375, 0.0016045570373535139, 0.0023427009582519523, 0.0021333694458007834, 0.010493278503417969]

            3. **Identified Root Cause**:
            suboptimal plan optimizer

            4. Historical Cases:

            5. Inspirations:

            ## Action Space  
            - If the root cause is "inappropriate query knobs", you MUST perform actions within Knob Space using SET knob = value.
            - If the root cause is "suboptimal plan optimizer", you MUST perform actions within Plan Optimizer Space using PostgreSQL comment-style hints ONLY. Output ONLY the hint block like: /*+ Set(knob_name value) Set(knob_name value) ... */. Do NOT include the SQL query.
            - If the root cause is "missing indexes", you MUST perform actions within Index Space using CREATE INDEX IF NOT EXISTS or DROP INDEX IF EXISTS.
            - If the root cause is "poorly written queries", you MUST perform actions within Query Rewrite Space using rewrite SQL.
            - **If multiple root causes are identified, you MUST consider all FIX_ACTION outputs jointly and determine the optimal solution that addresses the combined root causes, while strictly limiting actions to the relevant action spaces of the identified root causes. Do not perform any actions outside of these specified root causes.**

            1. Knob Space:  
            - Knob Space is strictly limited to the following knobs: 
            constraint_exclusion: type: enum, min: null, max: null, default: partition
            cpu_index_tuple_cost: type: float, min: 0.0, max: 10, default: 0.005
            cpu_operator_cost: type: float, min: 0.0, max: 1, default: 0.0025
            cpu_tuple_cost: type: float, min: 0.0, max: 10, default: 0.01
            cursor_tuple_fraction: type: float, min: 0.0, max: 1.0, default: 0.1
            default_statistics_target: type: integer, min: 1, max: 5000, default: 100
            effective_cache_size: type: integer, min: 1, max: 2147483647, default: 524288
            from_collapse_limit: type: integer, min: 1, max: 24, default: 8
            geqo: type: boolean, min: 0, max: 1, default: 0
            hash_mem_multiplier: type: float, min: 1.0, max: 1000.0, default: 2.0
            jit: type: boolean, min: 0, max: 1, default: 1
            jit_above_cost: type: float, min: -1.0, max: 500000, default: 100000.0
            jit_expressions: type: boolean, min: 0, max: 1, default: 1
            jit_inline_above_cost: type: float, min: -1.0, max: 5000000, default: 500000.0
            jit_optimize_above_cost: type: float, min: -1.0, max: 5000000, default: 500000.0
            jit_tuple_deforming: type: boolean, min: 0, max: 1, default: 1
            join_collapse_limit: type: integer, min: 1, max: 24, default: 8
            max_parallel_workers: type: integer, min: 0, max: 1024, default: 8
            max_parallel_workers_per_gather: type: integer, min: 0, max: 1024, default: 2
            min_parallel_index_scan_size: type: integer, min: 0, max: 715827882, default: 64
            min_parallel_table_scan_size: type: integer, min: 0, max: 715827882, default: 1024
            parallel_setup_cost: type: float, min: 0.0, max: 50000.0, default: 1000.0
            parallel_tuple_cost: type: float, min: 0.0, max: 10, default: 0.1
            plan_cache_mode: type: enum, min: null, max: null, default: auto
            random_page_cost: type: float, min: 0.0, max: 100, default: 4.0
            seq_page_cost: type: float, min: 0.0, max: 100, default: 1.0
            work_mem: type: integer, min: 64, max: 2147483647, default: 4096

            2. Plan Optimizer Space: 
            - Plan Optimizer Space is strictly limited to the following knobs: 
            enable_async_append: type: boolean, min: 0, max: 1, default: 1
            enable_bitmapscan: type: boolean, min: 0, max: 1, default: 1
            enable_gathermerge: type: boolean, min: 0, max: 1, default: 1
            enable_hashagg: type: boolean, min: 0, max: 1, default: 1
            enable_hashjoin: type: boolean, min: 0, max: 1, default: 1
            enable_incremental_sort: type: boolean, min: 0, max: 1, default: 1
            enable_indexonlyscan: type: boolean, min: 0, max: 1, default: 1
            enable_indexscan: type: boolean, min: 0, max: 1, default: 1
            enable_material: type: boolean, min: 0, max: 1, default: 1
            enable_memoize: type: boolean, min: 0, max: 1, default: 1
            enable_mergejoin: type: boolean, min: 0, max: 1, default: 1
            enable_nestloop: type: boolean, min: 0, max: 1, default: 1
            enable_parallel_append: type: boolean, min: 0, max: 1, default: 1
            enable_parallel_hash: type: boolean, min: 0, max: 1, default: 1
            enable_partition_pruning: type: boolean, min: 0, max: 1, default: 1
            enable_partitionwise_aggregate: type: boolean, min: 0, max: 1, default: 0
            enable_partitionwise_join: type: boolean, min: 0, max: 1, default: 0
            enable_seqscan: type: boolean, min: 0, max: 1, default: 1
            enable_sort: type: boolean, min: 0, max: 1, default: 1
            - **IMPORTANT**: When using Plan Optimizer Space, output ONLY the hint block. Do NOT include the original SQL. Example: /*+ Set(enable_hashjoin on) Set(enable_sort on) */

            3. Index Space: 
            - The naming convention for newly created indexes must follow the pattern: (table_name)_(col1)_(col2)_idx.
            - Index creation should be strictly guided by the slow_sql queries.
            - Avoid generating duplicate or subsumed indexes, and do not drop primary keys, unique indexes, or any indexes that enforce constraints.
            - Existing indexes in the database: 

            4. Query Rewrite Space: 
            - The output must always be a semantically equivalent SELECT statement that produces exactly the same result set as the original query.
            - The following rules are provided in the format ["rule name": "rule description"], consider the rules when rewriting the SQL:
            "name": "SORT_UNION_TRANSPOSE_MATCH_NULL_FETCH", "rewrite_rules_structured": ["conditions": "1. A SQL query performs a `UNION ALL` operation on multiple select statements. 2. Following the `UNION ALL`, there's an `ORDER BY` clause intending to sort the results, with no `OFFSET` clause present in the query.3. Each select statement involved in the `UNION ALL` could potentially benefit from pre-application of the order defined by the `ORDER BY` clause, without violating the intended sort order across the combined result set.", "transformations": "1. Identify the `ORDER BY` clause following a `UNION ALL` that combines multiple select statements and verify there's no associated `OFFSET` directive. 2. For each select statement involved in the `UNION ALL`:   - Apply the `ORDER BY` clause directly to the select statement if doing so does not change the intended order of the final result set. This effectively means prepending an `ORDER BY` to each select statement based on the `ORDER BY` clause initially following the `UNION ALL` operation. 3. Retain the original `ORDER BY` clause following the `UNION ALL`. This ensures that the final result set is ordered as intended, taking into account that individual select statements are now pre-sorted. 4. Implement this transformation only if it's determined that sorting individual select statements before the `UNION ALL` would maintain the overall desired sort order and is expected to either improve performance by reducing sorting overhead or have no negative impact."]

            ## Constraints
            - You **must** use the following MCP APIs when producing the fix:
            ['indexes_info_collect(postgres_config)', 'indexes_action_space(query)', 'plan_info_collect(query, postgres_config)', 'plan_action_space(query)']
            - Allowed optimization methods: change online knobs, add indexes, add query hints, rewrite SQL, or other safe optimization actions.
            - Fix must be executable directly in postgresql.
            - The fix should be safe and should not drop or delete data unless explicitly required by the root cause.

            ## Output Format
            You must output ONLY in the following format, with explanation and fix action:

            Explanation:
            <concise technical reason why the root cause is causing the slowdown now>

            Fix_Action:
            <your fix SQL or optimization action here>

            Query_Rewrite:
            <yes or no>

            ## Output Requirements
            - "Explanation" must be a short, precise, technical summary — no generic statements.
            - If the fix rewrites the SQL, "Query_Rewrite" must be "yes", else "no".
            - If rewriting, provide the optimized SQL under Fix_Action.
            - If adding indexes or changing knobs, provide the exact SQL or command.
            - Do not output any extra sections, comments, or steps.
"""
    
    model_name = "gpt-4o-mini"  # 也可以改成 "gpt-3.5-turbo" 或 "gpt-4o"
    token_count = count_tokens(sample_text, model_name)
    
    print(f"模型 {model_name} 下的 token 数: {token_count}")
