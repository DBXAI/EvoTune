import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel

claude_api_key = "sk-6B29GYJUvOdinhDgD4Be997d5b2a4e399dA5B7F2A275F3B6"  # 需要设置实际的API密钥
claude_client = AsyncOpenAI(base_url="https://api.gpt.ge/v1", api_key=claude_api_key)

def get_agent_instructions():
    """获取主Agent的指令"""
    return """You are an expert in database performance diagnosis and SQL optimization.
        Your role is to:
        1. Understand detailed database and query execution context.
        2. Identify how the root cause leads to performance degradation.
        3. Output exactly two labeled sections: **Explanation** and **SQL Fix**.
        4. Always make the SQL fix executable directly in the target database.
        5. Never include extra text, comments, or sections beyond the required format.
        6. Always apply the given MCP APIs when producing fixes, if applicable.
        7. Assume the provided execution plan, metrics, and environment info are accurate.
        8. Keep the Explanation concise and technical, avoiding generic descriptions."""

code_refactoring_agent = Agent(
    name="Claude agent",
    instructions=get_agent_instructions(),
    model=OpenAIChatCompletionsModel( 
        model="claude-3-7-sonnet-20250219",
        openai_client=claude_client,
    ),
)

async def main():
    messy_code = """
        As an SQL optimization expert, your task is to diagnose and fix the following slow SQL query.
        You have the context of slow SQL and the root cause of the performance degradation as below:

        ## Context
        1. Query Information:
        • SQL: select
        l_returnflag,
        l_linestatus,
        sum(l_quantity) as sum_qty,
        sum(l_extendedprice) as sum_base_price,
        sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
        sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
        avg(l_quantity) as avg_qty,
        avg(l_extendedprice) as avg_price,
        avg(l_discount) as avg_disc,
        count(*) as count_order
from
        lineitem
where
        l_shipdate <= date '1998-12-01' - interval '80' day
group by
        l_returnflag,
        l_linestatus
order by
        l_returnflag,
        l_linestatus;

        • Execution Plan: [[{"Plan": {"Node Type": "Aggregate", "Strategy": "Sorted", "Partial Mode": "Finalize", "Parallel Aware": false, "Async Capable": false, "Startup Cost": 2303247.43, "Total Cost": 2303249.39, "Plan Rows": 6, "Plan Width": 236, "Group Key": ["l_returnflag", "l_linestatus"], "Plans": [{"Node Type": "Gather Merge", "Parent Relationship": "Outer", "Parallel Aware": false, "Async Capable": false, "Startup Cost": 2303247.43, "Total Cost": 2303248.83, "Plan Rows": 12, "Plan Width": 236, "Workers Planned": 2, "Plans": [{"Node Type": "Sort", "Parent Relationship": "Outer", "Parallel Aware": false, "Async Capable": false, "Startup Cost": 2302247.41, "Total Cost": 2302247.42, "Plan Rows": 6, "Plan Width": 236, "Sort Key": ["l_returnflag", "l_linestatus"], "Plans": [{"Node Type": "Aggregate", "Strategy": "Hashed", "Partial Mode": "Partial", "Parent Relationship": "Outer", "Parallel Aware": false, "Async Capable": false, "Startup Cost": 2302247.19, "Total Cost": 2302247.33, "Plan Rows": 6, "Plan Width": 236, "Group Key": ["l_returnflag", "l_linestatus"], "Planned Partitions": 0, "Plans": [{"Node Type": "Seq Scan", "Parent Relationship": "Outer", "Parallel Aware": true, "Async Capable": false, "Relation Name": "lineitem", "Alias": "lineitem", "Startup Cost": 0.0, "Total Cost": 1436969.35, "Plan Rows": 24722224, "Plan Width": 25, "Filter": "(l_shipdate <= '1998-09-12 00:00:00'::timestamp without time zone)"}]}]}]}]}, "JIT": {"Functions": 9, "Options": {"Inlining": true, "Optimization": true, "Expressions": true, "Deforming": true}}}]]
        • Execution Runtime: 23.78244185447693 s
        • Internal Metrics: Tuples Returned:59990556, Blocks Hit:20038, Blocks Read:1106590, Tuples Fetched:1381, Index Tuples Fetched:0, Sequential Tuples Read:59986052, Sequential Scans:3, Index Scans:0, Heap Blocks Hit:17952, Heap Blocks Read:1106590, Index Blocks Hit:1, Index Blocks Read:0

        2. Database Environment Information:
        • Database Type: postgresql
        • Workload Type: OLAP
        • Database Size: ['15 GB']
        • Database Schema: {'all_customer': {'columns': [{'column_name': 'c_custkey', 'data_type': 'integer', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'c_name', 'data_type': 'character varying', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'c_address', 'data_type': 'character varying', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'c_nationkey', 'data_type': 'integer', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'c_phone', 'data_type': 'character', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'c_acctbal', 'data_type': 'numeric', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'c_mktsegment', 'data_type': 'character', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'c_comment', 'data_type': 'character varying', 'is_primary_key': 'NO', 'foreign_key': None}]}, 'customer': {'columns': [{'column_name': 'c_custkey', 'data_type': 'integer', 'is_primary_key': 'YES', 'foreign_key': None}, {'column_name': 'c_name', 'data_type': 'character varying', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'c_address', 'data_type': 'character varying', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'c_nationkey', 'data_type': 'integer', 'is_primary_key': 'NO', 'foreign_key': {'referenced_table': 'nation', 'referenced_column': 'n_nationkey'}}, {'column_name': 'c_phone', 'data_type': 'character', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'c_acctbal', 'data_type': 'numeric', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'c_mktsegment', 'data_type': 'character', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'c_comment', 'data_type': 'character varying', 'is_primary_key': 'NO', 'foreign_key': None}]}, 'lineitem': {'columns': [{'column_name': 'l_orderkey', 'data_type': 'integer', 'is_primary_key': 'YES', 'foreign_key': {'referenced_table': 'orders', 'referenced_column': 'o_orderkey'}}, {'column_name': 'l_partkey', 'data_type': 'integer', 'is_primary_key': 'NO', 'foreign_key': {'referenced_table': 'partsupp', 'referenced_column': 'ps_suppkey'}}, {'column_name': 'l_partkey', 'data_type': 'integer', 'is_primary_key': 'NO', 'foreign_key': {'referenced_table': 'partsupp', 'referenced_column': 'ps_partkey'}}, {'column_name': 'l_suppkey', 'data_type': 'integer', 'is_primary_key': 'NO', 'foreign_key': {'referenced_table': 'partsupp', 'referenced_column': 'ps_partkey'}}, {'column_name': 'l_suppkey', 'data_type': 'integer', 'is_primary_key': 'NO', 'foreign_key': {'referenced_table': 'partsupp', 'referenced_column': 'ps_suppkey'}}, {'column_name': 'l_linenumber', 'data_type': 'integer', 'is_primary_key': 'YES', 'foreign_key': None}, {'column_name': 'l_quantity', 'data_type': 'numeric', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'l_extendedprice', 'data_type': 'numeric', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'l_discount', 'data_type': 'numeric', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'l_tax', 'data_type': 'numeric', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'l_returnflag', 'data_type': 'character', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'l_linestatus', 'data_type': 'character', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'l_shipdate', 'data_type': 'date', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'l_commitdate', 'data_type': 'date', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'l_receiptdate', 'data_type': 'date', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'l_shipinstruct', 'data_type': 'character', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'l_shipmode', 'data_type': 'character', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'l_comment', 'data_type': 'character varying', 'is_primary_key': 'NO', 'foreign_key': None}]}, 'nation': {'columns': [{'column_name': 'n_nationkey', 'data_type': 'integer', 'is_primary_key': 'YES', 'foreign_key': None}, {'column_name': 'n_name', 'data_type': 'character', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'n_regionkey', 'data_type': 'integer', 'is_primary_key': 'NO', 'foreign_key': {'referenced_table': 'region', 'referenced_column': 'r_regionkey'}}, {'column_name': 'n_comment', 'data_type': 'character varying', 'is_primary_key': 'NO', 'foreign_key': None}]}, 'orders': {'columns': [{'column_name': 'o_orderkey', 'data_type': 'integer', 'is_primary_key': 'YES', 'foreign_key': None}, {'column_name': 'o_custkey', 'data_type': 'integer', 'is_primary_key': 'NO', 'foreign_key': {'referenced_table': 'customer', 'referenced_column': 'c_custkey'}}, {'column_name': 'o_orderstatus', 'data_type': 'character', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'o_totalprice', 'data_type': 'numeric', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'o_orderdate', 'data_type': 'date', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'o_orderpriority', 'data_type': 'character', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'o_clerk', 'data_type': 'character', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'o_shippriority', 'data_type': 'integer', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'o_comment', 'data_type': 'character varying', 'is_primary_key': 'NO', 'foreign_key': None}]}, 'part': {'columns': [{'column_name': 'p_partkey', 'data_type': 'integer', 'is_primary_key': 'YES', 'foreign_key': None}, {'column_name': 'p_name', 'data_type': 'character varying', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'p_mfgr', 'data_type': 'character', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'p_brand', 'data_type': 'character', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'p_type', 'data_type': 'character varying', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'p_size', 'data_type': 'integer', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'p_container', 'data_type': 'character', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'p_retailprice', 'data_type': 'numeric', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'p_comment', 'data_type': 'character varying', 'is_primary_key': 'NO', 'foreign_key': None}]}, 'partsupp': {'columns': [{'column_name': 'ps_partkey', 'data_type': 'integer', 'is_primary_key': 'YES', 'foreign_key': {'referenced_table': 'part', 'referenced_column': 'p_partkey'}}, {'column_name': 'ps_suppkey', 'data_type': 'integer', 'is_primary_key': 'YES', 'foreign_key': {'referenced_table': 'supplier', 'referenced_column': 's_suppkey'}}, {'column_name': 'ps_availqty', 'data_type': 'integer', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'ps_supplycost', 'data_type': 'numeric', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'ps_comment', 'data_type': 'character varying', 'is_primary_key': 'NO', 'foreign_key': None}]}, 'region': {'columns': [{'column_name': 'r_regionkey', 'data_type': 'integer', 'is_primary_key': 'YES', 'foreign_key': None}, {'column_name': 'r_name', 'data_type': 'character', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 'r_comment', 'data_type': 'character varying', 'is_primary_key': 'NO', 'foreign_key': None}]}, 'supplier': {'columns': [{'column_name': 's_suppkey', 'data_type': 'integer', 'is_primary_key': 'YES', 'foreign_key': None}, {'column_name': 's_name', 'data_type': 'character', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 's_address', 'data_type': 'character varying', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 's_nationkey', 'data_type': 'integer', 'is_primary_key': 'NO', 'foreign_key': {'referenced_table': 'nation', 'referenced_column': 'n_nationkey'}}, {'column_name': 's_phone', 'data_type': 'character', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 's_acctbal', 'data_type': 'numeric', 'is_primary_key': 'NO', 'foreign_key': None}, {'column_name': 's_comment', 'data_type': 'character varying', 'is_primary_key': 'NO', 'foreign_key': None}]}, 'system_knobs_action_space': {'columns': [{'column_name': 'id', 'data_type': 'integer', 'is_primary_key': 'YES', 'foreign_key': None}]}}
        • CPU: [97.9, 99.025, 99.4, 100.275, 99.45, 99.9, 99.9, 99.9, 73.9]
        • Read I/O: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        • Write I/O: [0.046875, 0.013671875, 0.210693359375, 0.11474609375000011, 0.107421875, 0.05419921874999991, 0.055664062499999965, 0.14453125000000003, 0.0546875]
        • Virtual Memory: [4.296939849853516, 4.311836242675781, 4.311836242675781, 4.311836242675781, 4.311836242675781, 4.311836242675781, 4.311836242675781, 4.311836242675781, 4.311836242675781]
        • Physical Memory: [0.07774734497070312, 0.13964080810546875, 0.13964080810546875, 0.13964080810546875, 0.13964080810546875, 0.13964080810546875, 0.13964080810546875, 0.13964080810546875, 0.13964080810546875]
        • Net Received: [0.005279541015625, 0.018500089645385742, 0.0030853748321533203, 0.008132934570312503, 0.005978107452392578, 0.005626320838928224, 0.003378629684448243, 0.003960967063903809, 0.019466400146484375]
        • Net Sent: [0.005279541015625, 0.018500089645385742, 0.0030853748321533203, 0.008132934570312503, 0.005978107452392578, 0.005626320838928224, 0.0037276744842529297, 0.003960967063903809, 0.019466400146484375]

        3. **Identified Root Cause**:
        inappropriate query knobs

        4. Historical Cases:
        

        5. Inspirations:
        

        ## Action Space  
        - If the root cause is "inappropriate query knobs", you MUST perform actions within Knob Space using SET knob = value.
        - If the root cause is "suboptimal plan optimizer", you MUST perform actions within Plan Optimizer Space using PostgreSQL comment-style hints (/*+ Set(knob_name value) Set(knob_name value) ... */).
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
        - Allowed optimization methods: change online knobs, add indexes, add query hints, rewrite SQL, or other safe optimization actions.
        - Fix must be executable directly in postgresql.
        - The fix should be safe and should not drop or delete data unless explicitly required by the root cause.

        ## Output Format
        You must output ONLY in the following format, with explanation and fix action:

        Explanation:
        <concise technical reason why the root cause is causing the slowdown now>

        FIX_ACTION:
        <your fix SQL or optimization action here>

        Query_Rewrite:
        <yes or no>

        ## Output Requirements
        - You must output plain text only. Do not use Markdown, bold (** **), italic (* *), bullet points, or any other formatting.
        - "Explanation" must be a short, precise, technical summary — no generic statements.
        - If the fix rewrites the SQL, "Query_Rewrite" must be "yes", else "no".
        - If rewriting, provide the optimized SQL under FIX_ACTION.
        - If adding indexes or changing knobs, provide the exact SQL or command.
        - Do not output any extra sections, comments, or steps.
    """

    result = await Runner.run(code_refactoring_agent, input=messy_code)
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())