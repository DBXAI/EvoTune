import psycopg
from psycopg.rows import dict_row
from typing import Dict, Any
from plumbum import local
from collections import defaultdict
import json
import time
from .resource_monitor import ResourceMonitor
from decimal import Decimal

class PostgresDB:
    def __init__(self, user, password, host, port, dbname, workload_type, postgres_path = None, postgres_data = None, log_path = None):
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.dbname = dbname
        self.db_type = 'postgresql'
        self.workload_type = workload_type
        self.postgres_path = postgres_path
        self.postgres_data = postgres_data
        self.log_path = log_path

        self.conn_str = self._connect_str()
        self.connection = None

    def _connect_str(self):
        return "host={host} port={port} dbname={dbname} user={user} password={password}".format(
                host=self.host,
                port=int(self.port),
                dbname=self.dbname,
                user=self.user,
                password=self.password
        )
    
    def connect(self):
        """建立并返回一个数据库连接"""
        if self.connection is None or self.connection.closed:
            self.connection = psycopg.connect(self.conn_str, autocommit=True, prepare_threshold=None)
        return self.connection

    def execute(self, sql: str, conn=None, timeout=0):
        if conn is None:
            conn = self.connect()
        
        with conn.cursor() as cur:
            try:
                cur.execute(f"SET statement_timeout = {timeout * 1000}") # 设置超时
                start_time = time.time()
                cur.execute(sql)
                end_time = time.time()
                if cur.description is not None:
                    result = cur.fetchall()
                else:
                    result = None
                return True, result, end_time - start_time, ""
            except psycopg.errors.SyntaxError as e:
                return False, None, -1, str(e)
            except psycopg.errors.QueryCanceled as e:
                return False, None, -1, str(e)
            except Exception as e:
                return False, None, -1, str(e)

    def fetch_results(self, sql, as_json=False, timeout=0):
        conn = self.connect()
        with conn.cursor() as cursor:
            try:
                cursor.execute(f"SET statement_timeout = {timeout * 1000}") # 设置超时
                cursor.execute(sql)
                if not cursor.description:
                    return []
                results = cursor.fetchall()
                if as_json:
                    columns = [col.name for col in cursor.description]
                    results = [dict(zip(columns, row)) for row in results]
                    return json.dumps(results, indent=4, ensure_ascii=False, default=self.decimal_default)
                if len(cursor.description) == 1:
                    results = [row[0] for row in results]
                    # 处理plan_json
                    if results and isinstance(results[0], (dict, list)):
                        return json.dumps(results, ensure_ascii=False)
                return results
            except psycopg.errors.QueryCanceled as e:
                print(f"查询超时 (>{timeout}秒): {str(e)}") 
                return []
            except Exception as e:
                print(f"查询执行失败: {str(e)}")
                return []
        
    def get_size(self):
        sql = f"""
        SELECT pg_size_pretty(pg_database_size('{self.dbname}'));
        """
        return self.fetch_results(sql)

    def get_views(self):
        sql = """
        SELECT viewname, definition FROM pg_views WHERE schemaname = 'public' AND viewname NOT IN ('pg_stat_statements', 'hypopg_list_indexes', 'hypopg_hidden_indexes', 'pg_stat_statements_info');
        """
        return self.fetch_results(sql)

    def get_tables(self):
        sql = """
        SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name NOT IN ('pg_stat_statements', 'hypopg_list_indexes', 'hypopg_hidden_indexes', 'pg_stat_statements_info');
        """
        return self.fetch_results(sql)

    def get_columns(self):
        sql = """
        SELECT column_name FROM information_schema.columns WHERE table_name NOT IN ('pg_stat_statements', 'hypopg_list_indexes', 'hypopg_hidden_indexes', 'pg_stat_statements_info') AND table_schema NOT IN ('information_schema', 'pg_catalog');
        """
        return self.fetch_results(sql)

    def fetch_schema_info(self):
        sql = """
        WITH
        pk_cols AS (
            SELECT kc.table_name, kc.column_name, 'YES' AS is_primary_key
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kc
            ON kc.constraint_name = tc.constraint_name
            AND kc.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'PRIMARY KEY'
            AND kc.table_schema = 'public'
        ),
        fk_cols AS (
            SELECT tc.table_name, kcu.column_name,
                ccu.table_name AS referenced_table,
                ccu.column_name AS referenced_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
            ON kcu.constraint_name = tc.constraint_name
            AND kcu.table_schema = tc.table_schema
            JOIN information_schema.constraint_column_usage ccu
            ON ccu.constraint_name = tc.constraint_name
            AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
            AND tc.table_schema = 'public'
        )
        SELECT c.table_name, c.column_name, c.data_type,
            COALESCE(pk.is_primary_key, 'NO') AS is_primary_key,
            fk.referenced_table, fk.referenced_column
        FROM information_schema.columns c
        LEFT JOIN pk_cols pk
        ON c.table_name = pk.table_name AND c.column_name = pk.column_name
        LEFT JOIN fk_cols fk
        ON c.table_name = fk.table_name AND c.column_name = fk.column_name
        WHERE c.table_schema = 'public'
        AND c.table_name NOT IN (
            'hypopg_hidden_indexes', 'hypopg_list_indexes',
            'pg_stat_statements', 'pg_stat_statements_info'
        )
        ORDER BY c.table_name, c.ordinal_position;
        """

        rows = self.fetch_results(sql)
        if not rows:
            return {}

        schema_dict = defaultdict(lambda: {"columns": []})
        for table_name, column_name, data_type, is_pk, ref_table, ref_col in rows:
            fk_info = None
            if ref_table and ref_col:
                fk_info = {"referenced_table": ref_table, "referenced_column": ref_col}
            schema_dict[table_name]["columns"].append({
                "column_name": column_name,
                "data_type": data_type,
                "is_primary_key": is_pk,
                "foreign_key": fk_info
            })

        return dict(schema_dict)

    def start_db(self):
        """启动PostgreSQL数据库服务"""
        if not all([self.postgres_path, self.postgres_data, self.log_path]):
            raise ValueError("postgres_path, postgres_data, and log_path must be configured to start the database.")
        pg_ctl_cmd = [
            f"{self.postgres_path}/pg_ctl",
            "-D", self.postgres_data,
            "--wait",
            "-t", "180",
            "-l", f"{self.log_path}/pg.log",
            "start"
        ]
        cmd = ["-u", "postgres"] + pg_ctl_cmd
        retcode, _, stderr = local["sudo"][cmd].run(retcode=None)
        if retcode != 0:
            raise Exception(f"Failed to start PostgreSQL: {stderr}")

        # 等待数据库准备好接受连接
        for _ in range(10):
            pg_isready_cmd = [
                f"{self.postgres_path}/pg_isready",
                "--host", self.host,
                "--port", str(self.port),
                "--dbname", self.dbname
            ]
            cmd = ["-u", "postgres"] + pg_isready_cmd
            retcode, _, _ = local["sudo"][cmd].run(retcode=None)
            if retcode == 0:
                return True
            time.sleep(5)
        raise Exception("PostgreSQL did not become ready in time.")

    def shutdown_db(self):
        """关闭PostgreSQL数据库服务"""
        if not all([self.postgres_path, self.postgres_data]):
            raise ValueError("postgres_path and postgres_data must be configured to shut down the database.")
        pg_ctl_cmd = [
            f"{self.postgres_path}/pg_ctl",
            "stop",
            "--wait",
            "-t", "180",
            "-D", self.postgres_data
        ]
        cmd = ["-u", "postgres"] + pg_ctl_cmd
        local["sudo"][cmd].run(retcode=None)

        # 确认数据库已关闭
        for _ in range(10):
            pg_isready_cmd = [
                f"{self.postgres_path}/pg_isready",
                "--host", self.host,
                "--port", str(self.port),
                "--dbname", self.dbname
            ]
            cmd = ["-u", "postgres"] + pg_isready_cmd
            retcode, _, _ = local["sudo"][cmd].run(retcode=None)
            if retcode != 0:
                return True
            time.sleep(2)
        raise Exception("Failed to shut down PostgreSQL.")

    def get_pid(self):
        pid = self.fetch_results("SELECT pg_backend_pid();")
        return pid[0]

    # def get_pid(self, data_dir="/var/lib/postgresql/14/main"):
    #     with open(os.path.join(data_dir, "postmaster.pid")) as f:
    #         pid = int(f.readline().strip())
    #     return pid

    def get_metrics(self, sql, interval=1, warmup=0, duration=300):
        """
        监控PostgreSQL主进程的资源（CPU、内存、IO等），返回平均值。
        interval: 采样间隔（秒）
        warmup: 预热时间（秒）
        duration: 监控总时长（秒）
        """
        pid = self.get_pid()

        monitor = ResourceMonitor(pid, interval, warmup, duration)
        monitor.run()

        self.execute(sql)

        monitor.terminate()
        cpu_usage, read_io, write_io, virtual_memory, physical_memory, net_recv, net_sent = monitor.get_monitor_data()
        return cpu_usage, read_io, write_io, virtual_memory, physical_memory, net_recv, net_sent

    def get_plan(self, sql):
        """
        获取SQL的执行计划。对于包含DDL语句的复杂SQL，提取主要的查询语句来获取计划。
        """
        # 清理SQL，去除注释和多余空白
        sql = sql.strip()
        
        # 如果SQL包含多个语句，尝试提取主要的查询语句
        if ';' in sql:
            statements = [stmt.strip() for stmt in sql.split(';') if stmt.strip()]
            # 查找第一个SELECT语句作为主要查询
            main_query = None
            for stmt in statements:
                if stmt.upper().startswith('SELECT'):
                    main_query = stmt
                    break
            
            if main_query:
                try:
                    plan_sql = f"EXPLAIN (FORMAT JSON) {main_query}"
                    return self.fetch_results(plan_sql)
                except Exception as e:
                    print(f"获取主要查询计划失败: {e}")
                    return None
            else:
                # 如果没有找到SELECT语句，尝试获取最后一个非DDL语句的计划
                for stmt in reversed(statements):
                    if not any(stmt.upper().startswith(keyword) for keyword in 
                              ['CREATE', 'DROP', 'ALTER', 'INSERT', 'UPDATE', 'DELETE']):
                        try:
                            plan_sql = f"EXPLAIN (FORMAT JSON) {stmt}"
                            return self.fetch_results(plan_sql)
                        except Exception as e:
                            print(f"获取语句计划失败: {e}")
                            continue
                return None
        else:
            # 单个语句，直接获取计划
            try:
                plan_sql = f"EXPLAIN (FORMAT JSON) {sql}"
                return self.fetch_results(plan_sql)
            except Exception as e:
                print(f"获取执行计划失败: {e}")
                return None
    
    def get_metrics_specification(self):
        return {
            "pg_stat_database": [
                "tup_returned", "blks_hit", "blks_read", "tup_fetched"
            ],
            "pg_stat_user_tables": [
                "idx_tup_fetch", "seq_tup_read", "seq_scan", "idx_scan"
            ],
            "pg_statio_user_tables": [
                "heap_blks_hit", "heap_blks_read", "idx_blks_hit", "idx_blks_read"
            ],
        }
    
    def collect_metrics(self, conn):
        data = {}
        METRICS_SPECIFICATION = self.get_metrics_specification()
        with conn.cursor(row_factory=dict_row) as cursor:
            for table, keys in METRICS_SPECIFICATION.items():
                try:
                    # 只采集当前数据库
                    if table == "pg_stat_database":
                        cursor.execute(f"SELECT * FROM {table} WHERE datname = current_database()")
                    else:
                        cursor.execute(f"SELECT * FROM {table}")
                    records = cursor.fetchall()
                    # 只保留关心的字段
                    filtered = []
                    for r in records:
                        filtered.append({k: r[k] for k in keys if k in r})
                    data[table] = filtered
                except Exception as e:
                    data[table] = f"Error: {e}"
        return data

    def get_log(self, sql=None):
        # 还得修改
        """
        采集SQL执行前后的PostgreSQL内部指标，并返回差值（后-前）。
        如果sql为None，则只采集一次。
        """
        conn = self.connect()

        metrics_before = self.collect_metrics(conn)
        if sql is not None:
            _, _, execution_time, _ = self.execute(sql)
        metrics_after = self.collect_metrics(conn)
        # 计算差值
        diff = self.get_log_diff(metrics_before, metrics_after)
        log_list = self.get_metrics_list(diff, execution_time)

        return log_list

    def get_metrics_list(self, diff, execution_time):
        metrics_list = []
        # 1. pg_stat_database
        db_metrics = diff.get("pg_stat_database", [{}])
        metrics_list.append(db_metrics[0].get("tup_returned", 0))
        metrics_list.append(db_metrics[0].get("blks_hit", 0))
        metrics_list.append(db_metrics[0].get("blks_read", 0))
        metrics_list.append(db_metrics[0].get("tup_fetched", 0))
        # 2. pg_stat_user_tables
        user_table_metrics = diff.get("pg_stat_user_tables", {})
        metrics_list.append(user_table_metrics.get("idx_tup_fetch", 0))
        metrics_list.append(user_table_metrics.get("seq_tup_read", 0))
        metrics_list.append(user_table_metrics.get("seq_scan", 0))
        metrics_list.append(user_table_metrics.get("idx_scan", 0))
        # 3. pg_statio_user_tables
        statio_metrics = diff.get("pg_statio_user_tables", {})
        metrics_list.append(statio_metrics.get("heap_blks_hit", 0))
        metrics_list.append(statio_metrics.get("heap_blks_read", 0))
        metrics_list.append(statio_metrics.get("idx_blks_hit", 0))
        metrics_list.append(statio_metrics.get("idx_blks_read", 0))
        # 4. execution_time
        metrics_list.append(execution_time)
        return metrics_list
    
    def get_log_diff(self, log_before, log_after):
        diff = {}
        for table in log_before:
            before = log_before.get(table, [])
            after = log_after.get(table, [])
            if isinstance(before, str) or isinstance(after, str):
                diff[table] = before if isinstance(before, str) else after
                continue
            if table in ["pg_stat_user_tables", "pg_statio_user_tables"]:
                # 对所有表的指标做加和
                METRICS_SPECIFICATION = self.get_metrics_specification()
                sum_row = {k: 0 for k in METRICS_SPECIFICATION[table]}
                for i in range(min(len(before), len(after))):
                    row_before = before[i]
                    row_after = after[i]
                    for k in sum_row.keys():
                        v_before = row_before.get(k, 0)
                        v_after = row_after.get(k, 0)
                        try:
                            sum_row[k] += (v_after if v_after is not None else 0) - (v_before if v_before is not None else 0)
                        except Exception:
                            pass
                diff[table] = sum_row
            else:
                # 其他表（如pg_stat_database）仍然返回单条
                table_diff = []
                for i in range(min(len(before), len(after))):
                    row_before = before[i]
                    row_after = after[i]
                    row_diff = {}
                    for k in before[i]:
                        v_before = row_before.get(k, 0)
                        v_after = row_after.get(k, 0)
                        try:
                            row_diff[k] = (v_after if v_after is not None else 0) - (v_before if v_before is not None else 0)
                        except Exception:
                            row_diff[k] = None
                    for id_key in ['relname', 'datname', 'table_name']:
                        if id_key in row_before:
                            row_diff[id_key] = row_before[id_key]
                    table_diff.append(row_diff)
                diff[table] = table_diff
        return diff
    
    def run_sql_and_collect_all(self, sql, interval=1, warmup=0, duration=300):
        conn = self.connect()

        # 1. 采集SQL执行前的log
        log_before = self.collect_metrics(conn)
        # 2. 获取执行计划
        plan = self.get_plan(sql)
        # 3. 启动资源监控
        pid = self.get_pid()
        monitor = ResourceMonitor(pid, interval, warmup, duration)
        monitor.run()
        # 4. 执行SQL
        _, _, execution_time, _ = self.execute(sql, timeout=duration)
        # 5. 关闭资源监控
        monitor.terminate()
        cpu_usage, read_io, write_io, virtual_memory, physical_memory, net_recv, net_sent = monitor.get_monitor_data()
        metrics = [cpu_usage, read_io, write_io, virtual_memory, physical_memory, net_recv, net_sent]

        # 6. 采集SQL执行后的log
        log_after = self.collect_metrics(conn)

        # 7. 计算log差值
        diff = self.get_log_diff(log_before, log_after)
        log_list = self.get_metrics_list(diff, execution_time)

        # 8. 返回所有结果
        return {
            "query": sql,
            "plan_json": plan,
            "internal_metrics": log_list,
            "external_metrics": metrics,
            "duration": execution_time
        }

    def decimal_default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        raise TypeError

    def compare_sql_results(self, sql1, sql2):
        """
        判断两个SQL语句的输出结果是否完全一致（包括内容和重复行）。
        返回True表示完全一致，False表示不一致。
        """
        try:
            # 清理SQL，去除注释、多余空白和结尾分号
            def clean_sql(sql):
                # 去除注释
                lines = sql.split('\n')
                cleaned_lines = []
                for line in lines:
                    # 去除行内注释
                    if '--' in line:
                        line = line.split('--')[0]
                    cleaned_lines.append(line.strip())
                
                # 合并并清理
                sql = ' '.join(cleaned_lines)
                # 去除多余空白
                sql = ' '.join(sql.split())
                # 去除结尾分号
                sql = sql.rstrip(';')
                return sql
            
            sql1 = clean_sql(sql1)
            sql2 = clean_sql(sql2)
            
            conn = self.connect()
            
            # 构建EXCEPT查询，确保SQL语法正确
            except1 = f"SELECT * FROM ({sql1}) AS t1 EXCEPT ALL SELECT * FROM ({sql2}) AS t2"
            except2 = f"SELECT * FROM ({sql2}) AS t2 EXCEPT ALL SELECT * FROM ({sql1}) AS t1"
            
            with conn.cursor() as cursor:
                cursor.execute(except1)
                diff1 = cursor.fetchall()
                cursor.execute(except2)
                diff2 = cursor.fetchall()
            
            return len(diff1) == 0 and len(diff2) == 0
            
        except Exception as e:
            print(f"比较SQL结果时出错: {e}")
            print(f"SQL1: {sql1}")
            print(f"SQL2: {sql2}")
            # 如果比较失败，返回False表示结果不一致
            return False


