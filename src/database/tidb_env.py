import json
import os
import re
import socket
import ssl
import subprocess
import time
from decimal import Decimal
from pathlib import Path

import psutil
import pymysql


class TiDB:
    def __init__(self, db_config):
        """
        Initialize TiDB connector

        Args:
            host: TiDB server address
            port: Port number
            user: Username
            password: Password
            database: Database name
            db_type: Database type
            ssl_mode: SSL mode (DISABLED, REQUIRED, VERIFY_IDENTITY)
            ssl_ca: SSL certificate path
        """
        self.host = db_config["host"]
        self.port = db_config["port"]
        self.user = db_config["user"]
        self.password = db_config["password"]
        self.database = db_config["dbname"]
        self.query_timeout = int(db_config.get("query_timeout", 300))
        self.ssl_mode = db_config.get("ssl_mode", "VERIFY_IDENTITY")
        self.ssl_ca = db_config.get("ssl_ca", "/etc/ssl/certs/ca-certificates.crt")
        self.db_type = db_config.get("db_type", "tidb")
        # Keep the configured baseline independent of TiDB's global state.
        # Previous tuning runs may have left global/session variables changed;
        # every DREAM connection therefore reapplies the experiment defaults.
        self.knob_config = db_config.get("KNOB_CONFIG", {}) or {}
        # DREAM's TiDB mode intentionally avoids the optional SQLSolver/QED
        # equivalence check.  Candidate rewrites are still executed with the
        # configured timeout below, which is the authoritative acceptance
        # test for this experiment.
        self.skip_equivalence_validation = bool(
            db_config.get("skip_equivalence_validation", True)
        )
        self.query_mem_quota_bytes = int(
            db_config.get("query_mem_quota_bytes", 1024 * 1024 * 1024)
        )
        self.cgroup_memory_limit_bytes = self._read_cgroup_value("memory.max")
        self.cgroup_memory_guard_bytes = int(
            db_config.get("cgroup_memory_guard_bytes", 90 * 1024**3)
        )
        self.connection = None

    @staticmethod
    def _sql_literal(value):
        if isinstance(value, bool):
            return "ON" if value else "OFF"
        if isinstance(value, (int, float)):
            return str(value)
        text = str(value)
        if text.lower() in {"on", "off"}:
            return text.upper()
        return "'" + text.replace("'", "''") + "'"

    @staticmethod
    def _read_cgroup_value(name):
        try:
            value = Path("/sys/fs/cgroup", name).read_text().strip()
            return None if value == "max" else int(value)
        except (OSError, ValueError):
            return None

    def memory_current_bytes(self):
        return self._read_cgroup_value("memory.current") or 0

    def memory_current_gib(self):
        return self.memory_current_bytes() / (1024**3)

    @staticmethod
    def _read_cgroup_stat():
        values = {}
        try:
            for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines():
                key, value = line.split(None, 1)
                values[key] = int(value)
        except (OSError, ValueError):
            return {}
        return values

    def memory_nonreclaimable_bytes(self):
        """Approximate memory that cannot be reclaimed as file cache."""
        stats = self._read_cgroup_stat()
        if not stats:
            return 0
        return sum(
            stats.get(key, 0)
            for key in (
                "anon",
                "kernel",
                "slab_unreclaimable",
                "unevictable",
                "file_dirty",
                "file_writeback",
            )
        )

    def memory_nonreclaimable_gib(self):
        return self.memory_nonreclaimable_bytes() / (1024**3)

    def _memory_safe_for_ddl(self):
        current = self.memory_current_bytes()
        nonreclaimable = self.memory_nonreclaimable_bytes()
        # memory.current includes a large, reclaimable TiKV data-file cache.
        # Guard both dimensions: configured non-reclaimable budget and a
        # small headroom below the actual cgroup hard limit.
        if nonreclaimable and nonreclaimable >= self.cgroup_memory_guard_bytes:
            return False
        if current and self.cgroup_memory_limit_bytes:
            headroom_limit = self.cgroup_memory_limit_bytes - 4 * 1024**3
            if headroom_limit > 0 and current >= headroom_limit:
                return False
        return True

    @staticmethod
    def _split_statements(sql):
        """Split DREAM action batches without splitting quoted semicolons."""
        statements, start, quote, escaped = [], 0, None, False
        for i, ch in enumerate(sql):
            if quote:
                if escaped:
                    escaped = False
                elif ch == "\\" and quote != "`":
                    escaped = True
                elif ch == quote:
                    quote = None
                continue
            if ch in ("'", '"', '`'):
                quote = ch
            elif ch == ";":
                statement = sql[start:i].strip()
                if statement:
                    statements.append(statement)
                start = i + 1
        tail = sql[start:].strip()
        if tail:
            statements.append(tail)
        return statements

    def _check_tidb_running(self):
        """Check if TiDB is running on the specified port"""
        try:
            # Try to connect to the port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((self.host, self.port))
            sock.close()
            return result == 0
        except Exception:
            return False

    def _check_tiup_playground_running(self):
        """Check if tiup playground process is running"""
        try:
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    cmdline = proc.info["cmdline"]
                    if cmdline and "tiup" in cmdline and "playground" in cmdline and "DREAM" in cmdline:
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return False
        except Exception:
            return False

    def _start_tiup_playground(self):
        """Start tiup playground"""
        try:
            print("Starting tiup playground --tag DREAM...")
            # use subprocess to start tiup playground, set to background
            process = subprocess.Popen(
                ["tiup", "playground", "--tag", "DREAM"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,  # create new process group
            )

            # wait for service to start
            time.sleep(10)

            # check if successfully started
            if self._check_tidb_running():
                print("tiup playground started successfully!")
                return True
            else:
                print("tiup playground startup failed or timed out")
                return False

        except Exception as e:
            print(f"Error starting tiup playground: {e}")
            return False

    def _ensure_tidb_running(self):
        # ensure TiDB is running, if not, start it
        if not self._check_tidb_running():
            print("TiDB is not running, checking tiup playground status...")

            if not self._check_tiup_playground_running():
                print("tiup playground process not found, starting...")
                if not self._start_tiup_playground():
                    raise Exception("Unable to start TiDB service")
            else:
                print("Found tiup playground process, waiting for service to start...")
                max_wait = 30
                wait_time = 0
                while not self._check_tidb_running() and wait_time < max_wait:
                    time.sleep(2)
                    wait_time += 2

                if not self._check_tidb_running():
                    raise Exception("TiDB service startup timeout")

        print("TiDB service is ready")

    def _get_ssl_config(self):
        # get SSL config
        if self.ssl_mode == "VERIFY_IDENTITY":
            return {
                "ssl": {
                    "ca": self.ssl_ca,
                    "check_hostname": True,
                    "verify_mode": ssl.CERT_REQUIRED,
                }
            }
        elif self.ssl_mode == "REQUIRED":
            return {
                "ssl": {
                    "ca": self.ssl_ca,
                    "check_hostname": False,
                    "verify_mode": ssl.CERT_REQUIRED,
                }
            }
        elif self.ssl_mode == "DISABLED":
            return {}
        else:
            return {}

    def connect(self):
        # ensure TiDB service is running
        self._ensure_tidb_running()

        if self.connection is None or not self.connection.open:
            ssl_config = self._get_ssl_config()
            self.connection = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset="utf8mb4",
                cursorclass=pymysql.cursors.DictCursor,
                **ssl_config,
            )
            self._apply_configured_session_defaults()
        return self.connection

    def _apply_configured_session_defaults(self):
        """Apply the experiment baseline to a newly opened TiDB session.

        TiDB exposes some knobs only on particular versions and some knobs
        are global-only.  Unsupported session assignments are intentionally
        ignored, matching the warmup script, while supported assignments are
        explicit so a prior run cannot silently contaminate the baseline.
        """
        if not self.knob_config:
            return
        conn = self.connection
        try:
            with conn.cursor() as cursor:
                cursor.execute("SHOW VARIABLES")
                variables = {
                    str(row.get("Variable_name") or row.get("VARIABLE_NAME")).lower()
                    for row in cursor.fetchall()
                }
                cursor.execute(
                    f"SET SESSION tidb_mem_quota_query={int(self.query_mem_quota_bytes)}"
                )
                for group in ("query_knobs", "system_knobs"):
                    for name, spec in (self.knob_config.get(group) or {}).items():
                        if name.lower() not in variables:
                            continue
                        try:
                            cursor.execute(
                                f"SET SESSION `{name}`={self._sql_literal(spec.get('default'))}"
                            )
                        except Exception:
                            conn.rollback()
                cursor.execute(f"SET SESSION MAX_EXECUTION_TIME={int(self.query_timeout * 1000)}")
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass

    def execute(self, sql, conn=None, timeout=None):
        if conn is None:
            conn = self.connect()

        timeout = int(timeout or self.query_timeout or 300)
        statements = self._split_statements(sql)
        if not statements:
            return True, [], 0.0, ""

        # Do not allow a new physical DDL to start when the cgroup is already
        # close to its 100-GiB limit.  Existing queries are allowed to finish.
        if any(re.match(r"(?is)^\s*(CREATE|ALTER)\s+INDEX", stmt) for stmt in statements):
            if not self._memory_safe_for_ddl():
                return False, None, 0.0, (
                    f"DDL paused by cgroup memory guard: "
                    f"{self.memory_current_gib():.2f} GiB"
                )

        try:
            start_time = time.time()
            with conn.cursor() as cursor:
                results = []
                for statement in statements:
                    # TiDB's MAX_EXECUTION_TIME applies to SELECT-like
                    # statements; the per-query memory quota protects both
                    # reads and aggregation operators.
                    cursor.execute(
                        f"SET SESSION MAX_EXECUTION_TIME={timeout * 1000}"
                    )
                    cursor.execute(
                        f"SET SESSION tidb_mem_quota_query={self.query_mem_quota_bytes}"
                    )
                    cursor.execute(statement)
                    if cursor.description is not None:
                        results = cursor.fetchall()
                conn.commit()
            end_time = time.time()
            return True, results, end_time - start_time, ""
        except pymysql.Error as e:
            try:
                conn.rollback()
            except Exception:
                pass
            return False, None, min(timeout, max(time.time() - start_time, 0.0)), str(e)
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            return False, None, min(timeout, max(time.time() - start_time, 0.0)), str(e)

    def fetch_results(self, sql, as_json=False, timeout=None):
        conn = self.connect()
        with conn.cursor() as cursor:
            if timeout:
                cursor.execute(f"SET SESSION MAX_EXECUTION_TIME={int(timeout * 1000)}")
            cursor.execute(sql)
            results = cursor.fetchall()

            if as_json:
                return json.dumps(results, indent=4, ensure_ascii=False, default=self._decimal_default)

            # if only one column, return single column result
            if results and len(results[0]) == 1:
                return [list(row.values())[0] for row in results]

            return results

    def get_tables(self):
        sql = """
        SELECT table_name FROM information_schema.tables 
        WHERE table_schema = %s AND table_type = 'BASE TABLE'
        """
        conn = self.connect()
        with conn.cursor() as cursor:
            cursor.execute(sql, (self.database,))
            results = cursor.fetchall()
            return [row["table_name"] for row in results]

    def get_views(self):
        sql = """
        SELECT table_name as viewname, view_definition as definition 
        FROM information_schema.views 
        WHERE table_schema = %s
        """
        conn = self.connect()
        with conn.cursor() as cursor:
            cursor.execute(sql, (self.database,))
            results = cursor.fetchall()
            return [(row["viewname"], row["definition"]) for row in results]

    def get_columns(self):
        sql = """
        SELECT column_name FROM information_schema.columns 
        WHERE table_schema = %s
        """
        conn = self.connect()
        with conn.cursor() as cursor:
            cursor.execute(sql, (self.database,))
            results = cursor.fetchall()
            return [row["column_name"] for row in results]

    def get_indexes(self):
        """Return TiDB secondary/primary indexes as prompt-friendly names."""
        sql = """
        SELECT table_name, index_name, non_unique, GROUP_CONCAT(column_name
               ORDER BY seq_in_index SEPARATOR ',') AS columns
        FROM information_schema.statistics
        WHERE table_schema = %s
        GROUP BY table_name, index_name, non_unique
        ORDER BY table_name, index_name
        """
        rows = self.fetch_results(sql % ("'" + self.database.replace("'", "''") + "'"))
        return [
            {
                "table_name": row.get("table_name"),
                "index_name": row.get("index_name"),
                "non_unique": row.get("non_unique"),
                "columns": row.get("columns"),
            }
            for row in rows
        ]

    def get_current_knob_values(self, knob_config):
        values = {"system_knobs": {}, "query_knobs": {}}
        names = []
        for group in ("system_knobs", "query_knobs"):
            names.extend((group, name) for name in (knob_config.get(group) or {}).keys())
        if not names:
            return values
        variables = [name for _, name in names]
        rows = self.fetch_results("SHOW VARIABLES", as_json=False)
        by_name = {}
        wanted = {name.lower() for name in variables}
        for row in rows:
            name = row.get("Variable_name") or row.get("VARIABLE_NAME") or row.get("variable_name")
            value = row.get("Value") if "Value" in row else row.get("VARIABLE_VALUE")
            if name and str(name).lower() in wanted:
                by_name[str(name).lower()] = value
        for group, name in names:
            values[group][name] = by_name.get(name.lower())
        return values

    def fetch_schema_info(self):
        sql = """
        SELECT 
            c.table_name,
            c.column_name,
            c.data_type,
            c.is_nullable,
            c.column_default,
            c.column_key,
            c.extra
        FROM information_schema.columns c
        WHERE c.table_schema = %s
        ORDER BY c.table_name, c.ordinal_position
        """

        conn = self.connect()
        with conn.cursor() as cursor:
            cursor.execute(sql, (self.database,))
            rows = cursor.fetchall()

        schema_dict = {}
        for row in rows:
            table_name = row["table_name"]
            if table_name not in schema_dict:
                schema_dict[table_name] = {"columns": []}

            schema_dict[table_name]["columns"].append(
                {
                    "column_name": row["column_name"],
                    "data_type": row["data_type"],
                    "is_nullable": row["is_nullable"],
                    "column_default": row["column_default"],
                    "column_key": row["column_key"],
                    "extra": row["extra"],
                }
            )

        return schema_dict

    def get_db_size(self):
        sql = """
        SELECT 
            ROUND(SUM(data_length + index_length) / 1024 / 1024, 2) as size_mb
        FROM information_schema.tables 
        WHERE table_schema = %s
        """
        conn = self.connect()
        with conn.cursor() as cursor:
            cursor.execute(sql, (self.database,))
            result = cursor.fetchone()
            return float(result["size_mb"]) if result["size_mb"] else 0.0

    def get_size(self):
        """PostgreSQL-compatible size shape used by DREAM's planner."""
        return [f"{self.get_db_size():.2f} MB"]

    def get_plan(self, sql):
        statement = sql.strip().rstrip(";")
        try:
            rows = self.fetch_results("EXPLAIN " + statement, timeout=self.query_timeout)
            return self._explain_rows_to_plan_tree(rows)
        except Exception as exc:
            return [{"explain_error": str(exc)}]

    @staticmethod
    def _explain_rows_to_plan_tree(rows):
        """Convert TiDB's tabular EXPLAIN into the tree shape used by RCRank.

        TiDB exposes operator rows such as ``└─TableReader_7`` while RCRank's
        PlanEncoder expects PostgreSQL-like ``Node Type``/``Plans`` nodes.  The
        conversion keeps the original operator information in ``Filter`` so
        the LLM still receives useful details while the small model gets a
        structurally valid tree.
        """
        if not isinstance(rows, list) or not rows:
            return rows
        if isinstance(rows[0], dict) and "Node Type" in rows[0]:
            return rows

        def value(row, *names, default=""):
            lowered = {str(k).lower(): v for k, v in row.items()}
            for name in names:
                if name.lower() in lowered:
                    return lowered[name.lower()]
            return default

        def depth_and_operator(raw_id):
            text = str(raw_id or "")
            depth = sum(text.count(ch) for ch in ("└", "├", "┴", "┬"))
            operator = text
            operator = operator.lstrip(" └├┴┬│─")
            return depth, operator or "Unknown"

        nodes = []
        for row in rows:
            depth, operator = depth_and_operator(value(row, "id"))
            info = str(value(row, "operator info", "operator_info", default=""))
            est_rows = value(row, "estRows", "est_rows", default=0)
            try:
                est_rows = float(est_rows)
            except (TypeError, ValueError):
                est_rows = 0.0
            table_match = re.search(r"(?:table|partition):\s*([\w.$-]+)", info, re.I)
            node = {
                "Node Type": re.sub(r"_[0-9]+$", "", operator),
                "Startup Cost": 0.0,
                "Total Cost": 0.0,
                "Plan Rows": est_rows,
                "Plan Width": 0,
            }
            if table_match:
                node["Relation Name"] = table_match.group(1)
            if info:
                node["Filter"] = info
            nodes.append((depth, node))

        if not nodes:
            return rows
        root = nodes[0][1]
        stack = [(nodes[0][0], root)]
        for depth, node in nodes[1:]:
            while stack and stack[-1][0] >= depth:
                stack.pop()
            parent = stack[-1][1] if stack else root
            parent.setdefault("Plans", []).append(node)
            stack.append((depth, node))
        return root

    def get_pid(self):
        """Return the current TiDB session thread id for diagnostics."""
        row = self.fetch_results("SELECT CONNECTION_ID() AS connection_id")
        return row[0].get("connection_id") if row else 0

    def _empty_external_metrics(self):
        # Keep the shape expected by QueryInfo/planner (7 series x 9 points).
        current = round(self.memory_current_gib(), 3)
        # Preserve DREAM's historical external-metrics shape/ordering; the
        # non-reclaimable figure is used only by the DDL guard above.
        return [[0.0] * 9 for _ in range(4)] + [[current] * 9] + [[0.0] * 9 for _ in range(2)]

    def _internal_metrics(self, duration):
        # TiDB does not expose PostgreSQL's pg_stat counters.  Preserve the
        # vector contract and provide execution time in the final position.
        return [0] * 12 + [round(float(duration), 3)]

    def run_sql_and_collect_all(self, sql, interval=1, warmup=0, timeout=None):
        timeout = int(timeout or self.query_timeout or 300)
        plan = self.get_plan(sql)
        success, _, duration, error = self.execute(sql, timeout=timeout)
        if not success and duration <= 0:
            duration = float(timeout)
        if not success and error:
            plan = plan or [{"execution_error": error}]
        return {
            "query": sql,
            "plan_json": plan,
            "internal_metrics": self._internal_metrics(duration),
            "external_metrics": self._empty_external_metrics(),
            "duration": round(float(duration), 3),
        }

    def rollback_tuning(self, index_names, knob_names):
        """Rollback only DREAM-created indexes and session knob changes."""
        conn = self.connect()
        with conn.cursor() as cur:
            for index_name in index_names or []:
                row = None
                cur.execute(
                    "SELECT DISTINCT TABLE_NAME FROM information_schema.statistics "
                    "WHERE TABLE_SCHEMA=%s AND INDEX_NAME=%s",
                    (self.database, index_name),
                )
                row = cur.fetchone()
                if not row:
                    continue
                table_name = row.get("TABLE_NAME") or row.get("table_name")
                try:
                    cur.execute(f"DROP INDEX IF EXISTS `{index_name}` ON `{table_name}`")
                except Exception:
                    conn.rollback()
            for knob_name in knob_names or []:
                try:
                    default = None
                    for group in ("query_knobs", "system_knobs"):
                        spec = (self.knob_config.get(group) or {}).get(knob_name)
                        if spec is not None:
                            default = spec.get("default")
                            break
                    if default is None:
                        cur.execute(f"SET SESSION `{knob_name}`=DEFAULT")
                    else:
                        cur.execute(
                            f"SET SESSION `{knob_name}`={self._sql_literal(default)}"
                        )
                except Exception:
                    conn.rollback()
            conn.commit()

    def check_sql_equivalence(self, sql1, sql2):
        # Explicitly disabled for the OSSInsight DREAM run; SQL execution in
        # evaluate_action remains the acceptance gate.
        return True

    def compare_sql_results(self, sql1, sql2):
        sql1 = sql1.strip().rstrip(";")
        sql2 = sql2.strip().rstrip(";")

        conn = self.connect()
        with conn.cursor() as cursor:
            try:
                cursor.execute(sql1)
                results1 = cursor.fetchall()

                cursor.execute(sql2)
                results2 = cursor.fetchall()

                if len(results1) != len(results2):
                    return False

                def normalize_results(results):
                    normalized = []
                    for row in results:
                        if isinstance(row, dict):
                            normalized.append(tuple(sorted(row.items())))
                        else:
                            normalized.append(tuple(row))
                    return sorted(normalized)

                normalized1 = normalize_results(results1)
                normalized2 = normalize_results(results2)

                return normalized1 == normalized2

            except Exception as e:
                print(f"Error comparing SQL results: {e}")
                return False

    def close(self):
        if self.connection and self.connection.open:
            self.connection.close()

    def _decimal_default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
