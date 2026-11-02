import re

from database.pg_env import PostgresDB
from database.tidb_env import TiDB


def base_information_collect(query_info, database_config):
    db_type = str(database_config.get("db_type", "postgres")).lower()
    database = TiDB(database_config) if db_type == "tidb" else PostgresDB(database_config)
    dbname = (
        getattr(database, "database", None)
        or getattr(database, "dbname", None)
        or database_config.get("dbname")
        or "unknown"
    )

    size = database.get_db_size() if db_type == "tidb" else database.get_size()
    if isinstance(size, list) and len(size) == 1:
        size = size[0]
    size = size or "unknown"

    db_info = {
        "db_type": getattr(database, "db_type", db_type),
        "workload_type": database_config.get("workload_type", "OLAP"),
        "dbname": dbname,
        "size": size,
        "schema": dbname,
    }

    base_info = {
        "query_info": query_info,
        "database_info": db_info,
    }

    return base_info


def extract_index_names(sql):
    return re.findall(r"CREATE\s+INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?([\w_]+)", sql, re.IGNORECASE)


def extract_knob_names(sql):
    return re.findall(r"SET\s+([a-zA-Z0-9_]+)\s*=", sql, re.IGNORECASE)


def parse_query_rewrite(fix_action):
    lines = fix_action.strip().splitlines()
    sql_start = None

    for idx, line in enumerate(lines):
        l = line.lstrip().upper()
        if l.startswith("WITH") or l.startswith("SELECT") or l.startswith("INSERT") or l.startswith("UPDATE") or l.startswith("DELETE"):
            sql_start = idx
            break

    if sql_start is not None:
        fix_part = "\n".join([line for line in lines[:sql_start] if line.strip()])
        sql_part = "\n".join(lines[sql_start:]).strip()
        return fix_part, sql_part
    return fix_action.strip(), ""
