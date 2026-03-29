import re

from database.pg_env import PostgresDB


def base_information_collect(query_info, database_config):
    postgres_db = PostgresDB(database_config)
    dbname = getattr(postgres_db, "dbname", None) or database_config.get("dbname") or "unknown"

    size = postgres_db.get_size()
    if isinstance(size, list) and len(size) == 1:
        size = size[0]
    size = size or "unknown"

    db_info = {
        "db_type": postgres_db.db_type,
        "workload_type": postgres_db.workload_type,
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
