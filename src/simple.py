#!/usr/bin/env python3
"""
Simplified DBAgent that skips the diagnosis stage and
uses static case retrieval plus direct LLM action generation.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime


def setup_logging():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"simple_diagnosis_{timestamp}.log")

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    print(f"Log file saved to: {log_file}")
    return log_file


setup_logging()

from agent.prompt import get_workload_best_sql_times, normalize_workload_name
from agent.simple_db_agent import SimpleDBAgent


def detect_workload_name(*candidates):
    for candidate in candidates:
        workload_name = normalize_workload_name(candidate)
        if workload_name:
            return workload_name
    return None


def parse_bool(value):
    if isinstance(value, bool):
        return value

    normalized = str(value or "").strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError("booster must be true or false")


def parse_args():
    parser = argparse.ArgumentParser(description="Simplified Database Query Optimization Tool")
    parser.add_argument("--data_path", type=str, required=True, help="Path to slow query data files")
    parser.add_argument("--order", type=str, required=True, help="Query execution order file")
    parser.add_argument("--duration", type=int, default=1, help="Optimization duration in hours")
    parser.add_argument("--config", type=str, required=True, help="Configuration file path")
    parser.add_argument(
        "--workload",
        type=str,
        help="Workload name: TPCH, JOB, or TPCDS",
    )
    parser.add_argument(
        "--booster",
        type=parse_bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable hardcoded workload best_sql_times. Default: false",
    )
    parser.add_argument(
        "--no_improvement_threshold",
        type=int,
        default=3,
        help="Reserved compatibility option; currently unused by simple agent",
    )

    return parser.parse_args()


async def main():
    """
    Example usage:
    python simple.py --data_path /root/DREAM/data/slow_queries/TPC-H --order qorder.txt --duration 1 --config /root/DREAM/config/base_config.json
    """
    logger = logging.getLogger(__name__)

    args = parse_args()
    logger.info("Starting Simplified Database Optimization System")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Execution order: {args.order}")
    logger.info(f"Optimization duration: {args.duration} hours")
    logger.info(f"Configuration file: {args.config}")
    logger.info(f"Workload argument: {args.workload or 'auto'}")
    logger.info(f"Booster enabled: {args.booster}")
    logger.info(
        f"No improvement threshold: {args.no_improvement_threshold} (currently unused by simple agent)"
    )

    try:
        with open(args.config, "r", encoding="utf-8") as f:
            configs = json.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file {args.config} not found.")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing configuration file: {e}")
        return

    specified_workload = normalize_workload_name(args.workload)
    if args.workload and not specified_workload:
        logger.error("Unsupported workload %s. Use TPCH, JOB, or TPCDS.", args.workload)
        return

    workload_name = specified_workload or detect_workload_name(args.data_path, args.config)
    if workload_name:
        logger.info("Resolved workload: %s", workload_name)
    else:
        logger.warning("Could not infer workload type from arguments/data path/config")

    if args.booster:
        workload_best_times = get_workload_best_sql_times(workload_name, booster=True)
        if workload_best_times:
            configs["WORKLOAD_QUERY_BEST_TIMES"] = workload_best_times
            configs["WORKLOAD_QUERY_BEST_TIME_NAME"] = workload_name
            logger.info(
                "Booster enabled for workload %s with %s best_sql_times entries",
                workload_name,
                len(workload_best_times),
            )
            logger.info("Booster workload name: %s", workload_name)
            logger.info("Booster best_sql_times: %s", workload_best_times)
            logger.info(
                "Booster total target time: %.4fs",
                sum(workload_best_times.values()),
            )
        else:
            logger.warning(
                "Booster requested but no best_sql_times were loaded for workload %s",
                workload_name or "unknown",
            )
    else:
        logger.info("Booster disabled; simple agent will start workload targets from 0.0")

    async with SimpleDBAgent(configs) as agent:
        logger.info("Starting simplified database optimization...")
        await agent.run(
            slow_query_path=args.data_path,
            order=args.order,
            duration=args.duration,
            no_improvement_threshold=args.no_improvement_threshold,
        )
        logger.info("Simplified database optimization completed!")


# cd /root/DREAM/src
# python simple.py --data_path /root/DREAM/data/slow_queries/TPC-H --order qorder.txt --duration 15 --config /root/DREAM/config/tpch_config.json --workload TPCH --booster true
# python simple.py --data_path /root/DREAM/data/slow_queries/JOB --order qorder.txt --duration 15 --config /root/DREAM/config/job_config.json --workload JOB --booster true
# python simple.py --data_path /root/DREAM/data/slow_queries/TPC-DS --order qorder.txt --duration 15 --config /root/DREAM/config/tpcds_config.json --workload TPCDS --booster true

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    except Exception as e:
        print(f"Error during optimization: {e}")
        sys.exit(1)
