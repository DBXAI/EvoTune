import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from agent.db_agent import DBAgent
from config_paths import resolve_config_paths

def setup_logging():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.environ.get("DREAM_LOG_DIR", os.path.join(project_root, "logs"))
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.environ.get(
        "DREAM_LOG_FILE",
        os.path.join(log_dir, f"dream_diagnosis_{timestamp}.log"),
    )
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)

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

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    print(f"Log file saved to: {log_file}")
    return log_file

setup_logging()

def parse_args():
    parser = argparse.ArgumentParser(description="Slow Query Diagnosis Tool")
    parser.add_argument("--data_path", type=str, required=True, help="Slow query data file path")
    parser.add_argument("--order", type=str, required=True, help="Slow query execution order")
    parser.add_argument("--duration", type=float, default=30, help="Optimization duration in hours")
    parser.add_argument("--max-rounds", type=int, default=None, help="Stop after this many completed tuning rounds")
    parser.add_argument("--config", type=str, required=True, help="Configuration file path")
    parser.add_argument(
        "--resume-state",
        type=str,
        default=None,
        help="Optional JSON checkpoint containing best per-query state for continuation",
    )

    return parser.parse_args()


def restore_online_predictor_config(configs, resume_state_path):
    """Use the latest online predictor when continuing the same experiment."""
    if not resume_state_path or not os.path.isfile(resume_state_path):
        return False
    output_dir = configs.get("MEMORY_MANAGER_CONFIG", {}).get("model_output_dir")
    if not output_dir:
        return False
    checkpoint = os.path.join(output_dir, "best_model.pt")
    if not os.path.isfile(checkpoint):
        return False
    planner_config = configs.setdefault("PLANNER_CONFIG", {})
    planner_config["model_path"] = checkpoint
    feedback_csv = os.path.join(output_dir, "memory_cases.csv")
    if os.path.isfile(feedback_csv):
        planner_config["train_data_path"] = feedback_csv
    return True


async def main():
    # cd /root/DREAM/src
    # python main.py --data_path /root/DREAM/data/slow_queries/TPC-H --order qorder.txt --duration 10
    # python main.py --data_path /root/DREAM/data/slow_queries/TPC-H --order qorder.txt --duration 1 --config /root/DREAM/config/tpch_config.json
    # python main.py --data_path /root/DREAM/data/slow_queries/TPC-DS --order qorder.txt --duration 1 --config /root/DREAM/config/tpcds_config.json
    # python main.py --data_path /root/DREAM/data/slow_queries/JOB --order qorder.txt --duration 1 --config /root/DREAM/config/job_config.json
    logger = logging.getLogger(__name__)

    args = parse_args()
    logger.info("Starting DREAM Database Diagnosis System")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Execution order: {args.order}")
    logger.info(f"Optimization duration: {args.duration} hours")
    logger.info(f"Configuration file: {args.config}")

    with open(args.config, "r", encoding="utf-8") as f:
        configs = json.load(f)
    resolve_config_paths(configs, args.config)

    if restore_online_predictor_config(configs, args.resume_state):
        logger.info(
            "Resuming online predictor from %s",
            configs["PLANNER_CONFIG"]["model_path"],
        )

    async with DBAgent(configs=configs) as agent:
        await agent.run(
            slow_query_path=args.data_path,
            order=args.order,
            duration=args.duration,
            max_rounds=args.max_rounds,
            no_improvement_threshold=configs["AGENT_CONFIG"]["no_improvement_threshold"],
            resume_state_path=args.resume_state,
        )

if __name__ == "__main__":
    asyncio.run(main())
