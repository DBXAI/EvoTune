import asyncio
import argparse
import sys
from pathlib import Path
import logging
import os
from datetime import datetime

# add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# setup logging, output to console and file
def setup_logging():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"dream_diagnosis_{timestamp}.log")
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    print(f"日志文件保存位置: {log_file}")
    return log_file

setup_logging()

from config.base_config import POSTGRES_CONFIG, KNOB_CONFIG, API_SETTINGS
from agent.db_agent import DBAgent

def parse_args():
    parser = argparse.ArgumentParser(description='慢查询诊断工具')
    parser.add_argument('--data_path', type=str, required=True, help='慢查询数据文件路径')
    parser.add_argument('--order', type=str, required=True, help='慢查询执行顺序')
    parser.add_argument('--duration', type=int, default=30, help='调优时间')
    return parser.parse_args()

async def main():
    # cd /root/DREAM/src
    # python main.py --data_path /root/DREAM/data/slow_queries/TPC-H --order qorder.txt --duration 10
    logger = logging.getLogger(__name__)
    
    args = parse_args()
    logger.info(f"启动DREAM数据库诊断系统")
    logger.info(f"数据路径: {args.data_path}")
    logger.info(f"执行顺序: {args.order}")
    logger.info(f"调优时间: {args.duration}")
    
    configs = {
        'POSTGRES_CONFIG': POSTGRES_CONFIG,
        'KNOB_CONFIG': KNOB_CONFIG,
        'API_SETTINGS': API_SETTINGS
    }
    async with DBAgent(configs=configs) as agent:
        await agent.run(
            slow_query_path=args.data_path,
            order=args.order,
            duration=args.duration
        )

if __name__ == "__main__":
    # asyncio.run(main())
    try:
        asyncio.run(main())
    except asyncio.CancelledError:
        # anyio/MCP 在正常关闭 stdio 时会抛取消，属于正常关机路径
        pass
