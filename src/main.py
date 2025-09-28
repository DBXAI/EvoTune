import asyncio
import argparse
import sys
from pathlib import Path
import logging
import os
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志，同时输出到控制台和文件
def setup_logging():
    # 创建logs目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名（包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"dream_diagnosis_{timestamp}.log")
    
    # 配置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 创建根日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除现有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    print(f"日志文件保存位置: {log_file}")
    return log_file

# 在导入其他模块之前先配置日志
setup_logging()

from config.base_config import POSTGRES_CONFIG, KNOB_CONFIG, API_SETTINGS
from agent.db_agent import DBAgent

def parse_args():
    parser = argparse.ArgumentParser(description='慢查询诊断工具')
    parser.add_argument('--data_path', type=str, required=True, help='慢查询数据文件路径')
    parser.add_argument('--order', type=str, required=True, help='慢查询执行顺序')
    parser.add_argument('--max_iterations', type=int, default=200, help='最大诊断迭代轮数')
    return parser.parse_args()

async def main():
    # cd /root/DREAM/src
    # python main.py --data_path /root/DREAM/data/slow_queries/TPC-H --order qorder.txt --max_iterations 10
    logger = logging.getLogger(__name__)
    
    args = parse_args()
    logger.info(f"启动DREAM数据库诊断系统")
    logger.info(f"数据路径: {args.data_path}")
    logger.info(f"执行顺序: {args.order}")
    logger.info(f"最大迭代轮数: {args.max_iterations}")
    
    configs = {
        'POSTGRES_CONFIG': POSTGRES_CONFIG,
        'KNOB_CONFIG': KNOB_CONFIG,
        'API_SETTINGS': API_SETTINGS
    }
    async with DBAgent(configs=configs) as agent:
        await agent.run(
            slow_query_path=args.data_path,
            order=args.order,
            max_iterations=args.max_iterations
        )

if __name__ == "__main__":
    # asyncio.run(main())
    try:
        asyncio.run(main())
    except asyncio.CancelledError:
        # anyio/MCP 在正常关闭 stdio 时会抛取消，属于正常关机路径
        pass
