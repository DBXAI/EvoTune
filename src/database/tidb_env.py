import pymysql
import time
import json
from typing import Dict, Any, List, Tuple, Optional
from decimal import Decimal
import ssl
import os
import subprocess
import psutil
import socket

class TiDBConnector:
    def __init__(self, host: str, port: int, user: str, password: str, database: str, db_type: str,
                 ssl_mode: str = "VERIFY_IDENTITY", ssl_ca: str = "/etc/ssl/certs/ca-certificates.crt"):
        """
        初始化TiDB连接器
        
        Args:
            host: TiDB服务器地址
            port: 端口号
            user: 用户名
            password: 密码
            database: 数据库名
            db_type: 数据库类型
            ssl_mode: SSL模式 (DISABLED, REQUIRED, VERIFY_IDENTITY)
            ssl_ca: SSL证书路径
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.ssl_mode = ssl_mode
        self.ssl_ca = ssl_ca
        self.db_type = db_type
        self.connection = None
        
    def _check_tidb_running(self) -> bool:
        """检查TiDB是否在指定端口运行"""
        try:
            # 尝试连接端口
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((self.host, self.port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    def _check_tiup_playground_running(self) -> bool:
        """检查是否有tiup playground进程在运行"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info['cmdline']
                    if cmdline and 'tiup' in cmdline and 'playground' in cmdline and 'DREAM' in cmdline:
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return False
        except Exception:
            return False
    
    def _start_tiup_playground(self) -> bool:
        """启动tiup playground"""
        try:
            print("正在启动tiup playground --tag DREAM...")
            # 使用subprocess启动tiup playground，设置为后台运行
            process = subprocess.Popen(
                ['tiup', 'playground', '--tag', 'DREAM'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # 创建新的进程组
            )
            
            # 等待一段时间让服务启动
            time.sleep(10)
            
            # 检查是否成功启动
            if self._check_tidb_running():
                print("tiup playground启动成功！")
                return True
            else:
                print("tiup playground启动失败或超时")
                return False
                
        except Exception as e:
            print(f"启动tiup playground时发生错误: {e}")
            return False
    
    def _ensure_tidb_running(self):
        """确保TiDB正在运行，如果没有则启动"""
        if not self._check_tidb_running():
            print("TiDB未在运行，检查tiup playground状态...")
            
            if not self._check_tiup_playground_running():
                print("未找到tiup playground进程，正在启动...")
                if not self._start_tiup_playground():
                    raise Exception("无法启动TiDB服务")
            else:
                print("找到tiup playground进程，等待服务启动...")
                # 等待服务完全启动
                max_wait = 30  # 最多等待30秒
                wait_time = 0
                while not self._check_tidb_running() and wait_time < max_wait:
                    time.sleep(2)
                    wait_time += 2
                
                if not self._check_tidb_running():
                    raise Exception("TiDB服务启动超时")
        
        print("TiDB服务已就绪")
        
    def _get_ssl_config(self) -> Dict[str, Any]:
        """获取SSL配置"""
        if self.ssl_mode == "VERIFY_IDENTITY":
            return {
                'ssl': {
                    'ca': self.ssl_ca,
                    'check_hostname': True,
                    'verify_mode': ssl.CERT_REQUIRED
                }
            }
        elif self.ssl_mode == "REQUIRED":
            return {
                'ssl': {
                    'ca': self.ssl_ca,
                    'check_hostname': False,
                    'verify_mode': ssl.CERT_REQUIRED
                }
            }
        elif self.ssl_mode == "DISABLED":
            return {}
        else:
            return {}
    
    def connect(self):
        """建立并返回一个数据库连接"""
        # 确保TiDB服务正在运行
        self._ensure_tidb_running()
        
        if self.connection is None or not self.connection.open:
            ssl_config = self._get_ssl_config()
            self.connection = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor,
                **ssl_config
            )
        return self.connection
    
    def execute(self, sql: str, conn=None) -> Tuple[bool, Optional[List], float, str]:
        """
        执行SQL语句
        
        Args:
            sql: SQL语句
            conn: 数据库连接，如果为None则使用默认连接
            
        Returns:
            (success, results, execution_time, error_message)
        """
        if conn is None:
            conn = self.connect()
        
        try:
            start_time = time.time()
            with conn.cursor() as cursor:
                cursor.execute(sql)
                results = cursor.fetchall()
                conn.commit()
            end_time = time.time()
            return True, results, end_time - start_time, ""
        except pymysql.Error as e:
            return False, None, -1, str(e)
        except Exception as e:
            return False, None, -1, str(e)
    
    def fetch_results(self, sql: str, as_json: bool = False) -> Any:
        """
        获取查询结果
        
        Args:
            sql: SQL查询语句
            as_json: 是否返回JSON格式
            
        Returns:
            查询结果
        """
        conn = self.connect()
        with conn.cursor() as cursor:
            cursor.execute(sql)
            results = cursor.fetchall()
            
            if as_json:
                return json.dumps(results, indent=4, ensure_ascii=False, default=self._decimal_default)
            
            # 如果只有一列，返回单列结果
            if results and len(results[0]) == 1:
                return [list(row.values())[0] for row in results]
            
            return results
    
    def get_tables(self) -> List[str]:
        """获取所有表名"""
        sql = """
        SELECT table_name FROM information_schema.tables 
        WHERE table_schema = %s AND table_type = 'BASE TABLE'
        """
        conn = self.connect()
        with conn.cursor() as cursor:
            cursor.execute(sql, (self.database,))
            results = cursor.fetchall()
            return [row['table_name'] for row in results]
    
    def get_views(self) -> List[Tuple[str, str]]:
        """获取所有视图名和定义"""
        sql = """
        SELECT table_name as viewname, view_definition as definition 
        FROM information_schema.views 
        WHERE table_schema = %s
        """
        conn = self.connect()
        with conn.cursor() as cursor:
            cursor.execute(sql, (self.database,))
            results = cursor.fetchall()
            return [(row['viewname'], row['definition']) for row in results]
    
    def get_columns(self) -> List[str]:
        """获取所有列名"""
        sql = """
        SELECT column_name FROM information_schema.columns 
        WHERE table_schema = %s
        """
        conn = self.connect()
        with conn.cursor() as cursor:
            cursor.execute(sql, (self.database,))
            results = cursor.fetchall()
            return [row['column_name'] for row in results]
    
    def fetch_schema_info(self) -> Dict[str, Dict]:
        """获取数据库模式信息"""
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
            table_name = row['table_name']
            if table_name not in schema_dict:
                schema_dict[table_name] = {"columns": []}
            
            schema_dict[table_name]["columns"].append({
                "column_name": row['column_name'],
                "data_type": row['data_type'],
                "is_nullable": row['is_nullable'],
                "column_default": row['column_default'],
                "column_key": row['column_key'],
                "extra": row['extra']
            })
        
        return schema_dict
    
    def get_db_size(self) -> float:
        """获取数据库大小（MB）"""
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
            return float(result['size_mb']) if result['size_mb'] else 0.0
    
    def get_plan(self, sql):
        sql = f"""
        EXPLAIN {sql};
        """
        return self.fetch_results(sql)
    
    def compare_sql_results(self, sql1: str, sql2: str) -> bool:
        """
        比较两个SQL语句的结果是否一致
        
        Args:
            sql1: 第一个SQL语句
            sql2: 第二个SQL语句
            
        Returns:
            是否一致
        """
        # 去掉结尾的分号和空白
        sql1 = sql1.strip().rstrip(';')
        sql2 = sql2.strip().rstrip(';')
        
        conn = self.connect()
        with conn.cursor() as cursor:
            try:
                # 执行两个查询
                cursor.execute(sql1)
                results1 = cursor.fetchall()
                
                cursor.execute(sql2)
                results2 = cursor.fetchall()
                
                # 比较结果数量
                if len(results1) != len(results2):
                    return False
                
                # 将结果转换为可比较的格式（元组列表）
                def normalize_results(results):
                    normalized = []
                    for row in results:
                        # 将字典转换为排序后的元组
                        if isinstance(row, dict):
                            normalized.append(tuple(sorted(row.items())))
                        else:
                            normalized.append(tuple(row))
                    return sorted(normalized)
                
                normalized1 = normalize_results(results1)
                normalized2 = normalize_results(results2)
                
                # 比较标准化后的结果
                return normalized1 == normalized2
                
            except Exception as e:
                print(f"比较SQL结果时发生错误: {e}")
                return False
    
    def close(self):
        """关闭数据库连接"""
        if self.connection and self.connection.open:
            self.connection.close()
    
    def _decimal_default(self, obj):
        """处理Decimal类型的JSON序列化"""
        if isinstance(obj, Decimal):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")