import os
import sys

# Add RCRank to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
rcrank_path = os.path.join(current_dir, "RCRank")
if rcrank_path not in sys.path:
    sys.path.append(rcrank_path)

import torch
import torch.nn as nn
import pandas as pd
import json
import numpy as np
from transformers import BertTokenizer, BertModel

# Local imports from RCRank
from model.modules.rcrank_model import GateComDiffPretrainModel
from RCRank.utils.config import Args, ArgsPara, TrainConfig
from model.modules.QueryFormer.utils import Encoding
from model.modules.FuseModel.CrossTransformer import CrossTransformer
from model.modules.FuseModel.Attention import MultiHeadedAttention
from model.modules.TSModel.ts_model import CustomConvAutoencoder
from RCRank.utils.plan_encoding import PlanEncoder

class RCRankPredictor:
    def __init__(self, model_path="/root/DREAM/src/agent/plan/model_res/GateComDiffPretrainModel slow_sql_data_gen eta0.07/best_model.pt", 
                train_data_path="/root/DREAM/src/agent/plan/slow_sql_data_gen.csv", device='cpu', opt_threshold=0.1, num_classes=4):
        #init传入的参数要修改
        self.device = device
        self.num_classes = num_classes
        
        # Get the absolute path to bert-base-uncased
        current_dir = os.path.dirname(os.path.abspath(__file__))
        bert_path = os.path.join(current_dir, "RCRank/bert-base-uncased")
        
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.opt_threshold = opt_threshold
        self.pred_type = "multilabel"
        
        # 初始化配置
        self.plan_args = Args()
        self.plan_args.device = device
        self.para_args = ArgsPara()
        
        # 从训练数据计算统计信息
        self._compute_statistics(train_data_path)
        
        # 初始化模型组件
        sql_model = BertModel.from_pretrained(bert_path)
        time_model = CustomConvAutoencoder()
        
        # 初始化 CrossTransformer 和 MultiHeadedAttention
        fuse_num_layers = 3
        fuse_head_size = 4
        emb_dim = 32
        fuse_ffn_dim = 128
        dropout = 0.1
        use_metrics = True
        use_log = True

        multihead_attn_modules_cross_attn = nn.ModuleList(
            [MultiHeadedAttention(fuse_head_size, emb_dim, dropout=dropout, use_metrics=use_metrics, use_log=use_log)
            for _ in range(fuse_num_layers)])
        fuse_model = CrossTransformer(num_layers=fuse_num_layers, d_model=emb_dim, heads=fuse_head_size, 
                                    d_ff=fuse_ffn_dim, dropout=dropout, attn_modules=multihead_attn_modules_cross_attn)

        r_attn_model = nn.ModuleList(
            [MultiHeadedAttention(fuse_head_size, emb_dim, dropout=dropout, use_metrics=use_metrics, use_log=use_log)
            for _ in range(fuse_num_layers)])
        rootcause_cross_model = CrossTransformer(num_layers=fuse_num_layers, d_model=emb_dim, heads=fuse_head_size,
                                               d_ff=fuse_ffn_dim, dropout=dropout, attn_modules=r_attn_model)
        
        # 初始化主模型
        self.model = GateComDiffPretrainModel(
            t_input_dim=9,
            l_input_dim=13,
            l_hidden_dim=64,
            t_hidden_him=64,
            emb_dim=32,
            device=device,
            plan_args=self.plan_args,
            sql_model=sql_model,
            cross_model=fuse_model,
            time_model=time_model,
            rootcause_cross_model=rootcause_cross_model
        ).to(device)
        
        # 加载预训练模型
        print(f"Loading model from {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def _compute_statistics(self, train_data_path):
        """计算训练数据的统计信息"""
        print(f"Computing statistics from training data: {train_data_path}")
        df = pd.read_csv(train_data_path)
        
        # 删除error不为空的行
        df = df[df['error'].isna()].copy()
        
        # 解析JSON数据
        df["internal_metrics"] = df["internal_metrics"].apply(json.loads)
        df["external_metrics"] = df["external_metrics"].apply(json.loads)
        logs = []
        timeseries = []
        for _, row in df.iterrows():
            logs.append(torch.tensor(row["internal_metrics"]))
            timeseries.append(torch.tensor(row["external_metrics"]))

        logs = torch.stack(logs, dim=0)
        self.logs_train_mean = logs.mean(dim=0)
        self.logs_train_std = logs.std(dim=0)

        timeseries = torch.stack(timeseries, dim=0)
        self.timeseries_train_mean = timeseries.mean(dim=[0, 2])
        self.timeseries_train_std = timeseries.std(dim=[0, 2])
        print("Statistics computation completed")

    def predict(self, query, plan, timeseries, log):
        """
        对单个输入进行预测
        Args:
            query: SQL查询语句
            plan: 执行计划数据
            timeseries: 时间序列数据
            log: 日志数据
        Returns:
            pred_label: 预测的多标签
            pred_label_binary: 二值化的预测标签
        """
        with torch.no_grad():
            # 处理SQL查询
            sql = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            
            # 处理执行计划数据
            plan = {
                'x': plan['x'].to(self.device).to(torch.float32),
                'attn_bias': plan['attn_bias'].to(self.device),
                'rel_pos': plan['rel_pos'].to(self.device),
                'heights': plan['heights'].to(self.device)
            }
            
            # 处理并标准化时间序列数据
            timeseries = torch.tensor(timeseries)
            if len(timeseries.shape) == 2:
                timeseries = timeseries.unsqueeze(0)  # 添加batch维度
            # 标准化
            timeseries = (timeseries - self.timeseries_train_mean.unsqueeze(1)) / (self.timeseries_train_std.unsqueeze(1) + 1e-6)
            timeseries = timeseries.to(self.device)
            
            # 处理并标准化日志数据
            log = torch.tensor(log)
            if len(log.shape) == 1:
                log = log.unsqueeze(0)  # 添加batch维度
            # 标准化
            log = (log - self.logs_train_mean) / (self.logs_train_std + 1e-6)
            log = log.to(self.device)
            
            # 模型预测
            pred_label_opt, _ = self.model(sql, plan, timeseries, log)
            
            # 二值化预测标签
            pred_label = torch.where(pred_label_opt > 0.5, 1, 0)
            
            return pred_label_opt, pred_label

def load_sample_data(data_path):
    """
    从CSV文件中加载第一行数据并进行预处理
    """
    print("Loading sample data from", data_path)
    
    # 读取CSV文件的第一行
    df = pd.read_csv(data_path)
    
    # 删除error不为空的行
    df = df[df['error'].isna()].copy()
    
    # JSON字符串转换为Python对象
    df["internal_metrics"] = df["internal_metrics"].apply(json.loads)
    df["external_metrics"] = df["external_metrics"].apply(json.loads)
    
    # 初始化编码器
    encoding = Encoding(None, {'NA': 0})
    
    # 处理执行计划
    pe = PlanEncoder(df=df, encoding=encoding)
    df = pe.df
    
    # 获取第一行数据
    row = df.iloc[0]
    
    # 构造处理后的样本数据
    sample = {
        "query": row["query"],
        "plan": row["json_plan_tensor"],
        "log": row["internal_metrics"],
        "timeseries": row["external_metrics"],
        "multilabel": torch.tensor(eval(row["multilabel"])),
        "duration": row["duration"]
    }
    
    return sample

def online_predict(data_path):
    # 初始化预测器
    predictor = RCRankPredictor(
        model_path="model_res/GateComDiffPretrainModel slow_sql_data_gen eta0.07/best_model.pt",
        train_data_path=data_path,
        device='cpu',
        opt_threshold=0.5
    )
    
    sample = load_sample_data(data_path)
    
    # 获取预测结果
    pred_label, pred_label_binary = predictor.predict(
        query=sample["query"],
        plan=sample["plan"],
        timeseries=sample["timeseries"],
        log=sample["log"]
    )
    
    print("\nResults:")
    print("Input Query:", sample["query"])
    print("\nPredicted labels:", pred_label.cpu().numpy())
    print("Binary Predicted labels:", pred_label_binary.cpu().numpy())
    print("\nTrue labels:", sample["multilabel"].numpy())

    
if __name__ == "__main__":
    # 加载示例数据
    data_path = "./slow_sql_data_gen.csv"
    online_predict(data_path)