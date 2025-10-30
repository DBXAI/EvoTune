# DREAM

Source code for the DREAM framework, proposed in "From Anomalies to Actions: An Anomaly-Aware Approach for Multi-Component Tuning in DBMSs". Please refer to the paper for the experimental details.

## Setup and Run

1) Environment setup

```bash
# Clone repository
git clone https://github.com/HarneyHong/DREAM.git
cd DREAM

# Create conda environment
conda create -n DREAM python=3.10
conda activate DREAM

# Install dependencies
pip install -r requirements.txt
```

2) Configure APIs

```bash
cd config
cp base_config.json.example base_config.json

# In base_config.py, set API keys and models if needed
# OPENAI_CONFIG / ANTHROPIC_CONFIG are already defined in example
```

3) Download detection data

For the diagnosis tool, we provide both the pre-collected training data and the pre-trained RCRank model, which can be downloaded directly from the [link](https://drive.google.com/drive/folders/1mRkA_CJvKImJeHuaXm-W0h0riVmvB_2w?usp=sharing). You can also retrain the model and recollect data using the modified RCRank [code](https://
github.com/HarneyHong/RCRank.git) if needed.

After downloading, please place the data and model files in the following specified paths:
- DREAM/src/agent/plan/model_res/GateComDiffPretrainModel slow_sql_data_gen eta0.07/best_model.pt
- DREAM/src/agent/plan/slow_sql_data_gen.csv

4) Run

```bash
cd src
python main.py \
  --data_path /root/DREAM/data/slow_queries/TPC-H \
  --order qorder.txt \
  --duration 30 \
  --config /root/DREAM/config/base_config.json
```