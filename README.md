# EvoTune

Source code for the EvoTune framework, proposed in "EvoTune: Learning from Feedback to Guide LLMs for Slow-Query Tuning". Please refer to the paper and appendix for the experimental details.

## Setup and Run

1) Environment setup

```bash
# Clone repository
git clone https://github.com/DBXAI/EvoTune.git
cd EvoTune
git submodule update --init --recursive src/agent/plan/RCRank
gzip -dc patches/rcrank_runtime.patch.gz | git -C src/agent/plan/RCRank apply -

# Create conda environment
conda create -n EvoTune python=3.10
conda activate EvoTune

# Install dependencies
pip install -r requirements.txt
```

2) Configure APIs

```bash
cp config/tpch_config.json.example config/tpch_config.json
cp config/tpcds_config.json.example config/tpcds_config.json
cp config/job_config.json.example config/job_config.json

# Set database connection and API settings in the copied configuration files.
# Relative model and memory paths are resolved from the config/ directory.
```

3) Install required tools

- **pg_hint_plan**: Required to support hint application in PostgreSQL. Please install `pg_hint_plan` extension following the [official documentation](https://github.com/ossc-db/pg_hint_plan).

- **QED (Query Equivalence Decider)**: Required for SQL query equivalence verification. For QED installation, please refer to `install_qed.sh` script:
  ```bash
  bash install_qed.sh
  ```
  Or manually follow the installation steps in the script.

4) Download detection data

- For the diagnosis tool, we provide both the pre-collected training data and the pre-trained RCRank model, which can be downloaded directly from the [link](https://drive.google.com/drive/folders/1mRkA_CJvKImJeHuaXm-W0h0riVmvB_2w?usp=sharing). You can also retrain the model and recollect data using the modified RCRank [code](https://github.com/HarneyHong/RCRank.git) if needed.

- From the repository root, save the downloaded predictor files to these paths:

  | Downloaded file | Save as | Config key |
  | --- | --- | --- |
  | Predictor checkpoint (`best_model.pt`) | `src/agent/plan/model_res/predictor.pt` | `PLANNER_CONFIG.model_path` |
  | Predictor training data (`slow_sql_data_gen.csv`) | `src/agent/plan/model_res/predictor_training.csv` | `PLANNER_CONFIG.train_data_path` |

  The example config files use `../src/...` for these paths because relative paths in a config are resolved from its `config/` directory. Keep the file names and config values in sync if you choose another location.
  The TPC-H TiDB example uses the separate `src/agent/plan/model_res/predictor_tidb.pt` checkpoint.

5) Setup Retriever Model

- The TPC-H offline-trained Retriever checkpoint (`retriever_all821.pt`) can be distributed as `soft_q_retriever.pt`. Place it at `src/agent/memory/model/soft_q_retriever.pt`, the path in `MEMORY_MANAGER_CONFIG.retriever_model_path`. This experimental checkpoint was trained with the old-knob Predictor (`predictor_tidb.pt` in these examples), so use that Predictor to reproduce its scores; its online tuning benefit has not been established.

- To train from the TPC-H offline samples, download the matching `retriever_samples.jsonl` and `memory.sqlite3` together. Place them at `src/agent/memory/experience/retriever_samples.jsonl` and `src/memory_results_tpch.sqlite3`, respectively. The JSONL stores case IDs for the next state and history; the trainer uses the SQLite cases to reconstruct their features and Bellman targets. With only the JSONL, missing next states are treated as terminal states.

- Train the Retriever from those files:
  ```bash
  python src/agent/memory/train_retriever_offline.py \
      --samples_path src/agent/memory/experience/retriever_samples.jsonl \
      --model_save_path src/agent/memory/model/soft_q_retriever.pt \
      --config_path config/tpch_config.json \
      --epochs 10 \
      --batch_size 64 \
      --learning_rate 0.0001
  ```
  For a fresh training run, set `MEMORY_MANAGER_CONFIG.retriever_model_path` in the copied config to a file that does not yet exist; otherwise the trainer loads that checkpoint before training.

- To collect new experience data from scratch, set `enable_save_samples` to `true` in the configuration file. The collected samples are saved to the path specified by `samples_save_path` (default: `src/agent/memory/experience/retriever_samples.jsonl` from the repository root).

6) Run

```bash
python src/main.py \
  --data_path data/slow_queries/TPC-H \
  --order qorder.txt \
  --duration 30 \
  --config config/tpch_config.json
```
