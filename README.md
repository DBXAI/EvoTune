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

- For the case retriever, we provide a pre-trained Soft Q Retriever model that can be downloaded directly from the [Google Drive](https://drive.google.com/drive/folders/1mRkA_CJvKImJeHuaXm-W0h0riVmvB_2w?usp=sharing). After downloading, place the model file at the path specified by `MEMORY_MANAGER_CONFIG.retriever_model_path` (for TPC-H: `src/agent/memory/model/soft_q_retriever.pt` from the repository root).

- Alternatively, you can retrain the retriever model using collected experience samples:
  ```bash
  python src/agent/memory/train_retriever_offline.py \
      --samples_path src/agent/memory/experience/retriever_samples.jsonl \
      --model_save_path src/agent/memory/model/soft_q_retriever.pt \
      --config_path config/tpch_config.json \
      --epochs 10 \
      --batch_size 64 \
      --learning_rate 0.0001
  ```

- To collect new experience data from scratch, set `enable_save_samples` to `true` in the configuration file. The collected samples are saved to the path specified by `samples_save_path` (default: `src/agent/memory/experience/retriever_samples.jsonl` from the repository root).

6) Run

```bash
python src/main.py \
  --data_path data/slow_queries/TPC-H \
  --order qorder.txt \
  --duration 30 \
  --config config/tpch_config.json
```

`--max-rounds 3` limits a run to three completed passes. `--resume-state PATH` resumes from the checkpoint's `next_round`; for example, `next_round: 20` logs rounds 20–22 when combined with `--max-rounds 3`. To start at Round 1, omit `--resume-state` and use a fresh empty `MEMORY_MANAGER_CONFIG.db_path` (and fresh sample/model output paths). If historical memory already contains IDs such as `1-01`, continue after its highest round to avoid overwriting cases.

Memory cases are stored in SQLite at `<db_path>.sqlite3` by default. On first startup, existing cases in the JSON file at `db_path` are copied into SQLite in their original order; the JSON file is retained as a backup and is not updated afterward. SQLite is authoritative on later runs. Set `MEMORY_MANAGER_CONFIG.sqlite_path` to override the SQLite location. Retrieval still loads all cases into memory. A fresh run needs a new SQLite path as well as a fresh legacy JSON path.
