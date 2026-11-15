"""Resolve repository file paths in a JSON config relative to that config file."""

from pathlib import Path


CONFIG_PATH_FIELDS = {
    "PLANNER_CONFIG": ("model_path", "train_data_path"),
    "MEMORY_MANAGER_CONFIG": (
        "db_path",
        "sqlite_path",
        "experience_path",
        "model_output_dir",
        "samples_save_path",
        "retriever_model_path",
        "bert_model_path",
    ),
}


def resolve_config_paths(configs, config_path):
    """Make configured project paths independent of the process working directory."""
    config_dir = Path(config_path).resolve().parent
    for section, fields in CONFIG_PATH_FIELDS.items():
        values = configs.get(section)
        if not isinstance(values, dict):
            continue
        for field in fields:
            value = values.get(field)
            if value:
                path = Path(value).expanduser()
                values[field] = str((config_dir / path).resolve())
    return configs
