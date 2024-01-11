import re
from pathlib import Path
from time import time
from typing import Dict, Tuple

import wandb
from tower_eval.utils import PATTERN_SHOT_NAME, load_json_file

METRICS_PER_TASK = {
    "mt": ["comet", "chrf", "bleu"],
    "ape": ["ter", "comet", "comet_kiwi", "chrf", "bleu"],
    "gec": ["errant", "ter"],
    "ner": ["f1sequence"],
}

PROJECT = "tower-eval"
ROOT_DIR = "<ROOT_DIR>"


def create_wandb_names_from_path(path: Path) -> Tuple[str]:
    model = path.parts[-2]
    model_type = path.parts[-3]
    task = path.parts[-5]
    shot_setting = path.parts[-6]
    subtask = path.parts[-4]
    dataset_name, lp_xor_language = subtask.split(".")
    shot_name_short = re.sub(PATTERN_SHOT_NAME, r"\1", shot_setting)
    model_name_and_setting = f"{model} ({shot_name_short} shot)"

    return (
        model,
        model_type,
        task,
        shot_setting,
        subtask,
        dataset_name,
        lp_xor_language,
        model_name_and_setting,
        shot_name_short,
    )


def get_data_to_log(
    model_name_and_setting: str,
    shot_name_short: str,
    task: str,
    subtask: str,
    dataset_name: str,
    lp_xor_language: str,
    metric: str,
    score: str,
    model_type: str,
    model: str,
    shot_setting: str,
) -> Tuple[Dict, Dict, str]:
    data_to_log = {
        "model": model_name_and_setting,
        "shots": shot_name_short,
        "task": task,
        "subtask": subtask,
        "dataset": dataset_name,
        "lp/language": lp_xor_language,
        "metric": metric,
        "score": score,
        "model_type": model_type,
        "model_raw_name": model,
        "shot_setting": shot_setting,
    }
    config_to_log = {k: v for k, v in data_to_log.items() if k != "score"}
    table_name = f"table_{task}_{dataset_name}_{metric}"

    return data_to_log, config_to_log, table_name


def log_one_entry(
    project: str,
    model_name_and_setting: str,
    shot_name_short: str,
    task: str,
    subtask: str,
    dataset_name: str,
    lp_xor_language: str,
    metric: str,
    score: str,
    model_type: str,
    model: str,
    shot_setting: str,
) -> None:
    data_to_log, config_to_log, table_name = get_data_to_log(
        model_name_and_setting,
        shot_name_short,
        task,
        subtask,
        dataset_name,
        lp_xor_language,
        metric,
        score,
        model_type,
        model,
        shot_setting,
    )
    wandb.init(
        project=project, name=table_name.split("table_")[-1], config=config_to_log
    )
    wandb.log(
        {
            table_name: wandb.Table(
                columns=list(data_to_log.keys()), data=[list(data_to_log.values())]
            )
        }
    )
    wandb.finish(quiet=True)


def log_from_existing_repo():
    wandb_tables = {}

    evaluation_paths = [p for p in Path(ROOT_DIR).rglob("*.json")]

    for p in evaluation_paths:
        (
            model,
            model_type,
            task,
            shot_setting,
            subtask,
            dataset_name,
            lp_xor_language,
            model_name_and_setting,
            shot_name_short,
        ) = create_wandb_names_from_path(p)
        metrics_to_log = METRICS_PER_TASK[task]

        evaluations = load_json_file(p)
        for metric in metrics_to_log:
            try:
                score = evaluations[metric]
            except KeyError:
                score = None
            data_to_log, config_to_log, table_name = get_data_to_log(
                model_name_and_setting,
                shot_name_short,
                task,
                subtask,
                dataset_name,
                lp_xor_language,
                metric,
                score,
                model_type,
                model,
                shot_setting,
            )
            if table_name not in wandb_tables:
                wandb_tables[table_name] = {
                    "columns": list(data_to_log.keys()),
                    "data": [],
                    "config": config_to_log,
                }
            else:
                wandb_tables[table_name]["data"].append(list(data_to_log.values()))

    for table_name, table_dict in wandb_tables.items():
        s = time()
        wandb.init(
            project=PROJECT,
            name=table_name.split("table_")[-1],
            config=table_dict["config"],
        )
        wandb.log(
            {
                table_name: wandb.Table(
                    columns=table_dict["columns"], data=table_dict["data"]
                )
            }
        )
        wandb.finish(quiet=True)
        e = time()
        print(f"Took {e-s:.2f}s")
