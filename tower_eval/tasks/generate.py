import sys
from pathlib import Path

from jsonargparse import CLI
from loguru import logger

from tower_eval.models import available_models
from tower_eval.utils import (
    add_average_generation_time,
    get_langs,
    make_dir_if_not_exists,
    parse_yaml_config,
)


def parse_gen_eval_config(config_path: str) -> dict:
    config_args = parse_yaml_config(config_path)
    # make eval and gen config
    configs = {"gen": {}, "eval": {}}
    for step in ["gen", "eval"]:
        configs[step]["output_dir"] = config_args[f"{step}_output_dir"]
        configs[step]["data_dir"] = config_args[f"{step}_data_dir"]
        configs[step]["tasks"] = []
        # create task structure
        for task in config_args.get("tasks"):
            subtask_dict = task["subtasks"]
            task_dict = {"name": task.get("name"), "subtasks": {}}
            for subtask in task.get("subtasks"):
                task_dict["subtasks"][subtask] = {}
                if step == "eval":
                    # if we define specific eval args for a given subtask
                    if subtask_dict[subtask] is not None:
                        for arg in subtask_dict[subtask]["eval_args"]:
                            task_dict["subtasks"][subtask][arg] = subtask_dict[subtask][
                                f"{step}_args"
                            ][arg]
                    task_dict["metrics"] = task.get("metrics")
            configs[step]["tasks"].append(task_dict)
        # create model structure
        configs[step]["models"] = []
        for model_dict in config_args.get("models"):
            # add hypothesis file to eval config
            if step == "eval":
                model_dict["hypothesis_dir"] = config_args["gen_output_dir"]
            configs[step]["models"].append(model_dict)
    return configs["gen"]


def generate(i: int, config_path: str, config_type: str, available_models: dict=available_models) -> None:
    if config_type == "generate":
        configs = parse_yaml_config(config_path)
    elif config_type == "gen-eval":
        configs = parse_gen_eval_config(config_path)
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    )
    data_dir = Path(configs.get("data_dir"))
    output_dir = Path(configs.get("output_dir"))
    average_time_metric = configs.get("average_time_metric", "lps")
    tasks = configs.get("tasks")
    model = configs.get("models")[i]
    model_name = model.get("name")
    model_type = model.get("type")
    model_args = model.get("arguments")
    model_args = {} if not model_args else model_args
    model = available_models[model_type](**(model_args))
    for task in tasks:
        task_name = task.get("name")
        subtasks = task.get("subtasks")
        for subtask, _ in subtasks.items():
            input_file = data_dir / task_name / subtask / "instructions.txt"
            output_path = output_dir / task_name / subtask / model_type / model_name
            output_file = output_path / "generation.txt"
            make_dir_if_not_exists(output_file)
            metadata_file = output_path / "metadata.json"
            logger.opt(colors=True).info(
                f"Running inference for task: <yellow> {task_name} </yellow>, subtask: <green> {subtask} </green> with model: <red> {model_type}/{model_name} </red> saving to: <red> {output_dir} </red>"
            )

            lp = subtask.split(".")[-1]
            src_lang, tgt_lang = get_langs(lp)

            model.source_language = src_lang
            model.target_language = tgt_lang
            model.generation_with_resume(
                input_file=input_file,
                output_file=output_file,
                metadata=configs,
                metadata_file=metadata_file,
            )
            add_average_generation_time(
                output_file, metadata_file, mode=average_time_metric
            )


def simple_generate(
    input_paths: list[str],
    output_paths: list[str],
    model_path: str,
    model_type: str,
    model_args: dict,
    metadata_file_paths: list[str],
    available_models: dict
):
    model_path_key = "model_dir" if model_type == "vllm" else "model"
    model_args[model_path_key] = model_path
    model = available_models[model_type](**(model_args))
    metadata = {
        "model_type": model_type,
        "model_args": model_args,
    }
    for input_path, output_path, metadata_file_path in zip(
        input_paths, output_paths, metadata_file_paths
    ):
        model.generation_with_resume(
            input_file=input_path,
            output_file=output_path,
            metadata=metadata,
            metadata_file=metadata_file_path,
        )


if __name__ == "__main__":
    CLI([generate], as_positional=False)
