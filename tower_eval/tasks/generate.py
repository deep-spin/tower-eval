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
def generate(i: int, config_path: str, available_models: dict=available_models) -> None:
    configs = parse_yaml_config(config_path)
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    )
    gen_data_dir = Path(configs.get("gen_data_dir"))
    gen_output_dir = Path(configs.get("gen_output_dir"))
    overwrite_generations = configs.get("overwrite_generations", False)
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
            input_file = gen_data_dir / task_name / subtask / "instructions.txt"
            output_path = gen_output_dir / task_name / subtask / model_type / model_name
            output_file = output_path / "generation.txt"
            make_dir_if_not_exists(output_file)
            metadata_file = output_path / "metadata.json"
            logger.opt(colors=True).info(
                f"Running inference for task: <yellow> {task_name} </yellow>, subtask: <green> {subtask} </green> with model: <red> {model_type}/{model_name} </red> saving to: <red> {output_file} </red>"
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
                overwrite_generations=overwrite_generations,
            )
            add_average_generation_time(
                output_file, metadata_file, language=tgt_lang, mode=average_time_metric
            )


def simple_generate(
    input_paths: list[str],
    output_paths: list[str],
    model_path: str,
    model_type: str,
    model_args: dict,
    metadata_file_paths: list[str],
    available_models: dict,
    overwrite_generations: bool = False,
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
            overwrite_generations=overwrite_generations
        )


if __name__ == "__main__":
    CLI([generate], as_positional=False)
