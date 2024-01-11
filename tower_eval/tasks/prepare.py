from pathlib import Path
from typing import Any

import jinja2
from loguru import logger
from tower_eval.fewshot_retrieval_utils import load_few_shot_data
from tower_eval.utils import load_data_to_records, sample_strings_from_list, write_lines


def apply_prompt_templates(
    prompt_templates: list[str],
    prompt_args: dict = {},
    data: dict[list[str]] = {},
    fewshot_examples_list: list[dict[list[str]]] = {},
) -> list[str]:
    """ """
    # Sample prompt templates from list of templates
    prompt_templates = sample_strings_from_list(prompt_templates, len(data))
    # prompt_args and data record should not have matching keys.
    if data:
        for record in data:
            record.update(prompt_args)
    else:
        data = [prompt_args] * len(prompt_templates)
    # add few shot examples data, if exists and create formatted data
    if fewshot_examples_list:
        for data_record, fewshot_examples in zip(data, fewshot_examples_list):
            data_record["examples"] = fewshot_examples
    # compile templates and render with arguments
    env = jinja2.Environment()
    compiled_templates = [env.from_string(prompt) for prompt in prompt_templates]
    formatted_data = [t.render(**record) for record, t in zip(data, compiled_templates)]

    return formatted_data


def prepare_data(
    prompt_templates: list[str],
    prompt_args: dict = {},
    prompt_data_path: str = "",
    fewshot_data_path: str = None,
    n_fewshots: int = 0,
    fewshot_retrieval_method: str = None,
    fewshot_retrieval_args: dict = {},
    task_name: str = "task",
    subtask_name: str = "subtask",
    index_path: str = None,
    output_dir: Path = "tests/data",
) -> None:
    """ """
    logger.opt(ansi=True).info(f"========================================")
    logger.opt(ansi=True).info(
        f"Preparing data of task: <yellow> {task_name} </yellow>, subtask: <green> {subtask_name} </green>"
    )
    # read data objects into lists of strings
    # It is crucial that the lists in data be of the same size and of the final size of the dataset
    if prompt_data_path:
        data = load_data_to_records(prompt_data_path)
    # read fewshot data objects into lists of strings
    fewshot_examples_list: list[dict[list[str]]] = []
    if fewshot_data_path is not None:
        total_shots = n_fewshots * len(data)
        fewshot_examples_list = load_few_shot_data(
            test_set=data,
            fewshot_data_path=fewshot_data_path,
            n_fewshots=n_fewshots,
            total_shots=total_shots,
            fewshot_retrieval_method=fewshot_retrieval_method,
            task=task_name,
            index_path=index_path,
            fewshot_retrieval_args=fewshot_retrieval_args,
        )
    prepared_data = apply_prompt_templates(
        prompt_templates=prompt_templates,
        prompt_args=prompt_args,
        data=data,
        fewshot_examples_list=fewshot_examples_list,
    )

    output_path = output_dir / f"{task_name}/{subtask_name}/instructions.txt"
    write_lines(output_path, prepared_data, escape_newline=True)
