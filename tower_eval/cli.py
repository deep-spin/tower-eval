# -*- coding: utf-8 -*-
"""
Tower-Eval command line interface (CLI)
==============
Composed by 4 main commands:
    - prepare       Used to prepare the test data with the given prompt so that you can directly use to run the tests
    - generate      Used to generate the hypotheses with the given prompt.
    - evaluate      Used to run the evaluation metrics.
    - gen-eval      Used to first run the generation and then the evaluation on the produced hypotheses.
"""
import argparse
import random
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
from loguru import logger

from tower_eval.metrics import available_metrics
from tower_eval.models import available_models
from tower_eval.tasks.generate import generate, simple_generate
from tower_eval.tasks.index import index_data
from tower_eval.tasks.prepare import prepare_data
from tower_eval.utils import (
    combine_metrics_args,
    get_eval_args_given_task,
    handle_subprocess,
    load_json_file,
    parse_dict_arg,
    parse_yaml_config,
    save_to_json,
)


def run_evaluations(configs: dict, available_metrics: dict = available_metrics) -> dict:
    all_scores = defaultdict(lambda: defaultdict(dict))
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    )
    config_to_save = deepcopy(configs)
    output_dir = Path(configs.get("output_dir", None))
    data_dir = Path(configs.get("data_dir"))
    for task in configs.get("tasks"):
        task_results = {}
        task_name = task.get("name")
        subtasks = task.get("subtasks")
        task_metrics = task.get("metrics")
        for task_metric, task_metric_args in task_metrics.items():
            # Use empty dictionary if the arguments is None.
            # This is done to be able to use update function later on.
            task_metric_args = {} if task_metric_args is None else task_metric_args
            instantiated_metric = available_metrics[task_metric]()
            for model in configs.get("models"):
                model_type = model["type"]
                model_name = model["name"]
                for subtask, subtask_args in subtasks.items():
                    logger.opt(colors=True).info(
                        f"Evaluating the results of model <green> {model_name} </green> on task: <yellow> {task_name} </yellow>, subtask: <green> {subtask} </green> with metric: <red> {task_metric} </red>"
                    )
                    subtask_results = {}
                    # Subtasks can overwrite/add some arguments to the metric argument defined on the task level
                    # For example, we should be able to define a specific tokenizer for Chinese only
                    if subtask_args is None:
                        subtask_args = {}
                    subtask_metrics = (
                        subtask_args.get("metrics", {})
                        if subtask_args is not None
                        else {}
                    )
                    output_path = (
                        output_dir
                        / task_name
                        / subtask
                        / model["type"]
                        / model["name"]
                        / "evaluation.json"
                    )
                    # update subtask specific args, if they are specified
                    subtask_metric_args = subtask_metrics.get(task_metric)
                    subtask_metric_args = (
                        {} if not subtask_metric_args else subtask_metric_args
                    )
                    eval_args = combine_metrics_args(
                        task_metric_args, subtask_metric_args
                    )

                    subtask_metrics[task_metric] = eval_args
                    eval_args.update(
                        {k: v for (k, v) in subtask_args.items() if k != "metrics"}
                    )
                    # make paths for source, hyp and ref, given task and subtask parameters
                    # different tasks have different source and reference file types
                    hypothesis_path, gold_data_path, eval_args = (
                        get_eval_args_given_task(
                            eval_args,
                            task_name,
                            data_dir,
                            subtask,
                            output_dir,
                            model_type,
                            model_name,
                        )
                    )
                    metric_score = None
                    if Path(output_path).exists():
                        subtask_results = load_json_file(output_path)
                        metric_score = subtask_results.get(task_metric)
                    # metric_score is None if the output file does't exist or if it doesn't contain the scores of this metric.
                    # In any of those cases we will need to run the metric.
                    if metric_score is None:
                        metric_result = instantiated_metric.run(
                            hypothesis_path=hypothesis_path,
                            gold_data_path=gold_data_path,
                            **eval_args,
                        )
                        subtask_results.update(metric_result)
                    save_to_json(
                        save_location=output_path,
                        data=subtask_results,
                    )
                    # save run metadata to the same path for better experiment tracking
                    save_to_json(
                        save_location=Path(output_path).parent / "metadata.json",
                        data=config_to_save,
                    )
            task_results.update({subtask: subtask_results})

        all_scores.update({task_name: task_results})

    return all_scores


def run_data_preparation(config: dict) -> None:
    """
    Function to prepare the test data with the given prompt so it can be directly used for generative LLMs.
    """
    output_dir_root = Path(config.get("output_dir"))
    output_dir_root.mkdir(parents=True, exist_ok=True)
    data_dir = config.get("data_dir", "")
    indexed_data_dir = config.get("index_dir")
    # set random seed for later sampling
    seed = config.get("seed", 42)
    for task in config.get("tasks"):
        seed = task.get("seed", seed)
        task_name = task.get("name")
        # Get reference column name to later save
        prompt_templates = task.get("prompt_templates")
        # get fewshot args
        n_fewshots = task.get("n_fewshots", 0)
        fewshot_retrieval_method = task.get("fewshot_retrieval_method", None)
        fewshot_retrieval_args = task.get("fewshot_retrieval_args", {})
        subtasks = task.get("subtasks")
        for subtask_name, subtask_args in subtasks.items():
            subtask_args = {} if subtask_args is None else subtask_args
            # Set random seed per task, so that we can change it by task, while reproducing everything else, if needeed
            seed = subtask_args.get("seed", seed)
            random.seed(seed)
            np.random.seed(seed)
            # subtask prompt template overwrites task prompt template (e.g., useful for noisy translation)
            prompt_templates = subtask_args.get("prompt_templates", prompt_templates)
            # specific arguments for the subtask's prompts
            prompt_args = subtask_args.get("prompt_args", {})
            # get data paths; we assume a specific directory structure
            test_data_filename = "test.jsonl"
            fewshot_data_filename = "dev.jsonl"
            test_data_path = (
                f"{data_dir}/{task_name}/{subtask_name}/{test_data_filename}"
            )
            datastore_index_path = None
            datastore_data_path = None
            if task.get("n_fewshots", 0) > 0:
                if fewshot_retrieval_method == "similarity":
                    # We have a index path were we store the indexed data. Additionally, we do have the data_path in which
                    # we store the (src, ref) pairs that will be used to retrieve the similar samples based on the indecies obtained from the index file
                    datastore_index_path = (
                        f"{indexed_data_dir}/{task_name}/{subtask_name}"
                    )
                datastore_data_path = (
                    f"{data_dir}/{task_name}/{subtask_name}/{fewshot_data_filename}"
                )
                # overwrite datastore path name if specific path is passed (e.g., we wish to use a different dataset)
                datastore_data_path = subtask_args.get(
                    "datastore_data_path", datastore_data_path
                )
                # overwrite indexed data path name if specific path is passed (e.g., we wish to use a different dataset)
                datastore_index_path = subtask_args.get(
                    "index_dir", datastore_index_path
                )

            prepare_data(
                prompt_templates=prompt_templates,
                prompt_args=prompt_args,
                test_data_path=test_data_path,
                datastore_data_path=datastore_data_path,
                n_fewshots=n_fewshots,
                fewshot_retrieval_method=fewshot_retrieval_method,
                fewshot_retrieval_args=fewshot_retrieval_args,
                task_name=task_name,
                subtask_name=subtask_name,
                datastore_index_path=datastore_index_path,
                output_dir=output_dir_root,
            )


def run_index(config: dict) -> None:
    """
    Function to index the data to be used for the similarity-based retrieval.
    """
    output_dir_root = Path(config.get("output_dir"))
    output_dir_root.mkdir(parents=True, exist_ok=True)
    data_dir = config.get("data_dir", "")

    seed = config.get("seed", 42)
    for task in config.get("tasks"):
        seed = task.get("seed", seed)
        task_name = task.get("name")
        subtasks = task.get("subtasks")
        # whether the data objects are in jsonl format; this requires special handling
        jsonl = task.get("jsonl", False)
        for subtask_name, subtask_args in subtasks.items():
            subtask_args = {} if subtask_args is None else subtask_args
            # get data paths; we assume a specific directory structure, like for generate and evaluate commands
            datastore_filename = "dev.jsonl"
            datastore_path = (
                f"{data_dir}/{task_name}/{subtask_name}/{datastore_filename}"
            )
            datastore_indexed_path = f"{output_dir_root}/{task_name}/{subtask_name}"
            index_data(
                datastore_filename=datastore_path,
                datastore_indexed_path=datastore_indexed_path,
                task_name=task_name,
                subtask_name=subtask_name,
                jsonl=jsonl,
            )


def run_generations(
    configs: dict,
    config_path: str,
    config_type: str,
    available_models: dict = available_models,
) -> dict:
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    )
    models = configs.get("models")
    for i, model in enumerate(models):
        model_args = model.get("arguments")
        model_args = {} if not model_args else model_args
        # We need to handle the VLLM processes with subprocesses
        if model["type"] == "vllm":
            subprocess_args = [
                f"python",
                "-m",
                "tower_eval.tasks.generate",
                "--i",
                f"{str(i)}",
                "--config_path",
                f"{config_path}",
                "--config_type",
                f"{config_type}",
            ]
            failure = handle_subprocess(subprocess_args)
            if failure:
                logger.error(
                    f"{model['name']} has run into an error. Double check generations before running evaluations."
                )
        else:
            try:
                # All the models except VLLM can be easily executed without requiring the subprocesses.
                generate(i, config_path, config_type, available_models)
            except Exception as e:
                print(e)
                logger.error(
                    f"{model['name']} has run into an error. Double check generations before running evaluations."
                )


def command_selector(
    args, available_metrics=available_metrics, available_models=available_models
):
    if args.command == "evaluate":
        # Either read the information from the config file or directly from the commandline
        # NOTE: Overwriting the parameters of the config file by the values provided via commandline is not supported
        if args.config:
            config_args = parse_yaml_config(args.config)
            scores = run_evaluations(config_args, available_metrics=available_metrics)
        else:
            paths_scores_correspondence = {o: {} for o in args.output_paths}
            for metric in args.metrics:
                eval_args = args.eval_args.get(metric, {})
                metric = available_metrics[metric](**(eval_args))
                logger.opt(colors=True).info(
                    f"Running metric: <green> {metric.metric_name()} </green>"
                )
                assert (
                    len(args.output_paths)
                    == len(args.raw_data_paths)
                    == len(args.generations_paths)
                ), "The number of output directories, raw data paths and generations paths should be the same."
                for output_path, raw_data_path, generations_path in zip(
                    args.output_paths, args.raw_data_paths, args.generations_paths
                ):
                    paths_scores_correspondence[output_path].update(
                        metric.run(
                            hypothesis_path=generations_path,
                            gold_data_path=raw_data_path,
                            **eval_args,
                        )
                    )
                    save_to_json(
                        save_location=output_path,
                        data=paths_scores_correspondence[output_path],
                    )
    elif args.command == "index":
        if args.config:
            config_args = parse_yaml_config(args.config)
            run_index(config_args)
    elif args.command == "prepare":
        if args.config:
            config_args = parse_yaml_config(args.config)
            run_data_preparation(config_args)

    elif args.command == "generate":
        if args.config:
            config_args = parse_yaml_config(args.config)
            run_generations(
                config_args,
                args.config,
                config_type="generate",
                available_models=available_models,
            )
        else:
            assert (
                len(args.input_paths)
                == len(args.output_paths)
                == len(args.metadata_file_paths)
            ), "The number of input, output, and metadata paths should be the same."
            simple_generate(
                args.input_paths,
                args.output_paths,
                args.model_path,
                args.model_type,
                args.model_args,
                args.metadata_file_paths,
                available_models,
            )

    elif args.command == "gen-eval":
        if args.config:
            config_args = parse_yaml_config(args.config)
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
                                task_dict["subtasks"][subtask][arg] = subtask_dict[
                                    subtask
                                ][f"{step}_args"][arg]
                        task_dict["metrics"] = task.get("metrics")
                configs[step]["tasks"].append(task_dict)
            # create model structure
            configs[step]["models"] = []
            for model_dict in config_args.get("models"):
                # add hypothesis file to eval config
                if step == "eval":
                    model_dict["hypothesis_dir"] = config_args["gen_output_dir"]
                configs[step]["models"].append(model_dict)
        run_generations(
            configs["gen"],
            args.config,
            config_type="gen-eval",
            available_models=available_models,
        )
        run_evaluations(configs["eval"], available_metrics=available_metrics)
    else:
        logger.error(f"{args.command} is not supported!")


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Calculates the scores of the given hypothesis for the given task",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # CONFIG CLI ARGS
    parser.add_argument(
        "command",
        choices=["index", "prepare", "evaluate", "generate", "gen-eval"],
        help="Determines the command that you want to run."
        "you can prepare the test set so that the sentences are in the format of your prompt (prepare), or"
        "generate and evaluate a generated hypothesis (generate, evaluate, gen-eval).",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        required=False,
        help="Path to the yaml file to read all the necessary information from."
        "NOTE: Overwriting the parameters of the config file by the values provided via commandline is NOT supported",
    )
    # GENERATE CLI ARGS
    parser.add_argument(
        "--input_paths",
        "-ip",
        type=Path,
        nargs="+",
        default=None,
        help="Path to the input file.",
    )
    parser.add_argument(
        "--output_paths",
        "-op",
        type=Path,
        nargs="+",
        default=None,
        help="Path to the output file.",
    )
    parser.add_argument(
        "--model_path",
        "-mp",
        type=str,
        default=None,
        help="Path to the model to use for generation.",
    )
    parser.add_argument(
        "--model_type",
        "-mt",
        type=str,
        default=None,
        help="Type of the model to use for generation.",
    )
    parser.add_argument(
        "--model_args",
        "-ma",
        type=parse_dict_arg,
        default=None,
        help="Model arguments dictionary.",
    )
    parser.add_argument(
        "--metadata_file_paths",
        "-mfp",
        type=Path,
        nargs="+",
        default=None,
    )
    # EVALUATE CLI ARGS
    parser.add_argument(
        "--raw_data_paths",
        "-rdp",
        type=Path,
        nargs="+",
        default=None,
        help="Path to raw data jsonl file.",
    )
    parser.add_argument(
        "--generations_paths",
        "-gp",
        type=Path,
        nargs="+",
        default=None,
        help="Path to generations txt file (1 generation per file).",
    )
    parser.add_argument(
        "--metrics",
        "-m",
        type=str,
        nargs="+",
        default=None,
        help="Metric to use for evaluating the generation.",
    )
    parser.add_argument(
        "--eval_args",
        "-ea",
        type=parse_dict_arg,
        default={},
        help="Evaluation arguments dictionary.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = argument_parser()
    command_selector(
        args, available_metrics=available_metrics, available_models=available_models
    )
