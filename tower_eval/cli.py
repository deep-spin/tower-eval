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
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
from loguru import logger

from tower_eval.metrics import available_metrics
from tower_eval.metrics.comet import DEFAULT_COMET_MODEL
from tower_eval.models import available_models
from tower_eval.tasks.evaluate import run_metric
from tower_eval.tasks.index import index_data
from tower_eval.tasks.prepare import prepare_data
from tower_eval.tools.logging import wandb_utils
from tower_eval.utils import (
    combine_metrics_args,
    get_eval_args_given_task,
    make_dir_if_not_exists,
    parse_yaml_config,
    save_to_json,
)


def run_evaluations(configs: dict, wandb_project_name: str = None) -> dict:
    if wandb_project_name is not None:
        logger.opt(colors=True).info(
            f"<green>Logging</green> results in WandB project: <green>{wandb_project_name}</green>"
        )
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
    for model in configs.get("models"):
        model_type = model["type"]
        model_name = model["name"]
        for task in configs.get("tasks"):
            task_results = {}
            task_name = task.get("name")
            subtasks = task.get("subtasks")
            task_metrics = task.get("metrics")
            for subtask, subtask_args in subtasks.items():
                subtask_results = {}
                # Subtasks can overwrite/add some arguments to the metric argument defined on the task level
                # For example, we should be able to define a specific tokenizer for Chinese only
                if subtask_args is None:
                    subtask_args = {}
                subtask_metrics = (
                    subtask_args.get("metrics", {}) if subtask_args is not None else {}
                )
                output_path = (
                    output_dir
                    / task_name
                    / subtask
                    / model["type"]
                    / model["name"]
                    / "evaluation.json"
                )
                dataset_name, language = subtask.split(".")
                for task_metric, task_metric_args in task_metrics.items():
                    logger.opt(colors=True).info(
                        f"Evaluating the results of model <green> {model_name} </green> on task: <yellow> {task_name} </yellow>, subtask: <green> {subtask} </green> with metric: <red> {task_metric} </red>"
                    )
                    # Use empty dictionary if the arguments is None.
                    # This is done to be able to use update function later on.
                    task_metric_args = {} if not task_metric_args else task_metric_args
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
                    eval_args = get_eval_args_given_task(
                        eval_args,
                        task_name,
                        data_dir,
                        subtask,
                        output_dir,
                        model_type,
                        model_name,
                    )
                    metric_score = run_metric(
                        metric_name=task_metric, eval_args=eval_args
                    )
                    subtask_results.update(metric_score)
                    if wandb_project_name is not None:
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
                        ) = wandb_utils.create_wandb_names_from_path(output_path)
                        wandb_utils.log_one_entry(
                            project=wandb_project_name,
                            model_name_and_setting=model_name_and_setting,
                            shot_name_short=shot_name_short,
                            task=task,
                            subtask=subtask,
                            dataset_name=dataset_name,
                            lp_xor_language=lp_xor_language,
                            metric=task_metric,
                            score=metric_score[task_metric],
                            model_type=model_type,
                            model=model_name,
                            shot_setting=shot_setting,
                        )
                if Path(output_path).exists():
                    existing_results = json.load(open(output_path, "r"))
                    subtask_results.update(existing_results)
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


def run_generations(configs: dict, config_path: str, config_type: str) -> dict:
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
        current_dir = os.getcwd()
        try:
            process = subprocess.Popen(
                [
                    f"python",
                    f"{current_dir}/tower_eval/tasks/generate.py",
                    "--i",
                    f"{str(i)}",
                    "--config_path",
                    f"{config_path}",
                    "--config_type",
                    f"{config_type}",
                ]
            )
            while process.poll() is None:
                time.sleep(1)
            logger.info("Generation process has finished.")
        except KeyboardInterrupt:
            process.terminate()


def command_selector(args):
    if args.command == "evaluate":
        # Either read the information from the config file or directly from the commandline
        # NOTE: Overwriting the parameters of the config file by the values provided via commandline is not supported
        if args.config:
            config_args = parse_yaml_config(args.config)
            scores = run_evaluations(
                config_args, wandb_project_name=args.wandb_project_name
            )
            # Store the scores of all the tasks and their corresponding metrics in a json file (default: stdout)
            logger.opt(colors=True).info(f"Evaluations finished. Saving the results.")
            args.scores_output_file.write(json.dumps(scores, indent=4) + "\n")
        else:
            output_path = os.path.join(
                args.output_dir,
                args.task,
                args.subtask,
                args.model_type,
                args.model_name,
                "evaluation.json",
            )
            metric_scores = {}
            for metric in args.metrics:
                metric_scores.update(
                    run_metric(metric_name=metric, eval_args=vars(args))
                )
                save_to_json(save_location=output_path, data=metric_scores)
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
            run_generations(config_args, args.config, config_type="generate")
        else:
            output_file = os.path.join(
                args.output_dir,
                args.task,
                args.subtask,
                args.model_type,
                args.model_name,
                "generation.txt",
            )
            model = available_models[args.model_type](**vars(args))
            make_dir_if_not_exists(output_file)
            logger.opt(colors=True).info(
                f"Running inference for task: <yellow> {args.task} </yellow>, subtask: <green> {args.subtask} </green> with model: <red> {args.model_type}/{args.model_name} </red> saving to: <red> {args.output_dir} </red>"
            )
            model.generation_with_resume(
                input_file=args.source, output_file=output_file
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
        wandb_project_name = args.wandb_project_name
        run_generations(configs["gen"], args.config, config_type="gen-eval")
        run_evaluations(configs["eval"], wandb_project_name=wandb_project_name)
    else:
        print(f"ERROR: {args['command']} is not supported, yet.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculates the scores of the given hypothesis for the given task",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
    parser.add_argument(
        "--log-wandb",
        action="store_true",
        required=False,
        help="Set to True if you want to have the results logged by WandB.",
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default=None,
        required=False,
        help="The name of the WandB project to store the evaluation results",
    )
    parser.add_argument(
        "--scores-output-file",
        "-o",
        type=argparse.FileType("w"),
        default=sys.stdout,
        required=False,
        help="Path to the output json file to store the results of all the tasks.",
    )
    parser.add_argument(
        "--output_dir", "-od", type=Path, default=None, help="Output directory."
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--subtask",
        "-st",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--hypothesis",
        "-hyp",
        type=Path,
        required=False,
        help="Path to the hypotheses file.",
    )
    parser.add_argument(
        "--references",
        "-ref",
        type=Path,
        nargs="+",
        required=False,
        help="Path to the reference file.",
    )
    parser.add_argument(
        "--source",
        "-src",
        type=Path,
        required=False,
        help="Path to the source file.",
    )
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        required=False,
        default="en",
        help="The language of the hypotheses",
    )
    parser.add_argument(
        "--metrics",
        "-m",
        nargs="+",
        choices=[m.metric_name() for m in available_metrics.values()].__add__(["all"]),
        required=False,
        help="Metric to use for evaluating the hypothesis.",
    )
    parser.add_argument(
        "--tokenizer",
        required=False,
        default=None,
        choices=[None, "zh", "13a", "char", "intl", "ja-mecab", "ko-mecab"],
        help="The tokenizer to apply to the hypotheses and the reference before sending to evaluation.",
    )
    parser.add_argument(
        "--lowercase",
        required=False,
        default=False,
        action="store_true",
        help="If True, enables case-insensitivity in sacrebleu metrics (chrF, BLEU).",
    )
    chrf_args = parser.add_argument_group("ChrF related arguments")
    chrf_args.add_argument(
        "--whitespace",
        action="store_true",
        default=False,
        help="Include whitespaces when extracting character n-grams. (Default: %(default)s)",
    )

    ter_args = parser.add_argument_group("TER related arguments")
    ter_args.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Enables case sensitivity. (Default: %(default)s)",
    )
    ter_args.add_argument(
        "--asian-support",
        action="store_true",
        help="Enables special treatment of Asian characters. (Default: %(default)s)",
    )
    ter_args.add_argument(
        "--no-punct",
        action="store_true",
        help="Removes punctuation. (Default: %(default)s)",
    )
    ter_args.add_argument(
        "--normalized",
        action="store_true",
        help="Applies basic normalization and tokenization. (Default: %(default)s)",
    )
    comet_args = parser.add_argument_group("COMET related arguments")
    comet_args.add_argument("--batch-size", type=int, default=16)
    comet_args.add_argument("--gpus", type=int, default=1)
    comet_args.add_argument(
        "--comet_model",
        type=str,
        required=False,
        default=DEFAULT_COMET_MODEL,
        help="COMET model to be used.",
    )
    errant_args = parser.add_argument_group("ERRANT related arguments")
    errant_args.add_argument(
        "--tokenize-source",
        action="store_true",
        help="Tokenize the original source file. (Default: %(default)s)",
    )
    errant_args.add_argument(
        "--tokenize-hypothesis",
        action="store_true",
        help="Tokenize the hypothesis file. (Default: %(default)s)",
    )
    error_span_detection_args = parser.add_argument_group(
        "Error Span Detection related arguments"
    )
    error_span_detection_args.add_argument(
        "--severity-mismatch-penalty",
        type=float,
        default=0.5,
        help="Penalty for severity mismatch. (Default: %(default)s)",
    )
    error_span_detection_args.add_argument(
        "--hyp_type",
        type=str,
        default="jsonl",
        choices=["jsonl", "tag", "det"],
        help="Hypothesis file format. (Default: %(default)s)",
    )
    # model args for generation
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Type of the model to use for inference.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Type of the model to use for inference.",
    )
    model_args = parser.add_argument_group("Model inference-related arguments")
    model_args.add_argument(
        "--api_base",
        type=str,
        default=None,
        help="Server address of the model to use for inference.",
    )
    model_args.add_argument(
        "--openai_model",
        type=str,
        default=None,
        choices=["gpt-3.5-turbo", "gpt-4"],
        help="Specific model to be used (open-ai).",
    )
    model_args.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate. (Default: %(default)s)",
    )
    args = parser.parse_args()
    if (
        args.command in ["evaluate", "gen-eval"]
        and args.log_wandb
        and args.wandb_project_name is None
    ):
        parser.error(
            "--wandb-project-name is mandatory for logging the results in wandb.\n"
            "So, either provide the wandb-project-name or remove the --log-wandb option.\n"
            "NOTE: in the latter case, the results will not be logged in wandb."
        )

    command_selector(args)
