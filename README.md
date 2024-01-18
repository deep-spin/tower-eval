# tower-eval
Repository for evaluation of LLMs on MT and related tasks. `tower-eval` also supports generation with [`vllm`](https://github.com/vllm-project/vllm) and the creation of custom test suites and instructions. 

## Contents

- [Installation](#installation)
- [Replicating our benchmarks](#replicating-our-benchmarks)
- [Usage Guide](#usage-guide)
  - [Generation](#run-inference)
  - [Evaluation](#evaluate-outputs)
  - [Generation followed by Evaluation](#run-inference-and-evaluation-consecutively)
  - [Prepare your own test instructions](#preparing-your-own-test-instructions)


## Installation

To install the package first clone the project by:
```sh
git clone https://github.com/deep-spin/tower-eval.git
```

Create a virtual env with (outside the project folder):
```bash
python -m venv tower-eval-env
source tower-eval-env/bin/activate
cd tower-eval
```

To install the project's dependencies, run:

```sh
poetry install
```
>Python 3.10 and Poetry 1.6.1 are known to work; Poetry 1.7.1 is known to **not** work.

and 

```sh
pip install vllm
```

## Replicating our benchmarks

First, download the test data:

```sh
huggingface-cli download Unbabel/TowerEval-Data-v0.1 --repo-type dataset --local-dir TowerEval-Data-v0.1
tar -xzf TowerEval-Data-v0.1/data.tar.gz -C TowerEval-Data-v0.1/
```

To replicate the results in the [blogpost](https://unbabel.com/announcing-tower-an-open-multilingual-llm-for-translation-related-tasks/) about [Tower](https://huggingface.co/collections/Unbabel/tower-7b-v01-659eaedfe36e6dd29eb1805c), run:

```sh
bash run_blogpost_benchmark.sh
```
>Note: some slight differences between our reported results are expected.

To run a new model, first write an entry under the `models` key in any of the configs inside `configs/blogpost`:

```yaml
models:
  ...
  - name: <model_name> # folder name to store generations and evaluations
    type: vllm
    arguments:
      model_dir: <path_to_huggingface_model>
      n_gpus: 1
      max_tokens: 1024
      run_async: True
      batch_size: -1
```
>Note: the model architecture must be supported by vllm.

For example, if you want to test your model on our 5-shot MT setting, add the corresponding entry to `configs/blogpost/5_shot_mt.yaml` and run `python -m tower_eval.cli gen-eval --config configs/blogpost/5_shot_mt.yaml`.

More details on general usage in the next section.

## Usage Guide

### Download the test sets

Run:

```sh
huggingface-cli download Unbabel/TowerEval-Data-v0.1 --repo-type dataset --local-dir TowerEval-Data-v0.1
tar -xzf TowerEval-Data-v0.1/data.tar.gz -C TowerEval-Data-v0.1/
```

The test data used by the model for generation will be under `TowerEval-Data-v0.1/data/instructions_data`.
`TowerEval-Data-v0.1/data/raw_data` contains data formatted for evaluation, or the creation of new instructions.

The dataset has the following structure:

```
TowerEval-Data-v0.1
    instructions_data
        prompt_format
            task_1
                subtask_1
                    instructions.txt
                subtask_2
                    ...
    raw_data
      task_1
        subtask_1
          dev.jsonl
          test.jsonl
        subtask_2
          ...
```


<a name="Generation"></a>
### Run inference

Once you have the data ready you can use tower-eval to run inference using the `generate` command:

```bash
python -m tower_eval.cli generate --config <ABSOLUTE_PATH_TO_CONFIG>
```
>We `vllm` for fast inference.

The config is a yaml file, which should have the following format:

```yaml
data_dir: <DATA_DIR>
output_dir: <OUTPUT_DIR>
tasks: 
  - name: <TASK_1>
    subtasks:
      <SUBTASK_1>:
      <SUBTASK_2>:
      <SUBTASK_3>:
      <SUBTASK_4>:
  - name: <TASK_2>
  ...
models:
  - name: <MODEL_NAME>
    type: <MODEL_TYPE>
    arguments:
      model_dir: <PATH_TO_HF_MODEL>
      n_gpus: <N_GPUS>
      max_tokens: 1024
      run_async: <ASYNC>
      batch_size: <BSZ>
  - name: <MODEL_2>
  ...
```
>Note: **don't forget the colon when enumerating subtasks.**

The command will, for each model under `models`:
1. Instantiate the model under `PATH_TO_HF_MODEL` with `vllm`, occupying `<N_GPUS>` GPUs, and allowing for at most `<MAX_NEW_TOKENS>` to be generated for each instance.
    1. Set `<ASYNC>` to `True` for speed improvements.
    2. Set `<BSZ>` -1 let `vllm` handle the list of prompts most efficiently (recommended).
2. Run inference for each `subtask` under each `task`.
    1. `data_dir` should be the parent directory to files containing instructions. For each task and subtask, `data_dir` **must** have the following children: `<TASK>/<SUBTASK>/instructions.txt`. For example, if `data_dir` is `/mnt/data`, then for task X and subtask Y, there should be exist a file `/mnt/data/X/Y/instructions.txt`.
3. Save outputs to `<OUTPUT_DIR>/<TASK>/<SUBTASK>/<MODEL_TYPE>/<MODEL_NAME>/generation.txt` (each line maps to an instance in `instructions.txt`).

Currently available metrics are:
`vllm`, `open-ai`, and respective arguments (check [Models section](#ModelsSection)).

You can find a sample config file of the generate task in `configs/examples/generate.yaml`.


### Evaluate Outputs
To evaluate outputs, use the following command:

```bash
python -m tower_eval.cli evaluate --config <PATH_TO_CONFIG>
```
The config is a yaml file, which should have the following format:

```yaml
data_dir: <DATA_DIR>
output_dir: <OUTPUT_DIR>
tasks:
  - name: mt
    subtasks:
      <SUBTASK_1>:
      <SUBTASK_2>:
        metrics:
          <METRIC_1>:
          <METRIC_2>:
            <ARG_1>: <SUBTASK_SPECIFIC_ARG>
      <SUBTASK_3>:
    metrics:
      <METRIC_1>:
      <METRIC_2>:
        <ARG_1>: <ARG_1>
models:
  - name: <MODEL_NAME>
    type: <MODEL_TYPE>
```
>Note: **don't forget the colon when enumerating subtasks.**

This command follows roughly the same logic as `generate`: for each model and subtask, it computes a set of `metrics`, storing the output in a `json` file at `<OUTPUT_DIR>/<TASK>/<SUBTASK>/<MODEL_TYPE>/<MODEL_NAME>/evaluation.json`.

`data_dir` should be the parent directory to files containing raw data. For each task and subtask, `data_dir` **must** have the following children: `<TASK>/<SUBTASK>/test.jsonl`. That file must also have the keys that the metric requires (e.g., COMET requires `src` and `ref` keys). The `errant` metric also requires a `test_corpus.m2` file.

`output_dir` should contain a folder called `evaluations`. This script then fetches model generations by replacing `evaluations` with `generations` in `output_dir` — `gen_dir` — and looking for files like `gen_dir/<TASK>/<SUBTASK>/<MODEL_TYPE>/<MODEL_NAME>/generation.txt`.

`metrics` can be set at the level of each task, or subtask. Keep in mind that defining metric arguments for a subtask will override the task-level metric arguments. This is useful for BLEU, for example, where the `tokenizer` argument should be different for chinese and korean.

Currently available metrics are:
`['ter', 'bleu', 'comet', 'comet_kiwi', 'chrf', 'errant', 'f1sequence']`.
For more details on the metrics and their respective arguments check [Metrics section](#MetricsSection).

An example config can be found in `configs/examples/evaluate.yaml`.


### Run inference and evaluation consecutively

You can also run generations and then evaluations automatically. The command is:

```bash
python -m tower_eval.cli gen-eval --config <PATH_TO_CONFIG>
```

The config logic is a combination of `generate` and `evaluate`, with a couple of nuances.
1. Output and data directories should be defined as:

```yaml
gen_data_dir: <GEN_DATA_DIR>
eval_data_dir: <EVAL_DATA_DIR>
gen_output_dir: <GEN_OUTPUT_DIR>
eval_output_dir: <EVAL_OUTPUT_DIR>
```

2. Inside each subtask, you can specify subtask-specific metric arguments as before like so:
```yaml
tasks: 
  - name: <TASK>
    subtasks:
      flores.en-pt:
      flores.en-zh:
        eval_args:
          metrics:
            chrf:
            bleu:
              tokenizer: zh
    metrics:
      <METRIC_1>:
      <METRIC_2>:
        <ARG_1>: <ARG_1>
models:
  - name: <MODEL_NAME>
    type: <MODEL_TYPE>
    arguments:
      ... # same as generate
  - name: <MODEL_2>
  ...
```
An example config can be found in `configs/examples/gen_eval.yaml`.

<a name="ModelsSection"></a>
### Model Arguments
Currently, tower-eval supports OpenAI models and those supported by ``vllm``.

In your config file you need to define the following parameteres:
- `name`: This field is primarily used for defining the output folder, and doesn't impact the underlying model used for inference.
- `type`: This field specifies the model type. You can set its value to `open-ai` if you want to run inference with OpenAI-based models, or `tgi` if you are going to use models supported by TGI.
- `arguments`: The additional arguments of the model (eg. the url to the remote server, temprature, etc) are defined under this category.
  - `max_tokens`: It determines the maximum number of tokens the model is supposed to generate.
  - `stop_sequences`: list of strings, which, if generated, the model will stop (will not be included in the output).
  - `do_sample`: **[vllm only]** whether to not perform greedy decoding; false by default. If set to True, temperature is set to 1.0 (can be customized). 
  - `seed`: **[vllm only]** random seed for sampling.
  - `run_async`: **[vllm only]** set to True for speed improvements.
  - `batch_size`: **[vllm only]** batch size if `run_async` is True; set to -1 to let vllm handle generations most efficiently.
  - `quantization` **[vllm only]** whether to quantize the model. See vllm docs for more information.
  - `vllm_sampling_params`: **[vllm only]** vllm sampling kwargs; see vllm docs for all the arguments you can pass.
  - `model`: **[OpenAI only]** This field is only used for the OpenAI based models and gets the following values: `gpt-3.5-turbo`, `gpt-4`.
  - `temperature`: **[OpenAI only]** This field defines the temprature that you want to use when calling OpenAI models and controls the randomness of the generation.
  - `top_p`: **[OpenAI only]** Defines the cumulative probability cutoff for token selection.
  - `frequency_penalty`: **[OpenAI only]** Controls the OpenAI models' likelihood to repeat the same line verbatim.
  - `presence_penalty`: **[OpenAI only]** Controls the OpenAI models' likelihood to use new words and topics.
  - `retry_max_attempts`: **[OpenAI only]** The maximum number of retries in case there is no response from OpenAI's generation endpoint.
  - `retry_max_interval`: **[OpenAI only]** The maximum time to wait before re-sending the request in case there is no response from OpenAI's generation endpoint.
  - `retry_min_interval`: **[OpenAI only]** The minimum time to wait before re-sending the request in case there is no response from OpenAI's generation endpoint.


<a name="MetricsSection"></a>
### Metrics
TowerEval currently supports the following metrics: COMET, COMET-Kiwi, BLEU, ChrF, TER, ERRANT (for GEC) and F1-Sequence (for sequence labeling tasks like NER).
Metrics have specific arguments like tokenization, lowercasing, etc... that can be specified in the config file.
#### COMET and COMET-Kiwi:
The arguments that COMET and COMET-Kiwi accept are:
- `lowercase`: If you want to lowercase the inputs, default: `False`
- `batch_size`: The batch size to run the evaluation, default: `16`
- `gpus`: The number of gpus that you want to run COMET on, default: `1`
- `comet_model`: The COMET model to use, default: `Unbabel/wmt22-comet-da` for COMET and `Unbabel/wmt22-cometkiwi-da` for COMET-Kiwi. Set to `Unbabel/XCOMET-XL` or `Unbabel/XCOMET-XXL` to use Unbabel's latest SotA releases.

#### BLEU:
TowerEval uses SacreBleu to calculate the BLEU scores of the hypotheses. 
The supported arguments for BLEU are:
- `lowercase`: Whether you want to lowercase the inputs, or not, default: `False`
- `tokenizer`: The tokenizer to apply on the inputs. It can be either of the following values: `[None, "zh", "13a", "char", "intl", "ja-mecab", "ko-mecab"]`. default: `None`

#### ChrF:
TowerEval uses SacreBleu to calculate the ChrF scores of the hypotheses.
The supported arguments for ChrF are:
- `lowercase`: Whether you want to lowercase the inputs, or not, default: `False`

#### TER:
TowerEval calculates the TER scores by calling SacreBLEU.
The supported argumets for TER are:
- `normalized`: Enable character normalization, default: `False`
- `no_punct`: Remove punctuation, default: `False`
- `asian_support`: Enable special treatment of Asian characters, default: `False`
- `case_sensitive`: If `True`, does not lowercase sentences, default: `False`

#### ERRANT:
The scores of the GEC models are calculated by ERRANT.
Since the source of the test sets are useually tokenized and the generative models tend to produce detokenized outputs, you might want to tokenize the hypothesis (or even the reference) before calculating the metric.
So, there are a few arguments that you might want to set for this metric:
- `tokenize_source`: Tokenize the source side of the test set, default: `False`
- `tokenize_hypothesis`: Tokenize the generated hypothesis default: `False`

#### F1-SEQUENCE:
This metric is mainly used for measuring the quality of the sequence tags produced by a sequence tagger like NER and POS-Tagger.
TowerEval uses the python implementation used for CONLL2003 SemEval shared task, and supports multiple formats for the generated hypothesis: 
- `text-tuple-list` (TowerInstruct format), where the output is a list of tuples; the first entry of each tuple is a token, and the second is its corresponding entity category. 
- `jsonl` same as above, but the file is jsonl instead of plain text.
- `xml` where the named entities are marked by XML tags
- `tsv` where each token is in a separate line along with its correspnding tag, separated by a `separator` character. There should be an empty line between the last token of sentence `S` and the first token of sentence `S+1`.
- `text` where each sentence is in a single line, with its tokens and tags separated by `separator` token.

The arguments that can be passed to F1-SEQUENCE are:
- `language`: The language of the hypothesis, mainly used for the tokenization step
- `hypothesis_format`: Determines the format of the hypothesis, and can take either of these values: `xml`, `tsv`, and `text`, default: `xml`
- `tokenize_hypothesis`: Whether you want to tokenize the hypothesis or not, default: `True`
- `default_noent_tag`: the tag to use for the no-entity tags. This is mainly used for the NER task. default: `O`
- `valid_ner_tags`: The list of valid tags for the task. If a token has a tag not listed here, it will be automatically mapped to the `default_noent_tag`.


### Preparing your own test instructions

`tower-eval` also allows you to convert raw data in a ``jsonl`` format into instructions that can be used for generation.
The command is called `prepare`, and works in a similar way to the others:

```bash
python -m tower_eval.cli prepare --config <ABSOLUTE_PATH_TO_CONFIG>
```
First, you **must** have the raw data — a `test.jsonl` file — under the following folder structure:

```
parent_folder
    task_1
        subtask_1
            test.jsonl
            dev.jsonl
        subtask_2
            ...
```

`test.jsonl` must contain keys with the information you will need in the prompts. For example, machine translation data contains source and reference keys (src, ref). `dev.jsonl` is required if you want to create few-shot data. **The files must have these names**.

The output of the command will be:

```
output_dir
    task_1
        subtask_1
            instructions.txt
        subtask_2
            ...
```

The config file must have the following structure:

```yaml
seed: <SEED>
data_dir: <RAW_DATA_DIR>
output_dir: <OUTPUT_DIR>
tasks:
  - name: task_1
    prompt_templates:
      - "<template_1>"
      - "<template_2>"
      ...
    n_fewshots: 0
    fewshot_retrieval_method: random
    fewshot_retrieval_args:
      f_arg_1: <F_ARG_1>
    subtasks:
      subtask_1:
        prompt_args:
          arg_1: <ARG_1>
          arg_2: <ARG_2>
  - name: task_2
    ...
```

- `seed` controls the random state of any sampling operation (e.g., random few-shot sampling, or sampling multiple prompt templates).
- `data_dir` is the path to the `parent_folder` of the raw data (its children should have the aforementioned folder structure).
- `output_dir` is the parent folder of where the data will be saved; the folder structure will be the same as the raw data, except the final file will be called `instructions.txt`.
- task and subtask logic is the same as previous commands.
- `prompt_templates` will be the templates used when creating instructions. If more than one is passed, they are randomly sampled uniformly. More details on the next [subsection](#creating-prompt-templates).
- `n_fewshots` is the number of few-shots the prompt should contain. If this is larger than 0, a `dev.jsonl` must exist, and the next two arguments will be considered.
- `fewshot_retrieval_method` controls how the fewshots are reetrieved for each data instance. Defaults to `random`, which corresponds to random sampling with replacement from `dev.jsonl`. There's a [section](#fewshot-retrieval-methods) on the other options.
- `fewshot_retrieval_args` are arguments for the retrieval methods.


#### Creating prompt templates

We use [jinja2](https://pypi.org/project/Jinja2/) for templating. For example, if your `test.jsonl` file has the following rows:

```json
{"col_1": "Hello", "col_2": "World"}
{"col_1": "Goodbye", "col_2": "Earth"}
```

And your template is:

```
"Please say {{ col_1 }} {{ col_2 }}."
```

The resulting instructions will be:

```
Please say Hello World.
Please say Goodbye Earth.
```

If you want extra arguments to be constant across all instances, and that are not present in the raw data, you can pass in the config:

```yaml
...
      fewshot_retrieval_args:
        arg_1: "politely"
```

Then, if the template is:

```
"Please say {{ col_1 }} {{ col_2 }} {{ arg_1 }}."
```

The output will be:

```
Please say Hello World politely.
Please say Goodbye Earth politely.
```

jinja2 allows for more complex logic, like for loops (which is what we use when there are several few-shot examples), if-else conditions, etc.... Please refer to their documentation for more details.

Our example preapre config (`configs/examples/prepare.yaml`) contains an example to recreate the 0-shot NER data and 5-shot GEC data for TowerInstruct.

#### Fewshot retrieval methods

- `random`: few-shots will be retrieved randomly from the `dev.jsonl` pool.
- `ordered`: few-shots will be retrieved in an ordered fashion from the `dev.jsonl` pool. For exampe, if `n_fewshots` is 2, the first test instance will have the first two dev instances as fewshots, the second will have the third and forth, and so on. If dev is shorter than test, we loop back to the beginning.
- `force_label_balance` can be used for tasks with name `ape` and `gec`. Will force `n_positive` exampes in the prompt that do not require correction.
- `similarity` can be used for MT. Requires an index (docs are WIP). Retrieves the examples whose source is most similar with the test instance's source.