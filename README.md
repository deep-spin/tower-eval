# tower-eval
Repository for evaluation of LLMs on MT and related tasks. `tower-eval` also supports generation with `vllm` and the creation of custom test suites and instructions. 

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
>Note: Some slight differences between our reported results are expected.

## Usage Guide

Work in progress. This section will cover how `tower-eval` can be used to perform generations and evaluations with ours, or your own test data.