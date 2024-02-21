#!/bin/bash
echo "Running WMT23."
python -m tower_eval.cli gen-eval --config configs/reference_benchmark/0_shot_wmt23.yaml

echo "Running standard multilingual benchmarks."
python -m tower_eval.cli lm_eval --config configs/reference_benchmark/standard_benchmarks.yaml