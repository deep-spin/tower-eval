#!/bin/bash

OPENAI=false

while (( "$#" )); do
  case "$1" in
    --openai)
      OPENAI=true
      shift
      ;;
    --) # end argument parsing
      shift
      break
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *) # preserve positional arguments
      PARAMS="$PARAMS $1"
      shift
      ;;
  esac
done

if [ "$OPENAI" = true ] ; then
    configs=(configs/tower_paper/0_shot_openai.yaml configs/tower_paper/5_shot_generic_models.yaml configs/tower_paper/5_shot_openai.yaml configs/tower_paper/tower_instruct_0_shot.yaml configs/tower_paper/tower_instruct_5_shot.yaml)
    echo "Running Tower paper benchmark including open-ai models."
else
    configs=(configs/tower_paper/5_shot_generic_models.yaml configs/tower_paper/tower_instruct_0_shot.yaml configs/tower_paper/tower_instruct_5_shot.yaml)
    echo "Running Tower paper benchmark for open models only."
fi

for config in "${configs[@]}"; do
    echo "Running $config"
    python -m tower_eval.cli gen-eval --config $config
done