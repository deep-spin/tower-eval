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
    configs=(configs/blogpost/tower_instruct_0_shot.yaml configs/blogpost/tower_instruct_5_shot.yaml configs/blogpost/5_shot_ape.yaml configs/blogpost/5_shot_gec.yaml configs/blogpost/5_shot_mt.yaml configs/blogpost/openai_ape.yaml configs/blogpost/openai_gec.yaml configs/blogpost/openai_mt.yaml configs/blogpost/openai_ner.yaml)
    echo "Running Tower blogpost benchmark including open-ai models."
else
    configs=(configs/blogpost/tower_instruct_0_shot.yaml configs/blogpost/tower_instruct_5_shot.yaml configs/blogpost/5_shot_ape.yaml configs/blogpost/5_shot_gec.yaml configs/blogpost/5_shot_mt.yaml)
    echo "Running Tower blogpost benchmark for open models only."
fi

for config in "${configs[@]}"; do
    echo "Running $config"
    python -m tower_eval.cli gen-eval --config $config
done