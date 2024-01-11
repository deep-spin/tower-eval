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
    configs=(     
      configs/blogpost/tower_instruct/tower_instruct_0_shot.yaml 
      configs/blogpost/tower_instruct/tower_instruct_5_shot.yaml
      configs/blogpost/5_shot/ape/5_shot_ape_llama_13b.yaml
      configs/blogpost/5_shot/ape/5_shot_ape_llama_70b.yaml
      configs/blogpost/5_shot/ape/5_shot_ape_llama_7b.yaml
      configs/blogpost/5_shot/ape/5_shot_ape_mixtral_8x7b.yaml
      configs/blogpost/5_shot/ape/5_shot_ape_tower_base.yaml
      configs/blogpost/5_shot/gec/5_shot_gec_llama_13b.yaml
      configs/blogpost/5_shot/gec/5_shot_gec_llama_70b.yaml
      configs/blogpost/5_shot/gec/5_shot_gec_mixtral_8x7b.yaml
      configs/blogpost/5_shot/gec/5_shot_gec_llama_7b.yaml
      configs/blogpost/5_shot/gec/5_shot_gec_tower_base.yaml
      configs/blogpost/5_shot/mt/5_shot_mt_alma_7b.yaml
      configs/blogpost/5_shot/mt/5_shot_mt_alma_13b.yaml
      configs/blogpost/5_shot/mt/5_shot_mt_llama_13b.yaml
      configs/blogpost/5_shot/mt/5_shot_mt_llama_70b.yaml
      configs/blogpost/5_shot/mt/5_shot_mt_mixtral_8x7b.yaml
      configs/blogpost/5_shot/mt/5_shot_mt_llama_7b.yaml
      configs/blogpost/5_shot/mt/5_shot_mt_tower_base.yaml  
      configs/blogpost/open_ai/openai_ape_gpt_4.yaml 
      configs/blogpost/open_ai/openai_ape.yaml 
      configs/blogpost/open_ai/openai_gec_gpt_4.yaml 
      configs/blogpost/open_ai/openai_gec.yaml 
      configs/blogpost/open_ai/openai_mt_gpt_4.yaml 
      configs/blogpost/open_ai/openai_mt.yaml 
      configs/blogpost/open_ai/openai_ner_gpt_4.yaml 
      configs/blogpost/open_ai/openai_ner.yaml
    )
    echo "Running Tower blogpost benchmark including open-ai models."
else
    configs=(     
      configs/blogpost/tower_instruct/tower_instruct_0_shot.yaml 
      configs/blogpost/tower_instruct/tower_instruct_5_shot.yaml
      configs/blogpost/5_shot/ape/5_shot_ape_llama_13b.yaml
      configs/blogpost/5_shot/ape/5_shot_ape_llama_70b.yaml
      configs/blogpost/5_shot/ape/5_shot_ape_llama_7b.yaml
      configs/blogpost/5_shot/ape/5_shot_ape_mixtral_8x7b.yaml
      configs/blogpost/5_shot/ape/5_shot_ape_tower_base.yaml
      configs/blogpost/5_shot/gec/5_shot_gec_llama_13b.yaml
      configs/blogpost/5_shot/gec/5_shot_gec_llama_70b.yaml
      configs/blogpost/5_shot/gec/5_shot_gec_mixtral_8x7b.yaml
      configs/blogpost/5_shot/gec/5_shot_gec_llama_7b.yaml
      configs/blogpost/5_shot/gec/5_shot_gec_tower_base.yaml
      configs/blogpost/5_shot/mt/5_shot_mt_alma_7b.yaml
      configs/blogpost/5_shot/mt/5_shot_mt_alma_13b.yaml
      configs/blogpost/5_shot/mt/5_shot_mt_llama_13b.yaml
      configs/blogpost/5_shot/mt/5_shot_mt_llama_70b.yaml
      configs/blogpost/5_shot/mt/5_shot_mt_mixtral_8x7b.yaml
      configs/blogpost/5_shot/mt/5_shot_mt_llama_7b.yaml
      configs/blogpost/5_shot/mt/5_shot_mt_tower_base.yaml
    )
    echo "Running Tower blogpost benchmark for open models only."
fi

for config in "${configs[@]}"; do
    echo "Running $config"
    python -m tower_eval.cli gen-eval --config $config
done