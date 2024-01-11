while getopts m:p:e:g:i:t:c:q: argument
do
    case "${argument}" in
        m) model=${OPTARG};;
        p) port=${OPTARG};;
        e) env=${OPTARG};;
        g) gpu=${OPTARG};;
        i) max_input_len=${OPTARG};;
        t) max_total_len=${OPTARG};;
        c) conda_path=${OPTARG};;
        q) quantize=${OPTARG};;
    esac
done

export CONDA="${conda_path}/bin"
export CONDA_PREFIX=${conda_path}
export PATH=$CONDA:$PATH
export CONDA_PROFILE="${conda_path}/etc/profile.d/conda.sh"
source $CONDA_PROFILE 2>/dev/null
echo $CONDA_PREFIX

if [ "${quantize}" != "none" ]; then
    quantize_string="--quantize ${quantize}"
else
    quantize_string=""
fi

# set master_port to integer of variable port plus 29500
# this is to avoid port collisions with other models that require >1 gpus
master_port=$((port+29500))

conda activate ${env}
export CUDA_VISIBLE_DEVICES=${gpu}
text-generation-launcher --model-id ${model} \
                         --port ${port} \
                         --max-input-length ${max_input_len} \
                         --max-total-tokens ${max_total_len} \
                         ${quantize_string} \
                         --shard-uds-path "/tmp/text-generation-server-23" \
                         --master-port ${master_port} \
