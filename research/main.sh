
TASK_NAMES=("cls" "rec")
STRATEGIES=("adapt")
ABLATIONS=("none" "causalsimi")
TEMPLATE_MODELS=("anthropic.claude-3-5-sonnet-20240620-v1:0" "meta.llama3-1-70b-instruct-v1:0")


run_script() {
    local task=$1
    local strategy=$2
    local ablation=$3
    local template_model=$4
    
    while true; do
        echo "Running with task_name=${task}, strategy=${strategy}, ablation=${ablation}, template_model=${template_model}"
        python3 main.py \
            --task_name "${task}" \
            --strategy "${strategy}" \
            --ablation "${ablation}" \
            --template_model "${template_model}" \
            --inference_model "qwen2_5_32b_instruct"
        
        if [ $? -ne 0 ]; then
            echo "The script stopped for task=${task}, strategy=${strategy}, ablation=${ablation}, template_model=${template_model}, retrying in 10 seconds..."
            sleep 10
        else
            echo "The script completed successfully for task=${task}, strategy=${strategy}, ablation=${ablation}, template_model=${template_model}"
            break
        fi
    done
}

for task in "${TASK_NAMES[@]}"; do
    for strategy in "${STRATEGIES[@]}"; do
        for ablation in "${ABLATIONS[@]}"; do
            for template_model in "${TEMPLATE_MODELS[@]}"; do
                echo "Starting new combination: task=${task}, strategy=${strategy}, ablation=${ablation}, template_model=${template_model}"
                run_script "$task" "$strategy" "$ablation" "$template_model"
                echo "Completed combination: task=${task}, strategy=${strategy}, ablation=${ablation}, template_model=${template_model}"
                echo "----------------------------------------"
            done
        done
    done
done

echo "All combinations have been processed!"