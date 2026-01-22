## With PM Reward
accelerate launch \
    --config_file configs/deepspeed_zero3.yaml \
    gspotrain.py \
    --dataset_name DeepMath-103k\
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B\
    --output_dir Result/Qwen-7B_TACReward \
    --logging_dir ./logs \
    --push_to_hub True \
    --save_safetensors True \
    --learning_rate 1e-6 \
    --torch_dtype bfloat16 \
    --max_prompt_length 1024 \
    --max_completion_length 16384 \
    --log_completions True \
    --save_strategy steps \
    --use_vllm True \
    --vllm_mode colocate \
    --save_steps 100 \
    --max_steps 1000 \
    --per_device_train_batch_size 2 \
    --num_generations 8 \
    --gradient_accumulation_steps 8 \
    --attn_implementation=flash_attention_2\
    --importance_sampling_level sequence \
