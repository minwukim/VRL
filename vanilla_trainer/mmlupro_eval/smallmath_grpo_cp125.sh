#!/bin/bash

save_dir="eval_results/"
global_record_file="eval_results/eval_record_collection.csv"
#model="../../outputs/qwen2.5-3b-grpo-math-small/checkpoint-125"
#model="../../outputs/qwen2.5-3b-grpo-math-small/checkpoint-250"

#models starting from base trained with 20 from each category
#model="../../outputs/qwen2.5-3b-grpo-mmlu-small/checkpoint-100"
#model="../../outputs/qwen2.5-3b-grpo-mmlu-small/checkpoint-200/"
#model="../../outputs/qwen2.5-3b-grpo-mmlu-small/checkpoint-300"
#model="../../outputs/qwen2.5-3b-grpo-mmlu-small/checkpoint-350"

#model starting from sft KK then trained with mmlu100 on humanities
#model="../../outputs/qwen2.5-3b-grpo-humanities-mmlu-small/checkpoint-50"
#model="../../outputs/qwen2.5-3b-grpo-humanities-mmlu-small/checkpoint-100"
#model="../../outputs/qwen2.5-3b-grpo-humanities-mmlu-small/checkpoint-150"
#model="../../outputs/qwen2.5-3b-grpo-humanities-mmlu-small/checkpoint-300"

#model="../../outputs/qwen2.5-3b-sftkkpro-math-small/checkpoint-125"
#model="../../outputs/qwen2.5-3b-sft-pro/checkpoint-1092"
#model="../../outputs/qwen2.5-3b-grpo-test/checkpoint-540"
#model="../../outputs/qwen2.5-3b-grpo-large/checkpoint-500/"
#model="Qwen/Qwen2.5-3B"

#this is the model I trained with 75% of humanities data
#model="../../outputs/qwen2.5-3b-grpo-mmlu-full/checkpoint-444"

#model trained from base with the 100 questions each category
model="../../outputs/qwen2.5-3b-grpo-mmlu100-humanities/checkpoint-250"

#model trained from base with 100 math questions
#model="../../outputs/qwen2.5-3b-grpo-math-small/checkpoint-250"

#model trained with sftkk then math100
#model="../../outputs/qwen2.5-3b-sftkkpro-math-small/checkpoint-125"

#model trained with sftkk then history100
#model="../../outputs/qwen2.5-3b-sftkk-history100/checkpoint-50"
#model="../../outputs/qwen2.5-3b-sftkk-history100/checkpoint-100"
#model="../../outputs/qwen2.5-3b-sftkk-history100/checkpoint-150"
#model="../../outputs/qwen2.5-3b-sftkk-history100/checkpoint-200"

#model trained from base with history100
#model="../../outputs/qwen2.5-3b-grpobase-history100/checkpoint-100/"
#model="../../outputs/qwen2.5-3b-grpobase-history100/checkpoint-150/"
#model="../../outputs/qwen2.5-3b-grpobase-history100/checkpoint-200/"

#model trained from base with physics100
#model="../../outputs/qwen2.5-3b-grpobase-physics100/checkpoint-50/"
#model="../../outputs/qwen2.5-3b-grpobase-physics100/checkpoint-100"
#model="../../outputs/qwen2.5-3b-grpobase-physics100/checkpoint-150"
#model="../../outputs/qwen2.5-3b-grpobase-physics100/checkpoint-200"

#model trained from sft kk with physics100
#model="../../outputs/qwen2.5-3b-sftkk-physics100/checkpoint-50"
#model="../../outputs/qwen2.5-3b-sftkk-physics100/checkpoint-100"
#model="../../outputs/qwen2.5-3b-sftkk-physics100/checkpoint-150"
#model="../../outputs/qwen2.5-3b-sftkk-physics100/checkpoint-200"

#selected_subjects="history"
selected_subjects="law,philosophy,history"
# biology,business,economics,law,chemistry,math,physics,history,other,philosophy,psychology"
#,history,other,health,economics,math,physics,philosophy,engineering"
gpu_util=0.9

#cd ../../
export CUDA_VISIBLE_DEVICES=0


python mmlu_eval.py \
		--ntrain 0 \
                 --selected_subjects "$selected_subjects" \
                 --save_dir $save_dir \
                 --model $model \
                 --global_record_file $global_record_file \
                 --gpu_util $gpu_util

