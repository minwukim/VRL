# --- run_all.sh ----------------------------------------------------
for gpu in $(seq 0 9); do
  CUDA_VISIBLE_DEVICES=$gpu \
  nohup python vllm_data_collection.py --gpu-id $gpu \
        > 15b_math_oat_log_gpu${gpu}_UPDATE.txt 2>&1 &
done
wait
# -------------------------------------------------------------------
