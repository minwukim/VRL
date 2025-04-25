# --- run_all.sh ----------------------------------------------------
for gpu in $(seq 0 9); do
  CUDA_VISIBLE_DEVICES=$gpu \
  nohup python vllm_data_collection.py --gpu-id $gpu \
        > cp200_log_gpu${gpu}.txt 2>&1 &
done
wait
# -------------------------------------------------------------------
