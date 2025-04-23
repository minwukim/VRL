# --- run_all.sh ----------------------------------------------------
for gpu in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$gpu \
  nohup python generate_math_train_test.py --gpu-id $gpu \
        > log_gpu${gpu}.txt 2>&1 &
done
wait
# -------------------------------------------------------------------
