nohup python run_evaluation.py --input llama_input.csv \
	--run run1 \
	--out_file llama_evals.csv \
	--trials 4 > llamalog.out 2>&1 &

