nohup python run_evaluation.py --input mmlu_input.csv \
	--run run1 \
	--out_file mmlu_evals.csv \
	--trials 4 > mmlulog.out 2>&1 &
