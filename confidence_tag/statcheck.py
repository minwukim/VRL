import pandas as pd

df_inter = pd.read_csv("aime25_results_vllm_interrupted_qwen3_32B_0827.csv")
df_normal = pd.read_csv("aime25_results_vllm_normal_qwen3_32B_0827.csv")

print("interrupted df shape: ", df_inter.shape)
print("normal df shape: ", df_normal.shape)
to_check = df_normal.shape[0]
print("interrupted tokens: ", df_inter[:to_check]['token_count'].mean())
print("normal tokens: ", df_normal[:to_check]['token_count'].mean())
print("interrupted reward: ", df_inter[:to_check]['reward'].mean())
print("normal reward: ", df_normal[:to_check]['reward'].mean())