import pandas as pd

df_inter = pd.read_csv("aime25_results_vllm_interrupted_qwen3_32B_0827.csv")
df_normal = pd.read_csv("aime25_results_vllm_normal_qwen3_32B_0827.csv")

print("interrupted df shape: ", df_inter.shape)
print("interrupted tokens: ", df_inter[:df_inter.shape[0]-df_inter.shape[0]%5]['token_count'].sum())
print("normal tokens: ", df_normal['token_count'].sum())
print("interrupted reward: ", df_inter[:df_inter.shape[0]-df_inter.shape[0]%5]['reward'].sum())
print("normal reward: ", df_normal['reward'].sum())