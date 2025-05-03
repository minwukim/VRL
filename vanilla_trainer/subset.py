import pandas as pd

df = pd.read_csv("processed_graph_qwq_samples.csv")

ids = []
for i in range(1000):
    for j in range(4):
        ids.append(i)

df['ids'] = ids
print(df.head())
df = df.loc[df['pattern'] != -2]

print(df.head())
print(len(df))

subset_cols=['ids']
# df = df.drop_duplicates(subset=['ids', 'pattern'])
dupes = df.groupby(subset_cols).cumcount()
df=df[dupes<2]

print(len(df))

df.to_csv("subset_graph_qwq.csv")
