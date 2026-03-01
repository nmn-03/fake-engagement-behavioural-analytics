import pandas as pd

df = pd.read_csv('data/synthetic_data.csv')

df['burst_ratio'] = df['engagement_burst'] / df['post_gap']

df['risk_index'] = (
    0.4 * df['timing_regularity'] +
    0.3 * df['comment_similarity'] +
    0.3 * df['burst_ratio']
)

df.to_csv('data/engineered_data.csv', index=False)

print("Features Engineered!")