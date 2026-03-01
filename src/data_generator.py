import pandas as pd
import numpy as np

np.random.seed(42)

def generate_data(n=1000):

    data = []

    for i in range(n):

        # Slightly imbalanced real-world scenario
        user_type = np.random.choice(['real','bot'], p=[0.75,0.25])

        # --- Behaviour with overlap ---
        if user_type == 'real':
            post_gap = np.random.normal(4,2)
            engagement_burst = np.random.normal(15,8)
            comment_similarity = np.random.uniform(0.2,0.7)
            timing_regularity = np.random.uniform(0.3,0.8)

        else:
            post_gap = np.random.normal(2.5,1.5)
            engagement_burst = np.random.normal(30,15)
            comment_similarity = np.random.uniform(0.4,0.9)
            timing_regularity = np.random.uniform(0.5,1)

        # --- Add behavioural noise (real-world randomness) ---
        post_gap += np.random.normal(0,0.5)
        engagement_burst += np.random.normal(0,3)
        comment_similarity += np.random.normal(0,0.05)
        timing_regularity += np.random.normal(0,0.05)

        # Keep values in logical range
        post_gap = max(post_gap, 0.1)
        engagement_burst = max(engagement_burst, 0.1)
        comment_similarity = min(max(comment_similarity, 0), 1)
        timing_regularity = min(max(timing_regularity, 0), 1)

        data.append([
            post_gap,
            engagement_burst,
            comment_similarity,
            timing_regularity,
            1 if user_type=='bot' else 0
        ])

    df = pd.DataFrame(data, columns=[
        'post_gap',
        'engagement_burst',
        'comment_similarity',
        'timing_regularity',
        'bot_label'
    ])

    df.to_csv('data/synthetic_data.csv', index=False)

generate_data()
print("Dataset Generated!")