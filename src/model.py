import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv('data/engineered_data.csv')

X = df[['post_gap','engagement_burst','comment_similarity','timing_regularity','burst_ratio','risk_index']]
y = df['bot_label']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train,y_train)

pred = model.predict(X_test)

print(classification_report(y_test,pred))

df['bot_probability'] = model.predict_proba(X)[:,1]
df['authenticity_score'] = 1 - df['bot_probability']

df.to_csv('outputs/final_scores.csv', index=False)

print("Model Complete!")