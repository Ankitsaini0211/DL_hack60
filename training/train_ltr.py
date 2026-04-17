import lightgbm as lgb
import pandas as pd
import numpy as np

def train_ltr_model(training_csv: str = "training_data.csv", output_model: str = "lambdarank_model.txt"):
    df = pd.read_csv(training_csv).sort_values('query_id')
    group = df.groupby('query_id').size().to_numpy()

    X = df[['bm25', 'dense', 'years', 'skill_overlap', 'quality', 'intent_conf']]
    y = df['relevance']

    train_data = lgb.Dataset(X, label=y, group=group)
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [5, 10],
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_data_in_leaf': 20,
        'verbose': -1,
    }
    print("Training LightGBM LambdaRank model...")
    gbm = lgb.train(params, train_data, num_boost_round=100)
    gbm.save_model(output_model)
    print(f"Model saved to {output_model}")

if __name__ == "__main__":
    train_ltr_model()