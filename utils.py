import pandas as pd
import numpy as np

def import_and_transform(data):
    """
    Import and transform music streaming data for churn prediction.
    
    Args:
        data: Pandas DataFrame containing raw data, or string containing the path to the data file.
    
    Returns:
        DataFrame with user-level aggregated features
    """
    if isinstance(data, str):
        print("Importing parquet file")
        df = pd.read_parquet(data)
    else:
        print("Using Dataframe")
        df = data
    
    df = df[df['userId'] != '']
    df['userId'] = df['userId'].astype(int)
    
    df["gender"] = df["gender"].map({'F': 0, 'M': 1})
    df["level"] = df["level"].map({'free': 0, 'paid': 1})
    
    churned_users = df[df['page'] == 'Cancellation Confirmation']['userId'].unique()
    df['churned'] = df['userId'].isin(churned_users).astype(int)
    
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df['registration'] = pd.to_datetime(df['registration'])
    
    df['session_length'] = df.groupby(['userId', 'sessionId'])['ts'].transform(
        lambda x: (x.max() - x.min()).total_seconds()  # Convert to seconds immediately
    )
    df['song_played'] = (df['page'] == 'NextSong').astype(int)
    return df

def aggregate(data: pd.DataFrame):
    user_df = df.groupby('userId').agg({
        'gender': 'first',
        'registration': 'first',
        'level': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
        'sessionId': 'nunique',
        'itemInSession': 'max',
        'ts': ['min', 'max'],
        'session_length': 'mean',
        'song_played': 'sum',
        'artist': pd.Series.nunique,
        'length': 'sum',
        'churned': 'max'    
    }).reset_index()
    
    user_df.columns = ['userId', 'gender', 'registration', 'level',
                       'num_sessions', 'max_item_in_session', 'ts_min', 'ts_max', 
                       'avg_session_length_seconds',  # Renamed for clarity
                       'num_songs_played', 'unique_artists', 'total_length', 'churned']
    
    user_df['days_active'] = (user_df['ts_max'] - user_df['ts_min']).dt.days
    user_df['membership_length'] = (user_df['ts_max'] - user_df['registration']).dt.days
    
    user_df = user_df.fillna(0)
    
    print(f"Processed {len(user_df)} users")
    print(f"Churn rate: {user_df['churned'].mean():.2%}")

    final_column_order = [
            'userId', 'gender', 'registration', 'level',
            'num_sessions', 'max_item_in_session', 'ts_min', 'ts_max',
            'avg_session_length_seconds', 'num_songs_played',
            'unique_artists', 'total_length', 'days_active',
            'membership_length', 'churned'
        ]
    
        
    user_df = user_df[final_column_order]

    return user_df

def evaluate_model(model, test_set, p=None, file_out='submission.csv'):
    '''
    Evalute the given model and test set and create a submission file for Kaggle.
    Assumes that the test set has the same columns as the train set used for fitting.
    Assumes that the user id is used as index of the given set.

    Args:
        model: classifier model (already fitted) that we would like to test
        test_set: X_test set from the test parquet file.
        file_out: Name of the submission file produced.\
    
    Returns: 
        None
    '''
    user_ids = test_set.index
    y_pred = model.predict(test_set)
    print(f"Base predicted churn: {y_pred.mean():.2%}")
    if p is None:
        submission = pd.DataFrame({
            'id': user_ids,
            'target': y_pred
        })
    elif isinstance(p, float) and (0 <= p) and (p <= 1):
        y_proba = model.predict_proba(test_set)[:, 1]
        y_pred_adjusted = (y_proba > p).astype(int) 
        print(f"Predicted churn at {p} threshold: {y_pred_adjusted.mean():.2%}")
        
        submission = pd.DataFrame({
            'id': test_set.index,
            'target': y_pred_adjusted
        })
    
    submission.to_csv(f"{file_out}", index=False)

    print(f"Submission saved to {file_out}")
    return

    