import pandas as pd
import numpy as np

def import_and_transform(data: str):
    """
    Import and transform music streaming data for churn prediction.
    
    Args:
        data: Path to parquet file
    
    Returns:
        DataFrame with user-level aggregated features
    """
    df = pd.read_parquet(data)
    
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