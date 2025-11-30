import pandas as pd
import numpy as np

def import_and_transform(data: str):
    df = pd.read_parquet(data)
    order_columns = [
        'userId',
        'firstName', 
        'lastName',
        'gender',
        'registration',
        
        'sessionId',
        'itemInSession',
        
        'ts',
        'time',
        
        'level',
        'auth',
        'location',
        'userAgent',
        
        'page',
        'method',
        'status',
        
        'song',
        'artist',
        'length'
    ]
    df = df[order_columns]
    df["gender"] = df["gender"].map({'F':0, 'M':1})
    df["level"] = df["level"].map({'free' : 0, 'paid': 1})
    df["method"] = df["method"].map({'GET' : 0, 'PUT' : 1 })
    
    # ! THIS WORKS ONLY ON TRAIN, NOT ON TEST !
    df["auth"] = df["auth"].map({'Cancelled' : 0, 'Logged In' : 1 }) 
    try:
        df['userId'].astype(int)
        df['userId'] = df['userId'].astype(int)
        print("Only integers in userId - transformation successful")
    except Exception as e:
        print(f"Some non integer values in userId: {e}")

    churned_users = df[df['page'] == 'Cancellation Confirmation']['userId'].unique()
    df['churned'] = df['userId'].isin(churned_users).astype(int)
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df['registration'] = pd.to_datetime(df['registration'])
    
    df['session_length'] = df.groupby(['userId', 'sessionId'])['ts'].transform(lambda x: x.max() - x.min())
    df['song_played'] = df['page'] == 'NextSong'
    
    user_df = df.groupby('userId').agg({
        'gender': 'first',
        'registration': 'first',
        'location': 'last',
        'level': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
        'sessionId': 'nunique',  # number of sessions
        'itemInSession': 'max',
        'ts': ['min', 'max'],
        'session_length': 'mean',
        'song_played': 'sum',
        'artist': pd.Series.nunique,
        'length': 'sum',
        'churned': 'max'    
    }).reset_index()
    
    user_df.columns = ['userId', 'gender', 'registration', 'location', 'level',
                       'num_sessions', 'max_item_in_session', 'ts_min', 'ts_max', 'avg_session_length',
                       'num_songs_played', 'unique_artists', 'total_length', 'churned']
    
    user_df['days_active'] = (user_df['ts_max'] - user_df['ts_min']).dt.days
    user_df['membership_length'] = (user_df['ts_max'] - user_df['registration']).dt.days
    return user_df