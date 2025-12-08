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

def aggregate_features_improved(data, observation_end):
    observation_end = pd.Timestamp(observation_end)
    
    # Time windows
    last_7_days = observation_end - pd.Timedelta(days=7)
    last_30_days = observation_end - pd.Timedelta(days=30)
    
    # Base aggregation
    user_df = data.groupby("userId").agg({
        "gender": "first",
        "registration": "first",
        "level": ["first", "last"],
        "sessionId": "nunique",
        "ts": ["min", "max"],
        "session_length": "mean",
        "song_played": "sum",
        "artist": "nunique",
        "song": "nunique",
        "length": ["sum", "mean"],
    }).reset_index()
    
    # Flatten columns
    user_df.columns = [
        "userId", "gender", "registration", 
        "level_first", "level_current",
        "num_sessions", "ts_min", "ts_max",
        "avg_session_length", "num_songs_played",
        "unique_artists", "unique_songs",
        "total_length", "avg_song_length"
    ]
    
    # Time-based features
    user_df["days_active"] = (observation_end - user_df["ts_min"]).dt.days
    user_df["membership_length"] = (observation_end - user_df["registration"]).dt.days
    user_df["days_since_last_activity"] = (observation_end - user_df["ts_max"]).dt.days
    
    # Recent activity
    recent = data[data["ts"] >= last_7_days]
    recent_counts = recent.groupby("userId").agg({
        "song_played": "sum",
        "sessionId": "nunique"
    }).rename(columns={
        "song_played": "songs_last_7_days",
        "sessionId": "sessions_last_7_days"
    })
    user_df = user_df.merge(recent_counts, on="userId", how="left")
    
    # Trend features
    user_df["songs_per_day_overall"] = user_df["num_songs_played"] / (user_df["days_active"] + 1)
    user_df["songs_per_day_recent"] = user_df["songs_last_7_days"] / 7
    user_df["activity_decline"] = (
        user_df["songs_per_day_recent"] / (user_df["songs_per_day_overall"] + 1)
    )
    
    # Engagement depth
    user_df["songs_per_session"] = user_df["num_songs_played"] / (user_df["num_sessions"] + 1)
    user_df["artist_diversity"] = user_df["unique_artists"] / (user_df["num_songs_played"] + 1)
    
    # Subscription behavior
    user_df["downgraded"] = (
        (user_df["level_first"] == 1) & (user_df["level_current"] == 0)
    ).astype(int)
    
    # ========================================
    # PAGE-SPECIFIC FEATURES (Fix here!)
    # ========================================
    
    # Create a helper function to count pages
    def count_page(df, page_name, column_name):
        counts = df[df["page"] == page_name].groupby("userId").size()
        return counts.rename(column_name)
    
    # Count all page types
    page_counts = pd.DataFrame({
        "friends_added": count_page(data, "Add Friend", "friends_added"),
        "thumbs_up_count": count_page(data, "Thumbs Up", "thumbs_up_count"),
        "thumbs_down_count": count_page(data, "Thumbs Down", "thumbs_down_count"),
        "playlists_created": count_page(data, "Add to Playlist", "playlists_created"),
        "help_visits": count_page(data, "Help", "help_visits"),
        "error_count": count_page(data, "Error", "error_count"),
        "settings_visits": count_page(data, "Settings", "settings_visits"),
        "cancel_page_visits": count_page(data, "Cancel", "cancel_page_visits"),
        "logout_count": count_page(data, "Logout", "logout_count"),
        "home_visits": count_page(data, "Home", "home_visits"),
        "about_visits": count_page(data, "About", "about_visits"),
        "ad_count": count_page(data, "Roll Advert", "ad_count"),
    }).reset_index()
    
    # Merge page counts to user_df
    user_df = user_df.merge(page_counts, on="userId", how="left")
    
    # Downgrade/upgrade attempts (multiple pages)
    downgrade_counts = (
        data[data["page"].isin(["Downgrade", "Submit Downgrade"])]
        .groupby("userId").size().rename("downgrade_attempts")
    )
    upgrade_counts = (
        data[data["page"].isin(["Upgrade", "Submit Upgrade"])]
        .groupby("userId").size().rename("upgrade_attempts")
    )
    
    user_df = user_df.merge(downgrade_counts, on="userId", how="left")
    user_df = user_df.merge(upgrade_counts, on="userId", how="left")
    
    # Total page views for actions_per_session
    total_views = data.groupby("userId").size().rename("total_page_views")
    user_df = user_df.merge(total_views, on="userId", how="left")
    
    # ========================================
    # DERIVED FEATURES (after merging)
    # ========================================
    
    # Social engagement
    user_df["has_social_activity"] = (user_df["friends_added"] > 0).astype(int)
    
    # Positive actions
    user_df["positive_actions"] = (
        user_df["thumbs_up_count"] + 
        user_df["playlists_created"] + 
        user_df["friends_added"]
    )
    
    # Satisfaction ratio
    user_df["satisfaction_ratio"] = (
        user_df["thumbs_up_count"] / 
        (user_df["thumbs_down_count"] + user_df["thumbs_up_count"] + 1)
    )
    
    # Engagement rate
    total_actions = (
        user_df["positive_actions"] + 
        user_df["thumbs_down_count"] + 
        user_df["help_visits"]
    )
    user_df["engagement_rate"] = (
        total_actions / (user_df["num_songs_played"] + 1)
    )
    
    # Problem signals
    user_df["problem_signals"] = (
        user_df["help_visits"] + 
        user_df["error_count"] + 
        user_df["settings_visits"]
    )
    
    # Ad metrics
    user_df["ads_per_song"] = (
        user_df["ad_count"] / (user_df["num_songs_played"] + 1)
    )
    
    # Actions per session
    user_df["actions_per_session"] = (
        user_df["total_page_views"] / (user_df["num_sessions"] + 1)
    )
    
    # Fill NaN with 0 (for users without certain activities)
    user_df = user_df.fillna(0)
    user_df.set_index("userId", inplace=True)
    
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

def evaluate_model_log_reg(model, test_set, threshold = 0.5, file_out='submission.csv'):
    '''
    Evalute the given model and test set and create a submission file for Kaggle.
    Assumes that the test set has the same columns as the train set used for fitting.
    Assumes that the user id is used as index of the given set.

    Args:
        model: classifier model (already fitted) that we would like to test
        test_set: X_test set from the test parquet file.
        file_out: Name of the submission file produced.
    
    Returns: 
        None
    '''
    
    y_proba = model.predict_proba(test_set)[:, 1]
    
    # Apply custom threshold
    y_pred = (y_proba >= threshold).astype(int)
    
    user_ids = test_set.index
    
    submission = pd.DataFrame({
        'id': user_ids,
        'target': y_pred
    })
    
    submission.to_csv(f"{file_out}", index=False)
    print(f"Submission saved to {file_out}")
    print(f"Predicted churn rate: {y_pred.mean():.2%}")
    return

    