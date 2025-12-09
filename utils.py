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
        df = pd.read_parquet(data)
    else:
        df = data
    
    df = df[df['userId'] != '']
    df['userId'] = df['userId'].astype(int)
    df["gender"] = df["gender"].map({'F': 0, 'M': 1})
    df["level"] = df["level"].map({'free': 0, 'paid': 1})
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df['registration'] = pd.to_datetime(df['registration'])
    
    df['session_length'] = df.groupby(['userId', 'sessionId'])['ts'].transform(
        lambda x: (x.max() - x.min()).total_seconds()
    )
    df['song_played'] = (df['page'] == 'NextSong').astype(int)
    
    return df

def get_churned_users(df, start_date, end_date):
    """
    Given a dataframe and observation window, get a list of all unique churned users.
    
    Args:
        df: DataFrame containing user actions information.
        start_date: Datetime format.
        end_date: Datetime format.
    
    Returns:
        Set containing userIds of all churned users within observation window.
    """
    # Convert the given dates to timestamps
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    # Identify users who cancelled their subscirptions in the given period
    cancellations = df[df["page"] == "Cancellation Confirmation"]
    churned = cancellations[
        (cancellations["ts"] > start) & (cancellations["ts"] <= end)
    ]["userId"].unique()

    # Return as a set
    return set(churned)

def aggregate(data: pd.DataFrame):
    """
    Simple aggregation by userId function. 
    Only does elementary aggregations, used for EDA purposes,
    Achieves rather poor accuracy when trying on the test set.
    
    Args: 
        data: DataFrame containing user information. 

    Returns:
        DataFrame with aggregated userId features.
    """
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
    """
    Improved aggregation function.

    Args:
        data: dataframe containing features.
        observation_end: DateTime format.

    Returns:
        DataFrame with aggregated userId features.
    """
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

def aggregate_features_improved2(data, observation_end):
    """
    Even more improved aggregation function.

    Args:
        data: dataframe containing features.
        observation_end: DateTime format.

    Returns:
        DataFrame with aggregated userId features.
    """
    observation_end = pd.Timestamp(observation_end)
    
    # Time windows
    last_7_days = observation_end - pd.Timedelta(days=7)
    last_14_days = observation_end - pd.Timedelta(days=14)
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
    
    # Recent activity (7 days)
    recent_7 = data[data["ts"] >= last_7_days]
    recent_counts_7 = recent_7.groupby("userId").agg({
        "song_played": "sum",
        "sessionId": "nunique",
        "artist": "nunique",
        "session_length": "mean"
    }).rename(columns={
        "song_played": "songs_last_7_days",
        "sessionId": "sessions_last_7_days",
        "artist": "artists_last_7_days",
        "session_length": "avg_session_length_recent"
    })
    user_df = user_df.merge(recent_counts_7, on="userId", how="left")
    
    # Recent activity (14-30 days for historical comparison)
    historical = data[(data["ts"] >= last_30_days) & (data["ts"] < last_14_days)]
    historical_counts = historical.groupby("userId").agg({
        "song_played": "sum",
        "sessionId": "nunique",
        "artist": "nunique"
    }).rename(columns={
        "song_played": "songs_historical",
        "sessionId": "sessions_historical",
        "artist": "artists_historical"
    })
    user_df = user_df.merge(historical_counts, on="userId", how="left")
    
    # Trend features
    user_df["songs_per_day_overall"] = user_df["num_songs_played"] / (user_df["days_active"] + 1)
    user_df["songs_per_day_recent"] = user_df["songs_last_7_days"] / 7
    user_df["activity_decline"] = (
        user_df["songs_per_day_recent"] / (user_df["songs_per_day_overall"] + 1)
    )
    
    # NEW: Recent vs Historical ratios
    user_df["recent_vs_historical_songs"] = (
        user_df["songs_last_7_days"] / (user_df["songs_historical"] + 1)
    )
    user_df["recent_vs_historical_sessions"] = (
        user_df["sessions_last_7_days"] / (user_df["sessions_historical"] + 1)
    )
    user_df["recent_vs_historical_artists"] = (
        user_df["artists_last_7_days"] / (user_df["artists_historical"] + 1)
    )
    user_df["session_length_decline"] = (
        user_df["avg_session_length_recent"] / (user_df["avg_session_length"] + 1)
    )
    
    # Engagement depth
    user_df["songs_per_session"] = user_df["num_songs_played"] / (user_df["num_sessions"] + 1)
    user_df["artist_diversity"] = user_df["unique_artists"] / (user_df["num_songs_played"] + 1)
    
    # NEW: Artist exploration rate
    user_df["artist_exploration_rate"] = (
        user_df["artists_last_7_days"] / (user_df["sessions_last_7_days"] + 1)
    )
    
    # Subscription behavior
    user_df["downgraded"] = (
        (user_df["level_first"] == 1) & (user_df["level_current"] == 0)
    ).astype(int)
    
    # NEW: Days on current level (detect recent level changes)
    level_changes = data[data["level"] != data.groupby("userId")["level"].transform("first")]
    last_level_change = level_changes.groupby("userId")["ts"].max()
    user_df = user_df.merge(
        last_level_change.rename("last_level_change"),
        on="userId",
        how="left"
    )
    user_df["days_on_current_level"] = (
        observation_end - user_df["last_level_change"]
    ).dt.days
    user_df["days_on_current_level"] = user_df["days_on_current_level"].fillna(
        user_df["membership_length"]
    )
    
    # ========================================
    # PAGE-SPECIFIC FEATURES
    # ========================================
    
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
    
    user_df = user_df.merge(page_counts, on="userId", how="left")
    
    # Downgrade/upgrade attempts
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
    
    # Total page views
    total_views = data.groupby("userId").size().rename("total_page_views")
    user_df = user_df.merge(total_views, on="userId", how="left")
    
    # NEW: Recent page activity (last 7 days)
    recent_cancel = recent_7[recent_7["page"] == "Cancel"].groupby("userId").size()
    recent_help = recent_7[recent_7["page"] == "Help"].groupby("userId").size()
    recent_thumbs_down = recent_7[recent_7["page"] == "Thumbs Down"].groupby("userId").size()
    
    user_df = user_df.merge(
        recent_cancel.rename("cancel_visits_recent"),
        on="userId",
        how="left"
    )
    user_df = user_df.merge(
        recent_help.rename("help_visits_recent"),
        on="userId",
        how="left"
    )
    user_df = user_df.merge(
        recent_thumbs_down.rename("thumbs_down_recent"),
        on="userId",
        how="left"
    )
    
    # NEW: Days since critical events
    cancel_dates = data[data["page"] == "Cancel"].groupby("userId")["ts"].max()
    help_dates = data[data["page"] == "Help"].groupby("userId")["ts"].max()
    downgrade_dates = data[data["page"].isin(["Downgrade", "Submit Downgrade"])].groupby("userId")["ts"].max()
    
    user_df = user_df.merge(cancel_dates.rename("last_cancel_date"), on="userId", how="left")
    user_df = user_df.merge(help_dates.rename("last_help_date"), on="userId", how="left")
    user_df = user_df.merge(downgrade_dates.rename("last_downgrade_date"), on="userId", how="left")
    
    user_df["days_since_cancel_visit"] = (observation_end - user_df["last_cancel_date"]).dt.days
    user_df["days_since_help_visit"] = (observation_end - user_df["last_help_date"]).dt.days
    user_df["days_since_downgrade_attempt"] = (observation_end - user_df["last_downgrade_date"]).dt.days
    
    # NEW: Session frequency trend (daily session counts slope)
    session_freq_features = []
    for user_id in data["userId"].unique():
        user_data = data[data["userId"] == user_id]
        daily_sessions = user_data.groupby(user_data["ts"].dt.date)["sessionId"].nunique()
        
        if len(daily_sessions) >= 3:
            x = np.arange(len(daily_sessions))
            slope = np.polyfit(x, daily_sessions.values, 1)[0]
            std_sessions = daily_sessions.std()
        else:
            slope = 0
            std_sessions = 0
        
        session_freq_features.append({
            "userId": user_id,
            "session_freq_trend": slope,
            "session_consistency": std_sessions
        })
    
    session_freq_df = pd.DataFrame(session_freq_features)
    user_df = user_df.merge(session_freq_df, on="userId", how="left")
    
    # NEW: Recency-weighted engagement (exponential decay)
    weighted_features = []
    for user_id in data["userId"].unique():
        user_data = data[data["userId"] == user_id]
        days_ago = (observation_end - user_data["ts"]).dt.days
        weights = np.exp(-days_ago / 7)  # 7-day half-life
        
        weighted_songs = (user_data["song_played"] * weights).sum() / (weights.sum() + 1)
        weighted_sessions = user_data.groupby("sessionId").first().apply(
            lambda x: np.exp(-(observation_end - x["ts"]).days / 7), axis=1
        ).sum()
        
        weighted_features.append({
            "userId": user_id,
            "weighted_songs": weighted_songs,
            "weighted_sessions": weighted_sessions
        })
    
    weighted_df = pd.DataFrame(weighted_features)
    user_df = user_df.merge(weighted_df, on="userId", how="left")
    
    # NEW: Last session quality
    last_session_data = data.sort_values("ts").groupby("userId").last()
    user_df = user_df.merge(
        last_session_data[["session_length"]].rename(columns={"session_length": "last_session_length"}),
        on="userId",
        how="left"
    )
    
    # ========================================
    # DERIVED FEATURES
    # ========================================
    
    user_df["has_social_activity"] = (user_df["friends_added"] > 0).astype(int)
    
    user_df["positive_actions"] = (
        user_df["thumbs_up_count"] + 
        user_df["playlists_created"] + 
        user_df["friends_added"]
    )
    
    user_df["satisfaction_ratio"] = (
        user_df["thumbs_up_count"] / 
        (user_df["thumbs_down_count"] + user_df["thumbs_up_count"] + 1)
    )
    
    total_actions = (
        user_df["positive_actions"] + 
        user_df["thumbs_down_count"] + 
        user_df["help_visits"]
    )
    user_df["engagement_rate"] = (
        total_actions / (user_df["num_songs_played"] + 1)
    )
    
    user_df["problem_signals"] = (
        user_df["help_visits"] + 
        user_df["error_count"] + 
        user_df["settings_visits"]
    )
    
    user_df["ads_per_song"] = (
        user_df["ad_count"] / (user_df["num_songs_played"] + 1)
    )
    
    user_df["actions_per_session"] = (
        user_df["total_page_views"] / (user_df["num_sessions"] + 1)
    )
    
    # NEW: Churn risk score (composite of warning signals)
    user_df["churn_risk_score"] = (
        (user_df["cancel_page_visits"] > 0).astype(int) * 3 +
        (user_df["help_visits"] > 2).astype(int) * 2 +
        user_df["downgraded"] * 3 +
        (user_df["days_since_last_activity"] > 3).astype(int) * 1 +
        (user_df["activity_decline"] < 0.5).astype(int) * 2 +
        (user_df["thumbs_down_recent"] > user_df["thumbs_down_count"] / 2).astype(int) * 1
    )
    
    # Fill NaN with 0
    user_df = user_df.fillna(0)
    
    # Drop temporary columns
    user_df = user_df.drop(columns=[
        "last_level_change", "last_cancel_date", 
        "last_help_date", "last_downgrade_date"
    ], errors="ignore")
    
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
