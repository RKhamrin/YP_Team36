import os
import pandas as pd

CSV_PATH = os.path.join(os.path.dirname(__file__), '../teams_matches_stats-2.csv')

_df = None

def get_logo_path(team_name: str) -> str:
    global _df
    if _df is None:
        _df = pd.read_csv(CSV_PATH)
    row = _df[_df['team'].str.lower() == team_name.lower()]
    if not row.empty and 'kaggle_logo_id' in row.columns:
        kaggle_id = row.iloc[0]['kaggle_logo_id']
        if pd.notna(kaggle_id):
            logo_filename = f'{kaggle_id}.png'
            logo_path = os.path.join(os.path.dirname(__file__), 'logos', logo_filename)
            if os.path.exists(logo_path):
                return logo_path
    logo_filename = f'{team_name.lower().replace(" ", "_")}.png'
    logo_path = os.path.join(os.path.dirname(__file__), 'logos', logo_filename)
    if os.path.exists(logo_path):
        return logo_path
    return None 