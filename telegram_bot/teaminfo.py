import os
import pandas as pd

CSV_PATH = os.path.join(os.path.dirname(__file__), '../teams_matches_stats-2.csv')

_df = None

def get_team_info(team_name: str):
    global _df
    if _df is None:
        _df = pd.read_csv(CSV_PATH)
    team_rows = _df[_df['team'].str.lower() == team_name.lower()]
    if team_rows.empty:
        return None
    row = team_rows.iloc[0]
    info = {
        'Команда': row['team'],
        'Страна': row.get('country', 'неизвестно'),
        'Лига': row.get('league', 'неизвестно'),
        'Средняя стоимость игроков': row.get('mean_player_value', 'нет данных'),
        'Матчей сыграно': row.get('matches_played', 'нет данных'),
    }
    return info 

def get_top_teams(n=5):
    global _df
    if _df is None:
        _df = pd.read_csv(CSV_PATH)
    if 'mean_player_value' not in _df.columns:
        return []
    # Сортируем и берём топ-n
    top = _df.sort_values('mean_player_value', ascending=False).head(n)
    result = []
    for _, row in top.iterrows():
        result.append({
            'Команда': row['team'],
            'Средняя стоимость игроков': row['mean_player_value'],
            'Страна': row.get('country', 'неизвестно'),
            'Лига': row.get('league', 'неизвестно'),
        })
    return result 