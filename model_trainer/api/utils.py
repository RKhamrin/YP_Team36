import pandas as pd


def getStats(team, date, df):
    """Функция подсчета статистик для каждой команды
    params:
      team: str
      date: str
      df: pd.DataFrame

    returns:
      stats: List
    """
    if 'result' in df.columns:
        df_filter = df[(df['team'] == team) & (df['date'] < date)]
        if 0 < len(df_filter) < 10:
            df_filter = df_filter.drop(['team', 'date', 'opponent', 'venue', 'result'], axis=1).sum()/len(df_filter)
        elif len(df_filter) >= 10:
            stats = df_filter.drop(['team', 'date', 'opponent', 'venue', 'result'], axis=1)[-10:].sum()/10
        else:
            df_filter = df[(df['team'] == team) & (df['date'] <= date)]
            stats = df_filter.drop(['team', 'date', 'opponent', 'venue', 'result'], axis=1).sum()

    else:
        df_filter = df[(df['team'] == team) & (df['date'] < date)]
        if 0 < len(df_filter) < 10:
            stats = df_filter.drop(['team', 'date', 'opponent', 'venue'], axis=1).sum()/len(df_filter)

        elif len(df_filter) >= 10:
            stats = df_filter.drop(['team', 'date', 'opponent', 'venue'], axis=1)[-10:].sum()/10

        else:
            df_filter = df[(df['team'] == team) & (df['date'] <= date)]
            stats = df_filter.drop(['team', 'date', 'opponent', 'venue'], axis=1).sum()

    return stats.values.tolist()


def GetTrain(data):
    """Функция получения подготовленных данных
    params:
      data: pd.DataFrame

    returns:
      features: List
    """
    features = []
    for i in range(len(data)):
        team1 = getStats(data['team'][i], data['date'][i], data)
        team2 = getStats(data['opponent'][i], data['date'][i], data)
        diff = [a - b for a, b in zip(team1, team2)]
        features.append(diff)
    return features


def GetPrediction(data, enc, scaler, model):
    """Функция получения предсказаний
    params:
      data: pd.DataFrame
      enc: pickle
      scaler: pickle
      model: pickle

    returns:
      preds: np.array
    """
    feat_data = GetTrain(data)
    x = pd.DataFrame(feat_data, columns=data.drop(['team', 'date', 'opponent', 'venue'], axis=1).columns)
    x = pd.concat([x, data['venue']], axis=1)

    for_enc = x[['venue']]
    encd = enc.transform(for_enc)
    one_hot_df = pd.DataFrame(encd, columns=enc.get_feature_names_out(['venue']))

    x = pd.concat([x.drop(['venue'], axis=1), one_hot_df], axis=1)
    x = scaler.transform(x)
    preds = model.predict(x)
    return preds
