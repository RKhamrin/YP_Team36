import pandas as pd
import numpy as np

def getStats(team, date, df):
    """
    
    """
    if 'result' in df.columns:
      srez = df[(df['team']== team)&(df['date'] < date)]
      if 0 < len(srez) < 10:
        stats = srez.drop(['team','date', 'opponent', 'venue', 'result'], axis = 1).sum()/len(srez)
      elif len(srez) >= 10:
        stats = srez.drop(['team','date', 'opponent', 'venue', 'result'], axis = 1)[-10:].sum()/10
      else:
        srez = df[(df['team']== team)&(df['date'] <= date)]
        stats = srez.drop(['team','date', 'opponent', 'venue', 'result'], axis = 1).sum()
    else:
      srez = df[(df['team']== team)&(df['date'] < date)]
      if 0 < len(srez) < 10:
        stats = srez.drop(['team','date', 'opponent', 'venue'], axis = 1).sum()/len(srez)
      elif len(srez) >= 10:
        stats = srez.drop(['team','date', 'opponent', 'venue'], axis = 1)[-10:].sum()/10
      else:
        srez = df[(df['team']== team)&(df['date'] <= date)]
        stats = srez.drop(['team','date', 'opponent', 'venue'], axis = 1).sum()  

    return stats.values.tolist()

def GetTrain(data):
  """
  
  """
  features = []
  for i in range(len(data)):
    team1 = getStats(data['team'][i], data['date'][i], data)
    team2 = getStats(data['opponent'][i], data['date'][i], data)
    diff = [a - b for a, b in zip(team1, team2)]
    features.append(diff)
  return features

def GetPrediction(data, enc, scaler, model):
  """
  
  """
  feat_data = GetTrain(data)
  x = pd.DataFrame(feat_data, columns = data.drop(['team','date', 'opponent', 'venue'], axis = 1).columns)
  x = pd.concat([x, data['venue']], axis = 1)

  for_enc = x[['venue']]
  encd = enc.transform(for_enc)
  one_hot_df = pd.DataFrame(encd, columns=enc.get_feature_names_out(['venue']))

  x = pd.concat([x.drop(['venue'], axis = 1),one_hot_df], axis = 1)
  x = scaler.transform(x)
  preds = model.predict(x)  
  return preds
