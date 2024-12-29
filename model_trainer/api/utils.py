import pandas as pd
import numpy as np

def getStats(team, date, df):
    srez = df[(df['team']== team)&(df['date'] < date)]
    if 0 < len(srez) < 10:
      stats = srez.drop(['team','date'], axis = 1).sum()/len(srez)
    elif len(srez) >= 10:
      stats = srez.drop(['team','date'], axis = 1)[-10:].sum()/10
    else:
      srez = df[(df['team']== team)&(df['date'] <= date)]
      stats = srez.drop(['team','date'], axis = 1).sum()

    return stats.values.tolist()

def GetTrain(data):
  features = []
  for i in range(len(data)):
    team1 = getStats(data['team'][i], data['date'][i], data)
    team2 = getStats(data['opponent'][i], data['date'][i], data)
    diff = [a - b for a, b in zip(team1, team2)]
    features.append(diff)
  return features

def GetPrediction(team1, team2, date, venue, data, enc, scaler, model):
  # preds = []
  # for i in range(len(data)):
  #   teamvec1 = getStats(team1, date)
  #   teamvec2 = getStats(team2, date)
  #   diff = [[a - b] for a, b in zip(teamvec1, teamvec2)]
  #   diff.append([venue])
  #   x = pd.DataFrame(np.transpose(diff), columns = data.columns)
  #   encd = enc.transform(x[['venue']])
  #   one_hot_df = pd.DataFrame(encd, columns=enc.get_feature_names_out(['venue']))
  #   x = pd.concat([x.drop(['venue'], axis = 1),one_hot_df], axis = 1)
  #   x_n = scaler.transform(x)
  #   y_pred = logreg.predict(x_n)
  #   preds.append(y_pred)
  #   if y_pred[0] == 1:
  #     return(team1+' выиграет')
  #   else:
  #     return(team1+' проиграет или ничья')
      

  feat_data = GetTrain(data)
  x = pd.DataFrame(feat_data, columns = data.drop(['team','date'], axis = 1).columns)
  x = pd.concat([x, data['venue']], axis = 1)

  for_enc = x[['venue']]
  encd = enc.transform(for_enc)
  one_hot_df = pd.DataFrame(encd, columns=enc.get_feature_names_out(['venue']))

  x = pd.concat([x.drop(['venue'], axis = 1),one_hot_df], axis = 1)
  x = scaler.transform(x)
  preds = model.predict(x)  
  return preds
