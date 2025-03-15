import pandas as pd

from sklearn.pipeline import Pipeline, TransformerMixin
from sklearn.neighbors import LocalOutlierFactor

from tqdm import tqdm

def get_outliers_iqr(data, x, col):

    med, q_25, q_75 = data[col].describe()['50%'], data[col].describe()['25%'], data[col].describe()['75%']
    low_bound, high_bound = med - 3 * q_25, med + 3 + q_75
    out = 1 if low_bound <= x <= high_bound else 0

    return out

class OutliersIQRDeleter(TransformerMixin):
    def __init__(self, num_cols, **kwargs):

        self.kwargs = kwargs
        self.num_cols = num_cols

    def transform(self, X):
        X_iqr = X.copy()
        for col in tqdm(self.num_cols):
            X_iqr[f'{col}_iqr'] = X_iqr[col].apply(lambda x: get_outliers_iqr(X_iqr, x, col))
        iqr_cols = pd.Series(X_iqr.columns)[pd.Series(X_iqr.columns).str.endswith('iqr')].values
        X_iqr['iqr'] = X_iqr[iqr_cols].sum(axis=1) / X_iqr[iqr_cols].shape[1]

        self.del_index = X[X_iqr['iqr'] == 1].index

        return X[X_iqr['iqr'] == 1]

    def fit(self, *args, **kwargs):
        return self
    
    def transform_y(self, y):
        return y[self.del_index.tolist()]
    

class OutliersLOFDeleter(TransformerMixin):
    def __init__(self, num_cols, **kwargs):

        self.num_cols = num_cols
        self.kwargs = kwargs

    def transform(self, X):
        X_lof = X.copy()
        for col in tqdm(self.num_cols):
            lcf = LocalOutlierFactor()
            preds = lcf.fit_predict(X_lof[[col]])
            X_lof[f'{col}_lof'] = [0 if x <= -1 else 1 for x in preds]
        lof_cols = pd.Series(X_lof.columns)[pd.Series(X_lof.columns).str.endswith('lof')].values
        X_lof['lof'] = X_lof[lof_cols].sum(axis=1) / X_lof[lof_cols].shape[1]

        self.del_index = X[X_lof['lof'] == 1].index

        return X[X_lof['lof'] == 1]

    def fit(self, *args, **kwargs):
        return self
    
    def transform_y(self, y):
        return y[self.del_index.tolist()]
    

class OutliersIQRCorrector(TransformerMixin):
    def __init__(self, num_cols, **kwargs):

        self.num_cols = num_cols
        self.kwargs = kwargs

    def transform(self, X):
        X_iqr = X.copy()
        for col in self.num_cols:
            X_iqr[col] = self.get_outliers_corrector_iqr(X_iqr, col)

        return X_iqr

    def fit(self, *args, **kwargs):
        return self
    
    def get_outliers_corrector_iqr(self, X, col):

        med, q_25, q_75 = X[col].describe()['50%'], X[col].describe()['25%'], X[col].describe()['75%']
        low_bound, high_bound = med - 3 * q_25, med + 3 + q_75
        corrections = X[col].apply(
            lambda x: low_bound if x < low_bound else x if low_bound <= x <= high_bound else high_bound
            )

        return corrections
    

class OutliersLOFCorrector(TransformerMixin):
    def __init__(self, num_cols, **kwargs):

        self.num_cols = num_cols
        self.kwargs = kwargs

    def transform(self, X):
        X_lof = X.copy()
        for col in self.num_cols:
            X_lof[col] = self.get_outliers_corrector_lof(X_lof, col)

        return X_lof

    def fit(self, *args, **kwargs):
        return self

    def get_outliers_corrector_lof(self, X, col):

        lcf = LocalOutlierFactor()
        preds = lcf.fit_predict(X[[col]])
        cur_X = X[[col]].copy()
        cur_X['lof'] = preds

        med, q_25, q_75 = cur_X[col].describe()['50%'], cur_X[col].describe()['25%'], cur_X[col].describe()['75%']
        low_bound, high_bound = med - 3 * q_25, med + 3 + q_75

        corrections = cur_X.apply(
            lambda x: low_bound if (x['lof'] == -1 and x[col] < low_bound) else\
                  med if (x['lof'] == -1 and low_bound <= x[col] <= high_bound) else\
                  high_bound if (x['lof'] == -1 and x[col] > low_bound) else x[col],
                  axis=1
            )

        return corrections