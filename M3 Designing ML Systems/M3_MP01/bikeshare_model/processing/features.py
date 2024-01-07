import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class WeekdayImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weekday' column by extracting dayname from 'dteday' column """

    def __init__(self, variables):
        if not isinstance(variables, str):
            raise ValueError('variables should be a str')
        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        wkday_null_idx = df[df[self.variables].isnull() == True].index
        df.loc[wkday_null_idx, self.variables] = df.loc[wkday_null_idx, 'dteday'].dt.day_name().apply(lambda x: x[:3])
        # print("Weekday imputation is done...")
        return df

class WeathersitImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weathersit' column by replacing them with the most frequent category value """

    def __init__(self, variables):

        if not isinstance(variables, str):
            raise ValueError('variables should be a str')
        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        self.fill_value=X[self.variables].mode()[0]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.variables]=X[self.variables].fillna(self.fill_value)
        # print('weathersit imputation is done.... fill value:', self.fill_value)
        return X
    

class Mapper(BaseEstimator, TransformerMixin):
    """
    Ordinal categorical variable mapper:
    Treat column as Ordinal categorical variable, and assign values accordingly
    """

    def __init__(self, variables: str, mappings: dict):

        if not isinstance(variables, str):
            raise ValueError("variables should be a str")

        self.variables = variables
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        #for feature in self.variables:
        X[self.variables] = X[self.variables].map(self.mappings).astype(int)
        # print(f'{self.variables} Mapper completed....\n')
        
        return X
    
    

class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values:
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """

    def __init__(self, variables):

        if not isinstance(variables, str):
            raise ValueError('variables should be a str')

        self.variables = variables

    def fit(self, X, y=None):
        df = X.copy()
        q1 = df.describe()[self.variables].loc['25%']
        q3 = df.describe()[self.variables].loc['75%']
        iqr = q3 - q1
        self.lower_bound = q1 - (1.5 * iqr)
        self.upper_bound = q3 + (1.5 * iqr)

        return self

    def transform(self, X):
        X_out = X.copy()
        for i in X_out.index:
            if X_out.loc[i,self.variables] > self.upper_bound:
                X_out.loc[i,self.variables]= self.upper_bound
            if X_out.loc[i,self.variables] < self.lower_bound:
                X_out.loc[i,self.variables]= self.lower_bound
        # print(f'Outlier for {self.variables} complete.....\n')
        return X_out



class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encode weekday column """

    def __init__(self):
        self.encoder_ = OneHotEncoder(sparse_output=False)

    def fit(self, X, y=None):
        self.encoder_.fit(X[['weekday']])
        return self

    def transform(self, X: pd.DataFrame):
        encoded_weekday = self.encoder_.transform(X[['weekday']])
        # Get encoded feature names
        enc_wkday_features = self.encoder_.get_feature_names_out(['weekday'])
        X[enc_wkday_features] = encoded_weekday
        
        X.drop(labels=['dteday', 'weekday'], axis=1, inplace=True)
        # print(X.head())
        return X