import pandas
from dask.dataframe import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import NMF
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


# Columns used in the whole project.
# taster_twitter_handle is not included because
# it is essentially a duplicate of twitter_taster
numeric_features = ['year', 'price']
categorical_features = ['country', 'province', 'region_1', 'region_2',
                        'taster_name', 'variety', 'winery']
text_features = ['designation', 'description', 'title']


class AverageColTransformer(BaseEstimator, TransformerMixin):
    """Transformer for categorical variables.
       Replaces a categorical value with the average of value_col
       over the same set of categorical values.

    Parameters
    ----------
    value_col: str
        the name of the column whose mean is calculated
    min_count: int
        minimum number of values to compute the mean
    unknown_value: str or float
        value to use when the categorical value was not found
        if 'mean', the average over the whole value_col is used
    """

    def __init__(self,
                 value_col: str,
                 min_count=2,
                 unknown_value='mean'):
        self.means = {}
        self.value_col = value_col
        self.min_count = min_count
        self.unknown_value = unknown_value

    def fit(self, X, y):

        Xc = X.copy()
        d = {self.value_col: y}
        Xc = Xc.assign(**d)

        # Overall mean
        if self.unknown_value == 'mean':
            self.unknown_value = Xc[self.value_col].mean()

        for col in Xc.columns:
            if col == self.value_col:
                continue
            # count categorical values in col
            counts = Xc[col].value_counts().reset_index()
            counts = counts.rename(columns={'index': col, col: 'count'})

            # average by categorical value
            means = Xc.groupby(col)[self.value_col].mean()\
                                                   .reset_index()
            mean_col_name = f"{col}_avg_{self.value_col}"
            means = means.rename(columns={self.value_col: mean_col_name})

            # filter on minimum count
            count_means = counts.merge(means, on=col)
            self.means[col] = count_means\
                                  .loc[count_means['count'] >= self.min_count,
                                       [col, mean_col_name]]
        return self

    def transform(self, X, y=None):
        ddf_out = X.copy()
        for col in X.columns:
            if col not in self.means:
                continue
            # TODO: one big merge instead of several?
            ddf_out = ddf_out.merge(self.means[col], on=col,
                                    how='left')
        col_diff = ddf_out.columns.difference(X.columns)
        if isinstance(ddf_out, pandas.DataFrame):
            return ddf_out.loc[:, col_diff].fillna(self.unknown_value)
        elif isinstance(ddf_out, DataFrame):
            return ddf_out.loc[:, col_diff].fillna(self.unknown_value).compute()


class SmallCategoriesAggregator(BaseEstimator, TransformerMixin):
    """Transformer for categorical variables.
       Values present less than min_count times in the dataset are pooled in
       a common value.
    """

    def __init__(self,
                 min_count: int,
                 pooled_value='_other_'):
        self.min_count = min_count
        self.pooled_value = pooled_value

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        ddf_out = X.copy()
        count_cols = []

        for col in X.columns:
            count_col_name = f"{col}_counts"
            count_cols.append(count_col_name)

            # Count values and assign their counts to them
            counts = X[col].value_counts().reset_index()
            counts = counts.rename(columns={'index': col, col: count_col_name})
            ddf_out = ddf_out.merge(counts, on=col, how='left')

            # Replace if count below threshold
            ddf_out[col] = ddf_out[col].where(
                ddf_out[count_col_name] >= self.min_count, self.pooled_value)

        col_diff = ddf_out.columns.difference(count_cols)
        if isinstance(ddf_out, pandas.DataFrame):
            return ddf_out.loc[:, col_diff]
        elif isinstance(ddf_out, DataFrame):
            return ddf_out.loc[:, col_diff].compute()
