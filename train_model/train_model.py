import sys
import logging

from dask import dataframe
import pandas
from pathlib import Path
import click
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import NMF
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import RFE, RFECV
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from dask_ml.model_selection import RandomizedSearchCV
from joblib import dump
from distributed import Client

# Necessary for unpickling
sys.path.append("/usr/share/data/utils")
from utils import numeric_features, categorical_features,\
                  AverageColTransformer,\
                  SmallCategoriesAggregator

logging.basicConfig(level=logging.INFO)


def get_features_generator() -> ColumnTransformer:
    """Returns a ColumnTransformer (feature generator) to be used in a
       prediction pipeline.
       Note: The default parameters have been selected based on a
       previously run  model selection and shoud yield decent results
       for a reasonnable training time.

    Parameters
    ----------

    Returns
    -------
    ColumnTransformer
    """

    numeric_transform = Pipeline(steps=[
        ('imputation', SimpleImputer(strategy='median'))])

    # categorical_features_with_points = categorical_features + ['points']
    categorical_features_with_points = categorical_features
    categorical_avg_outcome = Pipeline(steps=[
        ('category points_average', AverageColTransformer(value_col='points',
                                                          min_count=30))])

    categorical_transform_not_winery = Pipeline(steps=[
        ('category_cutoff', SmallCategoriesAggregator(min_count=30)),
        ('encoding', OneHotEncoder(handle_unknown='ignore'))])

    categorical_transform_winery = Pipeline(steps=[
        ('category_cutoff', SmallCategoriesAggregator(min_count=10)),
        ('encoding', OneHotEncoder(handle_unknown='ignore'))])

    text_transform_designation = Pipeline(steps=[
            ('vectorizer',
             TfidfVectorizer(input='content', sublinear_tf=False,
                             lowercase=False)),
            ('decomposition', NMF(n_components=50, max_iter=500))])

    text_transform_description = Pipeline(steps=[
            ('vectorizer',
             TfidfVectorizer(input='content', sublinear_tf=True,
                             lowercase=False)),
            ('decomposition', NMF(n_components=50, max_iter=500))])

    text_transform_title = Pipeline(steps=[
            ('vectorizer',
             TfidfVectorizer(input='content', sublinear_tf=True,
                             lowercase=False)),
            ('decomposition', NMF(n_components=20, max_iter=500))])

    feature_generator = ColumnTransformer(
        transformers=[
            ('num', numeric_transform, numeric_features),

            ('cat_avg', categorical_avg_outcome,
             categorical_features_with_points),

            ('cat_not_winery', categorical_transform_not_winery,
             ['country', 'province', 'region_1', 'region_2', 'taster_name',
              'variety']),

            ('cat_winery', categorical_transform_winery, ['winery']),

            ('designation', text_transform_designation, 'designation'),
            ('description', text_transform_description, 'description'),
            ('title', text_transform_title, 'title')],

        remainder="drop")

    return feature_generator


def model_selection(pipeline: Pipeline,
                    X,
                    y,
                    n_iter: int,
                    log) -> Pipeline:
    """Performs model selection using randomized search with cross-validation

    Parameters
    ----------
    pipline: Pipeline
        pipeline on which the search is to be performed
    X:
        dataframe containing the features on which model selection is performed
    y:
        dataframe containing the outcome on which model selection is performed
    n_iter: int
        number of search steps to be performed
    log:
        logger object

    Returns
    -------
    None
    """

    param_dists = {
        "feature_gen__cat_avg__category points_average__min_count": [15, 30, 50],

        "feature_gen__cat_not_winery__category_cutoff__min_count": [15, 30, 50],
        "feature_gen__cat_winery__category_cutoff__min_count": [5, 10],

        "feature_gen__designation__decomposition__n_components": [5, 20, 50],
        "feature_gen__designation__vectorizer__sublinear_tf": [True, False],
        "feature_gen__description__decomposition__n_components": [20, 50, 75],
        "feature_gen__description__vectorizer__sublinear_tf": [True, False],
        "feature_gen__title__decomposition__n_components": [5, 20, 50],
        "feature_gen__title__vectorizer__sublinear_tf": [True, False],

        "regressor__min_samples_leaf": [5, 10, 25, 50, 100],
        "regressor__max_features": ['sqrt', 'log2'],
        "regressor__n_estimators": [50, 100, 300]
    }

    log.info('Running model selection')
    log.info(f'n_iter = {n_iter}')
    searchcv = RandomizedSearchCV(estimator=pipeline,
                                  param_distributions=param_dists,
                                  n_iter=int(n_iter),
                                  cv=5,
                                  scoring='neg_mean_squared_error',
                                  return_train_score=False)
    searchcv.fit(X, y)
    log.info('Model selection done')

    return searchcv


@click.command()
@click.option('--dataset')
@click.option('--out-dir')
@click.option('--n-iter')
@click.option('--select-model')
def train_model(dataset: str,
                out_dir: str,
                n_iter: str,
                select_model: str) -> None:
    """Trains (fits) a model on the input dataset and serializes the transformer
       pipeline (feature generator) and the estimator (regressor) separately.
       Model selection can be performed instead of simple fitting.
       In this case, a table containing the results is output in file
       cv_results.csv

    Parameters
    ----------
    dataset: str
        path to the dataset used for training the model.
    out_dir: str
        path to the directory where the results should be saved to.
    n_iter: str
        number of model selection steps
    select_model: str
        "True" to run model selection, "False" if not. Errors otherwise.

    Returns
    -------
    None
    """

    # Does not work on my machine:
    # "Timed out trying to connect to 'tcp://dask-scheduler:8787' after 10 s"
    # c = Client('dask-scheduler:8786')
    c = Client()
    log = logging.getLogger('train-model')
    # Because of a bug with luigi BoolParameter, we cannot
    # have boolean flags
    assert select_model == "False" or select_model == "True"

    # Number of randomized search steps to perform
    n_iter = str(n_iter)

    # ddf = pandas.read_parquet(dataset)
    ddf = dataframe.read_parquet(dataset)
    X = ddf.loc[:, ddf.columns != 'points']
    y = ddf['points']

    feature_generator = get_features_generator()
    regressor = RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                      max_features='sqrt')
    pipeline = Pipeline(steps=[('feature_gen', feature_generator),
                               ('regressor', regressor)])

    out_dir_path = Path(out_dir)
    Path.mkdir(out_dir_path, exist_ok=True)

    if select_model == "False":
        log.info('Training model')
        pipeline.fit(X, y)
        log.info('Training done')

        feature_generator = pipeline.named_steps['feature_gen']
        regressor = pipeline.named_steps['regressor']

        log.info('Performing feature extraction')
        Xt = feature_generator.transform(X)
        regressor = RFE(regressor, n_features_to_select=29, 
                        step=10).fit(Xt, y)
        log.info(regressor.n_features_)
        log.info('Feature extraction done')

    else:
        selection = model_selection(pipeline, X, y, n_iter, log)
        feature_generator = selection.best_estimator_.named_steps['feature_gen']
        regressor = selection.best_estimator_.named_steps['regressor']

        pandas.DataFrame(selection.cv_results_).sort_values('rank_test_score')\
                                               .to_csv(str(out_dir_path / 'cv_results.csv'))

        log.info('Performing cross-validated feature extraction')
        Xt = feature_generator.transform(X)
        regressor = RFECV(regressor, scoring='neg_mean_squared_error',
                          step=10).fit(Xt, y)
        log.info(f'Best number of features: {regressor.n_features_}')
        log.info('Cross-validated feature extraction done')

    dump(feature_generator,
         str(out_dir_path / 'feature_generator.joblib'))
    dump(regressor, str(out_dir_path / 'regressor.joblib'))

    log.info(f"Results written in {out_dir}")

    flag = out_dir_path / '.SUCCESS'
    flag.touch()


if __name__ == '__main__':
    train_model()
