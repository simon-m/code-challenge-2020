import sys
import logging

from pathlib import Path
import click
import numpy as np
from dask import dataframe
from pickle import dump
from joblib import load
from scipy.sparse import save_npz, issparse

# Necessary for unpickling
sys.path.append("/usr/share/data/utils")
from utils import numeric_features, categorical_features,\
                  text_features, AverageColTransformer,\
                  SmallCategoriesAggregator


logging.basicConfig(level=logging.INFO)


def get_feature_info(feature_generator) -> dict:
    """Creates a dictionary containing feature names to be used for model
       interpretation.
       Topic words are also included for text variables procesed with a
       decomposition method (NMF, LDA).
       Note: quite horribly ad-hoc. Too tightly coupled with
       get_features_generator() in train_model/train_model.py.


    Parameters
    ----------
    feature_generator: str
        name of the joblib file containing a serialized processor
        (e.g. a pipeline or a transformer).

    Returns
    -------
    dict:
    """

    feature_info = {}

    col_names = numeric_features

    col_names.extend(list(map(lambda x: x + "_avg_points",
                          categorical_features)))

    ohe = feature_generator.named_transformers_['cat_not_winery']\
                           .named_steps['encoding']

    col_names.extend(ohe.get_feature_names(['country', 'province', 'region_1',
                                            'region_2', 'taster_name', 'variety']))

    ohe = feature_generator.named_transformers_['cat_winery']\
                           .named_steps['encoding']

    col_names.extend(ohe.get_feature_names(['winery']))

    for feat in text_features:
        feature_info[f"{feat}_vectorizer_tokens"] = feature_generator\
                                               .named_transformers_[feat]\
                                               .named_steps['vectorizer']\
                                               .get_feature_names()

        decomposition_step = feature_generator.named_transformers_[feat]\
                                              .named_steps['decomposition']

        feature_info[f"{feat}_decomposition_components"] = decomposition_step.components_

        col_names.extend([f"{feat}_topic_{i}"
                          for i in range(decomposition_step.n_components_)])

    feature_info["feature_names"] = col_names
    return feature_info


@click.command()
@click.option('--dataset')
@click.option('--feature-generator')
@click.option('--out-dir')
@click.option('--feature-info')
def generate_features(dataset: str,
                      feature_generator: str,
                      out_dir: str,
                      feature_info: str) -> None:
    """Applies a fitted serialized transformer or a pipeline thereof to a
       dataset.
       Splits the resulting dataset in feature (X) versus outcome (y) and
       writes them to the output directory.
       Extra feature information is also written out (see get_feature_info()).

    Parameters
    ----------
    dataset: str
        path to the parquet file containing data to be processed.
    feature_generator: str
        path to the the joblib file containing a serialized processor (e.g. a
        pipeline or a transformer).
    out_dir: str
        path to the directory where the results should be saved to.
    feature info: str
        "True" if additional feature info should be written, "False" otherwise.

    Returns
    -------
    None
    """

    log = logging.getLogger('generate-features')
    assert feature_info == "False" or feature_info == "True"

    ddf = dataframe.read_parquet(dataset)
    Xd = ddf.loc[:, ddf.columns != 'points']
    yd = ddf['points']

    feature_gen = load(feature_generator)
    log.info(f"Loading done")

    X = feature_gen.transform(Xd)
    y = np.asarray(yd.to_dask_array())
    log.info(f"Transform done")

    out_dir_path = Path(out_dir)
    Path.mkdir(out_dir_path, exist_ok=True, parents=True)

    # Sometimes X is sparse, sometimes not;
    # I could not find the reason for that.
    if issparse(X):
        save_npz(str(out_dir_path / "X"), X)
    else:
        np.save(str(out_dir_path / "X"), X)

    np.save(str(out_dir_path / "y"), y)

    if feature_info == "True":
        with open(out_dir_path / "feature_info.pkl", 'wb') as ofh:
            dump(get_feature_info(feature_gen), ofh)

    log.info(f"Results written in {out_dir}")

    flag = out_dir_path / '.SUCCESS'
    flag.touch()


if __name__ == '__main__':
    generate_features()
