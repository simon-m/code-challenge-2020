import string
import logging

import click
from pathlib import Path
import numpy as np
from unidecode import unidecode
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from dask import dataframe
from distributed import Client


nltk.download('stopwords')
nltk.download('punkt')

logging.basicConfig(level=logging.INFO)

ps = PorterStemmer()
stopw = set(stopwords.words('english') + list(string.punctuation))


def stem_description(s: str) -> str:
    """ Removes accents and stop words to a string.
        Applies stemming to the remaining words.

    Parameters
    ----------
    s: str
        the string to be processed.

    Returns
    -------
    str
    """

    words = word_tokenize(unidecode(s))
    sel_words = []
    for word in words:
        if word not in stopw:
            sel_words.append(ps.stem(word))
    return " ".join(sel_words)


def preprocess_dataset(ddf: dataframe) -> dataframe:
    """Preprocesses a dataFrame:
        - constant missing value replacement
        - lower case
        - strip accentuated characters
        - extract year from title and simplifies it to avoid redundancy
        - Stop words removal and stemming

    Parameters
    ----------
    ddf: str
        the dataframe to be processed.

    Returns
    -------
    dataframe
    """

    text_cols = ['country', 'designation', 'province', 'region_1',
                 'region_2', 'taster_name', 'variety', 'winery']

    ddf = ddf.map_partitions(lambda d: d.assign(country=d['country'].fillna("_missing_").str.lower().apply(unidecode)))
    ddf = ddf.map_partitions(lambda d: d.assign(designation=d['designation'].fillna("_missing_").str.lower().apply(unidecode)))
    ddf = ddf.map_partitions(lambda d: d.assign(province=d['province'].fillna("_missing_").str.lower().apply(unidecode)))
    ddf = ddf.map_partitions(lambda d: d.assign(region_1=d['region_1'].fillna("_missing_").str.lower().apply(unidecode)))
    ddf = ddf.map_partitions(lambda d: d.assign(region_2=d['region_2'].fillna("_missing_").str.lower().apply(unidecode)))
    ddf = ddf.map_partitions(lambda d: d.assign(taster_name=d['taster_name'].fillna("_missing_").str.lower().apply(unidecode)))
    ddf = ddf.map_partitions(lambda d: d.assign(variety=d['variety'].fillna("_missing_").str.lower().apply(unidecode)))
    ddf = ddf.map_partitions(lambda d: d.assign(winery=d['winery'].fillna("_missing_").str.lower().apply(unidecode)))

    # Get year from the title
    ddf = ddf.map_partitions(lambda d: d.assign(year=d['title'].str.extract('(\d{4,})', expand=False).astype(float)))

    # Remove year and geographical info from the tilte. They are in already other columns.
    ddf = ddf.map_partitions(lambda d: d.assign(
                                           title=d['title'].fillna("_missing_").str.lower()
                                              .apply(unidecode)
                                              .str.replace('(\d+ )', '')
                                              .str.replace('\((.+)\)\s*$', '')
                                              .str.replace('\s{2,}', ' ')
                                              .fillna("_missing_")))

    ddf = ddf.map_partitions(lambda d: d.assign(
                                          description=d['description']
                                              .fillna("_missing_")
                                              .str.lower()
                                              .apply(stem_description)
                                              .fillna("_missing_")))

    return ddf


@click.command()
@click.option('--in-csv')
@click.option('--out-dir')
@click.option('--test-size')
def preprocess_split_dataset(in_csv: str,
                             out_dir: str,
                             test_size: str) -> None:
    """Preprocesses a csv dataset and randomly splits it in a training and
       a test set.
       Saves the resulting datasets in separate files.

    Parameters
    ----------
    in_csv: str
        path to the csv file to be processed.
    out_dir: str
        path to the directory where the results should be saved to.
    test_size: str
        Test set size relative to the full dataset.

    Returns
    -------
    None
    """

    # Does not work on my machein:
    # "Timed out trying to connect to 'tcp://dask-scheduler:8787' after 10 s"
    # c = Client('dask-scheduler:8786')
    c = Client()
    log = logging.getLogger('preprocess-split-dataset')

    test_size_f = float(test_size)
    log.info(f"Test size is set to: {test_size_f}")

    # Preprocessing
    log.info('Starting preprocessing')
    # no blocksize should auto select the right block size
    ddf = dataframe.read_csv(in_csv)
    ddf = ddf.set_index('Unnamed: 0')
    ddf = preprocess_dataset(ddf)
    log.info('Preprocessing done')

    # Splitting
    log.info('Start random splitting')
    n_samples = len(ddf)
    log.info(f"Dataset shape: {ddf.shape}")

    idx = np.arange(n_samples)

    train_idx = idx[int(n_samples * test_size_f):]
    train = ddf.loc[train_idx]

    test_idx = idx[:int(n_samples * test_size_f)]
    test = ddf.loc[test_idx]
    log.info('Random splitting done')

    # Results saving
    out_dir_path = Path(out_dir)
    Path.mkdir(out_dir_path, exist_ok=True)

    out_train_features = out_dir_path / 'train_set.parquet'
    train.to_parquet(str(out_train_features))

    out_test_features = out_dir_path / 'test_set.parquet'
    test.to_parquet(str(out_test_features))

    log.info(f"Files written in {out_dir}")

    flag = out_dir_path / '.SUCCESS'
    flag.touch()


if __name__ == '__main__':
    preprocess_split_dataset()
