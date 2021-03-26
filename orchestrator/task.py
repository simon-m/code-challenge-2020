import logging

import luigi
import os
from pathlib import Path

from util import DockerTask

VERSION = os.getenv('PIPELINE_VERSION', '0.1')


class Debug(DockerTask):
    """Use this task with appropriate image to debug things."""

    @property
    def image(self):
        return f'code-challenge/download-data:{VERSION}'

    @property
    def command(self):
        return [
            'sleep', '3600'
        ]


class DownloadData(DockerTask):
    """Initial pipeline task downloads dataset."""

    fname = luigi.Parameter(default='wine_dataset')
    out_dir = luigi.Parameter(default='/usr/share/data/data/raw/')
    url = luigi.Parameter(
        default='https://github.com/datarevenue-berlin/code-challenge-2019/'
                'releases/download/0.1.0/dataset_sampled.csv'
    )

    @property
    def image(self):
        return f'code-challenge/download-data:{VERSION}'

    @property
    def command(self):
        return [
            'python', 'download_data.py',
            '--name', self.fname,
            '--url', self.url,
            '--out-dir', self.out_dir
        ]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / f'{self.fname}.csv')
        )


class PreprocessSplitDataset(DockerTask):

    out_dir = luigi.Parameter(default='/usr/share/data/data/intermediate')
    test_size = luigi.Parameter(default='0.2')

    @property
    def image(self):
        return f'code-challenge/preprocess-split-dataset:{VERSION}'

    def requires(self):
        return DownloadData()

    @property
    def command(self):
        in_csv = self.requires().output().path
        return [
            'python', 'preprocess_split_dataset.py',
            '--in-csv', in_csv,
            '--out-dir', self.out_dir,
            '--test-size', self.test_size
        ]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / '.SUCCESS')
        )


class TrainModel(DockerTask):

    out_dir = luigi.Parameter(default='/usr/share/data/models')
    n_iter = luigi.Parameter(default="100")
    # Crashes when using BoolParameter
    select_model = luigi.Parameter(default="False")

    @property
    def image(self):
        return f'code-challenge/train-model:{VERSION}'

    def requires(self):
        return PreprocessSplitDataset()

    @property
    def command(self):
        dataset = str(Path(self.requires().out_dir) / 'train_set.parquet')
        return [
            'python', 'train_model.py',
            '--dataset', dataset,
            '--out-dir', self.out_dir,
            '--n-iter', self.n_iter,
            '--select-model', self.select_model
        ]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / '.SUCCESS')
        )


class GenerateTrainFeatures(DockerTask):

    out_dir = luigi.Parameter(default='/usr/share/data/data/processed/train_set')
    feature_info = luigi.Parameter(default='False')

    @property
    def image(self):
        return f'code-challenge/generate-features:{VERSION}'

    def requires(self):
        return PreprocessSplitDataset(), TrainModel()

    @property
    def command(self):
        pps_dataset, train_model = self.requires()
        dataset = str(Path(pps_dataset.out_dir) / 'train_set.parquet')
        feature_generator = str(Path(train_model.out_dir) / 'feature_generator.joblib')
        return [
            'python', 'generate_features.py',
            '--dataset', dataset,
            '--feature-generator', feature_generator,
            '--out-dir', self.out_dir,
            '--feature-info', self.feature_info
        ]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / '.SUCCESS')
        )


class GenerateTestFeatures(DockerTask):

    out_dir = luigi.Parameter(default='/usr/share/data/data/processed/test_set')
    feature_info = luigi.Parameter(default='True')

    @property
    def image(self):
        return f'code-challenge/generate-features:{VERSION}'

    def requires(self):
        return PreprocessSplitDataset(), TrainModel()

    @property
    def command(self):
        pps_dataset, train_model = self.requires()
        dataset = str(Path(pps_dataset.out_dir) / 'test_set.parquet')
        feature_generator = str(Path(train_model.out_dir) / 'feature_generator.joblib')
        return [
            'python', 'generate_features.py',
            '--dataset', dataset,
            '--feature-generator', feature_generator,
            '--out-dir', self.out_dir,
            '--feature-info', self.feature_info
        ]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / '.SUCCESS')
        )


class ReportEvaluation(DockerTask):

    out_dir = luigi.Parameter(default='/usr/share/data/results/')
    out_fname = luigi.Parameter(default='report.html')

    @property
    def image(self):
        return f'code-challenge/report-evaluation:{VERSION}'

    def requires(self):
        return GenerateTrainFeatures(), GenerateTestFeatures(), TrainModel()

    @property
    def command(self):
        return [
            'pweave', 'report_evaluation.py',
            '--output', str(Path(self.out_dir) / f'{self.out_fname}')
        ]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / f'{self.out_fname}')
        )
