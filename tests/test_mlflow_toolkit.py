import warnings

import numpy as np
import pandas as pd
import pytest

from mlflow_toolkit import MLflowWorker
from mlflow_toolkit.utils import FileHandler


@pytest.fixture(scope='module')
def worker(tmp_path_factory):
    db_path = tmp_path_factory.mktemp('tracking') / 'mlflow.db'
    return MLflowWorker(tracking_uri=f"sqlite:///{db_path}")


@pytest.fixture(scope='module')
def experiment_id(worker, tmp_path_factory):
    artifact_root = tmp_path_factory.mktemp('artifacts')
    return worker.create_experiment('test-exp', artifact_location=artifact_root.as_uri())


@pytest.fixture()
def run_id(worker, experiment_id):
    return worker.create_run(experiment_id).info.run_id


@pytest.fixture()
def df():
    return pd.DataFrame(np.random.random((20, 4)), columns=['a', 'b', 'c', 'd'])


class TestDataframeRoundTrip:
    def test_parquet(self, worker, run_id, df):
        worker.log_dataframe(run_id, df, 'data/train.parquet')
        loaded = worker.load_dataframe(run_id, 'data/train.parquet')
        pd.testing.assert_frame_equal(loaded, df)

    def test_csv(self, worker, run_id, df):
        worker.log_dataframe(run_id, df, 'data/train.csv', index=False)
        loaded = worker.load_dataframe(run_id, 'data/train.csv')
        pd.testing.assert_frame_equal(loaded, df)

    def test_series_as_parquet(self, worker, run_id, df):
        worker.log_dataframe(run_id, df['a'], 'series.parquet')
        loaded = worker.load_dataframe(run_id, 'series.parquet')
        pd.testing.assert_series_equal(loaded['a'], df['a'])

    def test_explicit_file_type_overrides_suffix(self, worker, run_id, df):
        worker.log_dataframe(run_id, df, 'data/blob.dat', output_file_type='csv', index=False)
        loaded = worker.load_dataframe(run_id, 'data/blob.dat', file_type='csv')
        pd.testing.assert_frame_equal(loaded, df)

    def test_unsupported_suffix_raises(self, worker, run_id, df):
        with pytest.raises(ValueError, match='Unsupported file type'):
            worker.log_dataframe(run_id, df, 'data/train.foo')
        with pytest.raises(ValueError, match='Unsupported file type'):
            worker.load_dataframe(run_id, 'data/train.foo')


class TestDictRoundTrip:
    params = {'iterations': 100, 'depth': 5, 'cat_features': ['a', 'b']}

    @pytest.mark.parametrize('artifact_path', ['params.json', 'params.yml', 'nested/params.yaml'])
    def test_round_trip(self, worker, run_id, artifact_path):
        worker.log_dict(run_id, self.params, artifact_path)
        assert worker.load_dict(run_id, artifact_path) == self.params

    def test_unsupported_suffix_raises(self, worker, run_id):
        with pytest.raises(ValueError):
            worker.log_dict(run_id, self.params, 'params.toml')


class TestPickleRoundTrip:
    obj = {'model': [1, 2, 3], 'threshold': 0.5}

    @pytest.mark.parametrize('artifact_path', ['obj.pkl', 'obj.pickle', 'obj.dill', 'obj.joblib'])
    def test_round_trip(self, worker, run_id, artifact_path):
        worker.log_as_pickle(run_id, self.obj, artifact_path)
        assert worker.load_pickle_artifact(run_id, artifact_path) == self.obj

    def test_unknown_suffix_warns_and_falls_back_to_pickle(self, tmp_path):
        with pytest.warns(UserWarning, match='pickle'):
            FileHandler.save_pickle_file(self.obj, tmp_path / 'obj.bin')
        with pytest.warns(UserWarning, match='pickle'):
            assert FileHandler.load_pickle_file(tmp_path / 'obj.bin') == self.obj


class TestTextRoundTrip:
    def test_round_trip(self, worker, run_id):
        text = 'a\nb\nc — юникод'
        worker.log_text(run_id, text, 'features.txt')
        assert worker.load_text_artifact(run_id, 'features.txt') == text


class TestLogLoadFile:
    def test_format_inference(self, worker, run_id, df):
        worker.log_file(run_id, df, 'auto/data.parquet')
        worker.log_file(run_id, {'a': 1}, 'auto/params.json')
        worker.log_file(run_id, 'hello', 'auto/note.txt')
        pd.testing.assert_frame_equal(worker.load_file(run_id, 'auto/data.parquet'), df)
        assert worker.load_file(run_id, 'auto/params.json') == {'a': 1}
        assert worker.load_file(run_id, 'auto/note.txt') == 'hello'

    def test_unsupported_raises(self, worker, run_id):
        with pytest.raises(ValueError, match='Unsupported file type'):
            worker.log_file(run_id, b'raw', 'blob.bin')

    def test_log_files_and_load_files(self, worker, run_id, df):
        worker.log_files(run_id, {
            'batch/data.csv': df,
            'batch/params.yml': {'depth': 5},
            'batch/note.txt': 'hi',
        })
        artifacts = worker.load_files(run_id, 'batch')
        assert set(artifacts) == {'batch/data.csv', 'batch/params.yml', 'batch/note.txt'}
        assert artifacts['batch/params.yml'] == {'depth': 5}

    def test_load_files_is_recursive(self, worker, run_id, df):
        worker.log_files(run_id, {
            'top.json': {'a': 1},
            'data/train.parquet': df,
            'data/deep/nested.yml': {'b': 2},
        })
        artifacts = worker.load_files(run_id)
        assert set(artifacts) == {'top.json', 'data/train.parquet', 'data/deep/nested.yml'}
        assert artifacts['data/deep/nested.yml'] == {'b': 2}

    def test_load_files_skips_unsupported(self, worker, run_id, tmp_path):
        blob = tmp_path / 'model.bin'
        blob.write_bytes(b'\x00\x01')
        worker.log_artifact(run_id, str(blob), 'mixed')
        worker.log_file(run_id, {'a': 1}, 'mixed/ok.json')
        artifacts = worker.load_files(run_id, 'mixed')
        assert artifacts == {'mixed/ok.json': {'a': 1}}


class TestRunParams:
    def test_types_restored(self, worker, run_id):
        worker.log_param(run_id, 'iterations', 100)
        worker.log_param(run_id, 'lr', 0.1)
        worker.log_param(run_id, 'cat_features', ['a', 'b'])
        worker.log_param(run_id, 'name', 'catboost')
        params = worker.get_run_params(run_id)
        assert params['iterations'] == 100
        assert params['lr'] == 0.1
        assert params['cat_features'] == ['a', 'b']
        assert params['name'] == 'catboost'


class TestModelRegistry:
    def test_latest_version(self, worker):
        worker.create_registered_model('my-model')
        worker.create_model_version('my-model', source='dummy/1')
        worker.create_model_version('my-model', source='dummy/2')
        latest = worker.get_latest_model_version('my-model')
        assert latest is not None
        assert int(latest.version) == 2

    def test_no_versions_returns_none(self, worker):
        worker.create_registered_model('empty-model')
        assert worker.get_latest_model_version('empty-model') is None


class TestFileHandler:
    def test_bytes_path_supported(self):
        assert FileHandler.is_parquet_file(b'data/file.parquet')
        assert FileHandler.is_csv_file(b'data/file.csv')

    def test_optional_backends_imported_independently(self):
        # regression: dill and joblib used to be imported in one try-block,
        # so a missing dill also disabled joblib
        from mlflow_toolkit import utils
        assert utils.joblib is not None
        assert utils.dill is not None

    def test_json_saved_indented(self, tmp_path):
        FileHandler.save_dict({'a': 1}, tmp_path / 'x.json')
        assert '\n' in (tmp_path / 'x.json').read_text()

    def test_no_warning_for_known_suffixes(self, tmp_path):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            FileHandler.save_pickle_file({'a': 1}, tmp_path / 'x.pkl')
            FileHandler.load_pickle_file(tmp_path / 'x.pkl')
