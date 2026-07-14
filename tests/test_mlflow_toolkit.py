import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlflow_toolkit import MLflowWorker, register_format, registered_suffixes
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


class TestFormatRegistry:
    def test_custom_format_round_trip(self, worker, run_id):
        register_format(
            'upper-text', ['.uptxt'],
            save=lambda data, path, **kw: FileHandler.save_text_file(data.upper(), path),
            load=lambda path, **kw: FileHandler.load_text_file(path),
        )
        worker.log_file(run_id, 'hello', 'custom/greeting.uptxt')
        assert worker.load_file(run_id, 'custom/greeting.uptxt') == 'HELLO'

    def test_duplicate_suffix_raises(self):
        with pytest.raises(ValueError, match='already registered'):
            register_format('json2', ['.json'], save=lambda d, p, **kw: None)

    def test_overwrite_allowed(self):
        register_format('tmp-fmt', ['.tmpfmt'], save=lambda d, p, **kw: None)
        register_format('tmp-fmt-v2', ['.tmpfmt'], save=lambda d, p, **kw: None, overwrite=True)

    def test_requires_save_or_load(self):
        with pytest.raises(ValueError, match='save.*load'):
            register_format('empty', ['.empty'])

    def test_builtin_suffixes_registered(self):
        assert {'.parquet', '.csv', '.json', '.yml', '.pkl', '.txt', '.npy', '.feather'} <= set(registered_suffixes())

    def test_save_only_format_cannot_load(self, worker, run_id):
        with pytest.raises(ValueError, match='Unsupported file type'):
            worker.load_file(run_id, 'figure.png')

    def test_non_figure_to_image_raises(self, worker, run_id):
        with pytest.raises(TypeError, match='figure'):
            worker.log_file(run_id, 'not a figure', 'plot.png')


class TestPolars:
    @pytest.fixture()
    def pl_df(self):
        return pl.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})

    @pytest.mark.parametrize('artifact_path', ['pl/data.parquet', 'pl/data.csv', 'pl/data.feather'])
    def test_round_trip(self, worker, run_id, pl_df, artifact_path):
        worker.log_file(run_id, pl_df, artifact_path)
        loaded = worker.load_file(run_id, artifact_path, backend='polars')
        assert loaded.equals(pl_df)

    def test_lazyframe_collected_on_save(self, worker, run_id, pl_df):
        worker.log_dataframe(run_id, pl_df.lazy(), 'pl/lazy.parquet')
        loaded = worker.load_dataframe(run_id, 'pl/lazy.parquet', backend='polars')
        assert loaded.equals(pl_df)

    def test_polars_series(self, worker, run_id, pl_df):
        worker.log_dataframe(run_id, pl_df['a'], 'pl/series.parquet')
        loaded = worker.load_dataframe(run_id, 'pl/series.parquet', backend='polars')
        assert loaded['a'].to_list() == [1, 2, 3]

    def test_load_pandas_by_default(self, worker, run_id, pl_df):
        worker.log_dataframe(run_id, pl_df, 'pl/default.parquet')
        loaded = worker.load_dataframe(run_id, 'pl/default.parquet')
        assert isinstance(loaded, pd.DataFrame)

    def test_unknown_backend_raises(self, worker, run_id, pl_df):
        worker.log_dataframe(run_id, pl_df, 'pl/backend.parquet')
        with pytest.raises(ValueError, match='Unsupported backend'):
            worker.load_dataframe(run_id, 'pl/backend.parquet', backend='dask')

    def test_unsupported_data_type_raises(self, worker, run_id):
        with pytest.raises(TypeError, match='Unsupported dataframe type'):
            worker.log_dataframe(run_id, {'not': 'a dataframe'}, 'pl/bad.parquet')


class TestNewFormats:
    def test_feather_round_trip(self, worker, run_id, df):
        worker.log_file(run_id, df, 'nf/data.feather')
        loaded = worker.load_file(run_id, 'nf/data.feather')
        pd.testing.assert_frame_equal(loaded, df)

    def test_npy_round_trip(self, worker, run_id):
        arr = np.random.random((5, 3))
        worker.log_file(run_id, arr, 'nf/arr.npy')
        np.testing.assert_array_equal(worker.load_file(run_id, 'nf/arr.npy'), arr)

    def test_npz_round_trip(self, worker, run_id):
        data = {'x': np.arange(10), 'y': np.eye(3)}
        worker.log_file(run_id, data, 'nf/arrays.npz')
        loaded = worker.load_file(run_id, 'nf/arrays.npz')
        assert set(loaded) == {'x', 'y'}
        np.testing.assert_array_equal(loaded['x'], data['x'])
        np.testing.assert_array_equal(loaded['y'], data['y'])


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
