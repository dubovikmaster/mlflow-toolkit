import ast
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion

from mlflow_toolkit.formats import (
    get_format_handler,
    registered_suffixes,
)
from mlflow_toolkit.utils import (
    DataFrameLike,
    FileHandler,
    FilePath,
)

logger = logging.getLogger(__name__)

_DEFAULT_MAX_WORKERS = 8

_SKIPPED = object()


class MLflowWorker(MlflowClient):
    """
    Represents an MLflow worker extension for enhanced artifact logging capabilities.

    The MLflowWorker class extends the core functionality of the MlflowClient to
    support additional methods for logging data such as Pandas DataFrames, Python
    dictionaries, pickle files, and other file types. This class aims to provide
    a simplified interface for working with MLflow artifacts, enabling users to
    easily serialize and store diverse data types within MLflow runs.

    All ``log_*`` methods take ``(run_id, data, artifact_path)`` and all ``load_*``
    methods take ``(run_id, artifact_path)``.
    """

    @contextmanager
    def _log_artifact_context(self, run_id: str, artifact_path: FilePath):
        """
        Context manager that yields a temporary local path for a to-be-logged artifact
        and uploads it under `artifact_path` of the run once the block exits.

        Parameters:
            run_id: The ID of the run to associate the artifact with.
            artifact_path: The relative path (including the file name) under which
                the artifact should be logged.
        Return:
            Yields the temporary local file path as a pathlib.Path object.
        """
        artifact_path = FileHandler._file_path_prepare(artifact_path)
        artifact_dir = artifact_path.parent.as_posix()
        artifact_dir = None if artifact_dir == '.' else artifact_dir
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / artifact_path.name
            yield tmp_path
            self.log_artifact(run_id, str(tmp_path), artifact_dir)

    @contextmanager
    def _load_artifact_context(self, run_id: str, artifact_path: FilePath):
        """
        Context manager for downloading an artifact into a temporary local directory.

        This method helps in managing the temporary download of an artifact to a local
        directory during its usage and ensures proper cleanup of resources.

        Parameters:
            run_id: Run identifier for the context in which the artifact is located.
            artifact_path: Path to the artifact within the run context to be loaded.
        Return:
            Yields the path of the artifact's local copy as a pathlib.Path object.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(self.download_artifacts(run_id, str(artifact_path), temp_dir))
            yield local_path

    def log_dataframe(self, run_id: str, data: DataFrameLike, artifact_path: FilePath,
                      output_file_type: str | None = None, **kwargs) -> None:
        """
        Log a pandas or polars DataFrame/Series (or a polars LazyFrame) as an artifact
        to a specific artifact path.

        The method saves the provided data in the specified output file format (either
        'parquet' or 'csv') at the given `artifact_path`. If `output_file_type` is not
        provided, it is inferred from the artifact path suffix.

        Parameters:
            run_id: The ID of the run to associate this artifact with.
            data: The pandas/polars DataFrame or Series to be logged.
            artifact_path: The relative path under which the artifact should be logged.
            output_file_type: The output file format for the data. Must be either 'parquet'
                    or 'csv'. Defaults to None (inferred from the suffix).
            kwargs: Additional keyword arguments applicable to the saving function for
                    the specified file type (e.g., compression options).
        Return: None
        Raises:
            ValueError: If an unsupported file type is provided.
        """
        if output_file_type is None:
            if FileHandler.is_parquet_file(artifact_path):
                output_file_type = 'parquet'
            elif FileHandler.is_csv_file(artifact_path):
                output_file_type = 'csv'
        if output_file_type not in ('parquet', 'csv'):
            raise ValueError(f"Unsupported file type: {output_file_type!r} for {artifact_path!r}. "
                             f"Only 'parquet' and 'csv' are supported.")
        with self._log_artifact_context(run_id, artifact_path) as tmp_path:
            if output_file_type == 'parquet':
                FileHandler.save_dataframe_as_parquet_file(data, tmp_path, **kwargs)
            else:
                FileHandler.save_dataframe_as_csv_file(data, tmp_path, **kwargs)

    def log_as_pickle(self, run_id: str, data, artifact_path: FilePath, **kwargs) -> None:
        """
        Logs the provided data as a pickle file to the specified artifact path.

        The method serializes the given `data` object and saves it to the specified
        `artifact_path`. The serialization backend ('pickle', 'dill' or 'joblib') is
        inferred from the artifact path suffix.

        Parameters:
            run_id: The identifier of the run with which to associate the artifact.
            data: The data object to be serialized and saved as a pickle file.
            artifact_path: The path where the pickle file will be saved.
        Return: None
        """
        with self._log_artifact_context(run_id, artifact_path) as tmp_path:
            FileHandler.save_pickle_file(data, tmp_path, **kwargs)

    def log_dict(self, run_id: str, data: dict, artifact_path: FilePath, **kwargs) -> None:
        """
        Logs a dictionary as an artifact to the specified run.

        This method facilitates logging a Python dictionary object as an artifact.
        The dictionary is saved as a JSON or YAML file depending on the artifact
        path suffix, and logged to the specified artifact path within the run.

        Parameters:
            run_id: Identifier of the run where the artifact will be logged.
            data: The dictionary to be logged as an artifact.
            artifact_path: The relative path within the run where the dictionary artifact will be stored.
        Return: None
        """
        with self._log_artifact_context(run_id, artifact_path) as tmp_path:
            FileHandler.save_dict(data, tmp_path, **kwargs)

    def log_file(self, run_id: str, data, artifact_path: FilePath, **kwargs) -> None:
        """
        Logs and saves a file artifact to MLflow using the specified `run_id`.

        The file format is resolved from the artifact path suffix via the format registry
        (see `mlflow_toolkit.register_format`). Built-in formats include parquet, csv,
        feather, json, yaml, pickle/dill/joblib, text, numpy arrays and figures (images).
        Unsupported file types raise a `ValueError`.

        Parameters:
            run_id: Unique identifier for the MLflow run to which the artifact belongs.
            data: The data to be logged, which can vary in format depending on the file type.
            artifact_path: File path of the artifact to be logged. The file type is inferred
                from the suffix of this path.
            kwargs: Additional keyword arguments passed to the format's save function.
        Return: None
        Raises:
            ValueError: If no format is registered for the artifact path suffix.
        """
        handler = get_format_handler(artifact_path)
        if handler is None or handler.save is None:
            raise ValueError(f"Unsupported file type for artifact: {artifact_path!r}. "
                             f"Registered suffixes: {', '.join(registered_suffixes())}. "
                             f"Use mlflow_toolkit.register_format() to add your own format.")
        with self._log_artifact_context(run_id, artifact_path) as tmp_path:
            handler.save(data, tmp_path, **kwargs)

    def log_files(self, run_id: str, data: dict, max_workers: int | None = None) -> None:
        """
        Logs multiple files into a given run identified by run_id. Each file is
        added as an artifact with the filename specified in the data dictionary's
        keys and corresponding file content from its values.

        Artifacts are serialized and uploaded concurrently using a thread pool,
        which speeds things up considerably on remote artifact stores (S3, GCS, ...).

        Parameters:
            run_id: Unique identifier for the run to which the files will be logged.
            data: A dictionary containing artifact paths as keys and their
                corresponding file content as values to be logged into the run.
            max_workers: Number of threads used to upload artifacts concurrently.
                Defaults to min(8, number of files). Pass 1 to force sequential logging.
        Return: None
        Raises:
            The first exception raised by any upload; remaining uploads are cancelled.
        """
        if not data:
            return
        if max_workers is None:
            max_workers = min(_DEFAULT_MAX_WORKERS, len(data))
        if max_workers <= 1 or len(data) == 1:
            for file_name, file_content in data.items():
                self.log_file(run_id, data=file_content, artifact_path=file_name)
            return
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self.log_file, run_id, file_content, file_name): file_name
                for file_name, file_content in data.items()
            }
            try:
                for future in as_completed(futures):
                    future.result()
            except Exception:
                pool.shutdown(cancel_futures=True)
                raise

    def load_parquet_artifact(self, run_id: str, artifact_path: FilePath, **kwargs):
        """
        Loads a parquet artifact from the given path and returns it as a DataFrame.

        Parameters:
            run_id: Identifier for the run associated with the artifact.
            artifact_path: Path to the artifact to load.
            kwargs: Additional keyword arguments for the reader, e.g. ``backend='polars'``
                to load with polars instead of pandas.
        Return:
            The pandas (default) or polars dataframe loaded from the parquet artifact.
        """
        with self._load_artifact_context(run_id, artifact_path) as tmp_path:
            return FileHandler.load_parquet_file(tmp_path, **kwargs)

    def load_csv_artifact(self, run_id: str, artifact_path: FilePath, **kwargs):
        """
        Loads a CSV artifact from a specified artifact path associated with a given run ID.

        Parameters:
            run_id: The identifier for the run associated with the artifact.
            artifact_path: The path to the artifact to be loaded.
            kwargs: Additional keyword arguments for the reader, e.g. ``backend='polars'``
                to load with polars instead of pandas.
        Return:
            A pandas (default) or polars DataFrame containing the data from the CSV artifact.
        """
        with self._load_artifact_context(run_id, artifact_path) as tmp_path:
            return FileHandler.load_csv_file(tmp_path, **kwargs)

    def load_text_artifact(self, run_id: str, artifact_path: FilePath) -> str:
        """
        Load a text artifact file from a specific run and return its content as a string.

        Parameters:
            run_id: The unique identifier for the run.
            artifact_path: The path to the artifact file within the run's artifacts.
        Return:
            The content of the text artifact file as a string.
        """
        with self._load_artifact_context(run_id, artifact_path) as tmp_path:
            return FileHandler.load_text_file(tmp_path)

    def load_pickle_artifact(self, run_id: str, artifact_path: FilePath, **kwargs):
        """
        Load a pickle artifact from a given run ID and artifact path. The serialization
        backend ('pickle', 'dill' or 'joblib') is inferred from the artifact path suffix.

        Parameters:
            run_id: The unique identifier of the run associated with the artifact.
            artifact_path: The relative path to the artifact within the storage system.
        Return:
            The deserialized object loaded from the pickle file.
        """
        with self._load_artifact_context(run_id, artifact_path) as tmp_path:
            return FileHandler.load_pickle_file(tmp_path, **kwargs)

    def load_dataframe(self, run_id: str, artifact_path: FilePath,
                       file_type: str | None = None, backend: str | None = None, **kwargs):
        """
        Loads a DataFrame artifact from the specified run ID and path. The method supports
        loading artifacts in both 'parquet' and 'csv' file formats. If `file_type` is not
        provided, it is inferred from the artifact path suffix.

        Parameters:
            run_id: The identifier of the run from which the artifact is to be loaded.
            artifact_path: The path to the artifact to load.
            file_type: The type of file to load. Must be either 'parquet' or 'csv'.
                Defaults to None (inferred from the suffix).
            backend: The dataframe library to load with: 'pandas' (default) or 'polars'.
            kwargs: Additional arguments to pass to the specific file loader.
        Return:
            A loaded pandas or polars DataFrame from the specified artifact path and run ID.
        Raises:
            ValueError: If an unsupported file type is provided.
        """
        if file_type is None:
            if FileHandler.is_parquet_file(artifact_path):
                file_type = 'parquet'
            elif FileHandler.is_csv_file(artifact_path):
                file_type = 'csv'
        if file_type == 'parquet':
            return self.load_parquet_artifact(run_id, artifact_path, backend=backend, **kwargs)
        elif file_type == 'csv':
            return self.load_csv_artifact(run_id, artifact_path, backend=backend, **kwargs)
        else:
            raise ValueError(f"Unsupported file type for artifact: {artifact_path!r}. "
                             f"Only .parquet and .csv are supported.")

    def load_dict(self, run_id: str, artifact_path: FilePath) -> dict:
        """
        Loads a dictionary object from the specified artifact path within the given run context.

        Parameters:
            run_id: The unique identifier of the run from which the artifact is being loaded.
            artifact_path: The path to the artifact file that contains the dictionary to be loaded.
        Return:
            A dictionary object loaded from the specified artifact path.
        """
        with self._load_artifact_context(run_id, artifact_path) as tmp_path:
            return FileHandler.load_dict(tmp_path)

    def load_file(self, run_id: str, artifact_path: FilePath, **kwargs):
        """
        Load a file artifact from a specified run ID and path, using the format registry
        (see `mlflow_toolkit.register_format`). Built-in formats include parquet, csv,
        feather, json, yaml, pickle/dill/joblib, text and numpy arrays. Raises an error
        if the file format is not supported.

        Parameters:
            run_id: A unique identifier for the run from which the artifact is being loaded.
            artifact_path: Path to the artifact file to be loaded.
            kwargs: Additional keyword arguments passed to the format's load function
                (e.g. ``backend='polars'`` for dataframe formats).
        Return:
            The loaded data, whose format depends on the file type.
        Raises:
             ValueError: If no format with a loader is registered for the artifact path suffix.
        """
        handler = get_format_handler(artifact_path)
        if handler is None or handler.load is None:
            raise ValueError(f"Unsupported file type for artifact: {artifact_path!r}. "
                             f"Registered suffixes: {', '.join(registered_suffixes())}. "
                             f"Use mlflow_toolkit.register_format() to add your own format.")
        with self._load_artifact_context(run_id, artifact_path) as tmp_path:
            return handler.load(tmp_path, **kwargs)

    def _iter_artifact_files(self, run_id: str, artifact_path: str | None = None):
        """
        Recursively yields the paths of all file artifacts of a run under `artifact_path`.

        Parameters:
            run_id: The unique identifier of the run whose artifacts are being listed.
            artifact_path: The directory to start from. Defaults to the artifact root of the run.
        Return:
            Yields artifact file paths as strings.
        """
        for item in self.list_artifacts(run_id, artifact_path):
            if item.is_dir:
                yield from self._iter_artifact_files(run_id, item.path)
            else:
                yield item.path

    def load_files(self, run_id: str, artifact_path: FilePath | None = None,
                   max_workers: int | None = None) -> dict:
        """
        Fetch and load artifacts associated with a specific run into memory.

        This method recursively retrieves artifacts associated with a provided run ID and
        artifact path, including files in nested directories. Based on the file extensions,
        different loading methods are applied to the files, and the loaded artifacts are
        stored in a dictionary. Unsupported file types are skipped, with a warning logged.

        Artifacts are downloaded and deserialized concurrently using a thread pool,
        which speeds things up considerably on remote artifact stores (S3, GCS, ...).

        Parameters:
            run_id: The unique identifier of the run whose artifacts are being fetched.
            artifact_path: The path to the directory containing the artifacts. Defaults to
                the artifact root of the run.
            max_workers: Number of threads used to download artifacts concurrently.
                Defaults to min(8, number of files). Pass 1 to force sequential loading.

        Returns:
            dict: A dictionary where the keys are artifact paths (with extensions), and the
                values are the loaded artifact contents. Key order follows the artifact
                listing regardless of which download finishes first.
        """
        logger.info(f"Fetching artifacts for run_id: {run_id}")
        path = str(artifact_path) if artifact_path is not None else None
        file_paths = list(self._iter_artifact_files(run_id, path))
        if not file_paths:
            return {}
        if max_workers is None:
            max_workers = min(_DEFAULT_MAX_WORKERS, len(file_paths))

        def load_one(file_path: str):
            try:
                return self.load_file(run_id, file_path)
            except ValueError as e:
                logger.warning(f"Skipping artifact {file_path}: {e}")
                return _SKIPPED

        if max_workers <= 1 or len(file_paths) == 1:
            results = [load_one(file_path) for file_path in file_paths]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                results = list(pool.map(load_one, file_paths))
        artifacts = {file_path: value for file_path, value in zip(file_paths, results, strict=True)
                     if value is not _SKIPPED}
        logger.info(f"Artifacts downloaded from {artifact_path}")
        return artifacts

    def get_run_params(self, run_id: str) -> dict:
        """
        Retrieve and parse the parameters of an MLflow run.

        This method fetches the parameters of a specific MLflow run by its unique
        ID, attempts to evaluate string values into their respective Python types
        when possible, and returns them as a dictionary. Parameters that cannot
        be evaluated are returned as their original string values.

        Parameters:
            run_id (str): The unique identifier of the MLflow run whose parameters
                need to be retrieved.

        Returns:
            dict: A dictionary containing the run parameters where keys are
            parameter names, and values are their evaluated or string-represented
            values.
        """
        run_data_dict = self.get_run(run_id).data.params
        params = {}
        for key, value in run_data_dict.items():
            try:
                params[key] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                params[key] = value
        return params

    def get_latest_model_version(self, model_name: str) -> ModelVersion | None:
        """
        Fetches the latest version information of a model based on the given model name.

        This function retrieves all available versions of a specified model by querying
        the client. It then determines the latest version by comparing the version
        numbers of retrieved results. If no versions are found, None is returned.

        Parameters:
            model_name: Name of the model for which the latest version information needs to be fetched.
        Return:
            The latest ModelVersion of the model, or None if no versions are available.
        """
        model_versions = self.search_model_versions(f"name='{model_name}'")
        if model_versions:
            return max(model_versions, key=lambda x: int(x.version))
        return None
