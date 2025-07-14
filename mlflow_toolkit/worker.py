import tempfile
from pathlib import Path
from typing import (
    Union,
    Optional,
    List,
    Dict,
)
import logging
from contextlib import contextmanager
import ast

import pandas as pd

import mlflow
from mlflow import MlflowClient
from mlflow.tracking.fluent import ActiveRun

from mlflow_toolkit.utils import (
    FileHandler,
    FilePath,
)

logger = logging.getLogger(__name__)


class MLflowWorker(MlflowClient):
    """
    Represents an MLflow worker extension for enhanced artifact logging capabilities.

    The MLflowWorker class extends the core functionality of the MlflowClient to
    support additional methods for logging data such as Pandas DataFrames, Python
    dictionaries, pickle files, and other file types. This class aims to provide
    a simplified interface for working with MLflow artifacts, enabling users to
    easily serialize and store diverse data types within MLflow runs.

    """
    def __init__(self, tracking_uri: Optional[str] = None, registry_uri: Optional[str] = None):
        """
        Parameters:
            tracking_uri: Address of local or remote tracking server. If not provided, defaults
                to the service set by ``mlflow.tracking.set_tracking_uri``. See
                `Where Runs Get Recorded <../tracking.html#where-runs-get-recorded>`_
                for more info.
            registry_uri: Address of local or remote model registry server. If not provided,
                defaults to the service set by ``mlflow.tracking.set_registry_uri``. If
                no such service was set, defaults to the tracking uri of the client.
        """
        super().__init__(tracking_uri=tracking_uri, registry_uri=registry_uri)

    def log_dataframe(self, run_id: str, data: Union[pd.Series, pd.DataFrame], artifact_path: FilePath,
                      output_file_type: Optional[str] = None, **kwargs) -> None:
        """
        Log a Pandas DataFrame or Series as an artifact to a specific artifact path.

        The method saves the provided Pandas DataFrame or Series in the specified output
        file format (either 'parquet' or 'csv') at the given `artifact_path`. This artifact
        is then logged into the system using the associated `run_id`. Additional keyword
        arguments can be passed to customize the behavior when saving the file.

        Parameters:
            run_id: The ID of the run to associate this artifact with.
            data: The Pandas DataFrame or Series to be logged.
            artifact_path: The relative path under which the artifact should be logged.
            output_file_type: The output file format for the data. Must be either 'parquet'
                    or 'csv'. Defaults to None.
            kwargs: Additional keyword arguments applicable to the saving function for
                    the specified file type (e.g., compression options).
        Return: None
        Raises:
            ValueError: If an unsupported file type is provided.
        """
        with self._log_artifact_helper(run_id=run_id, artifact_file=artifact_path) as tmp_path:
            if output_file_type is None:
                if FileHandler.is_parquet_file(artifact_path):
                    output_file_type = 'parquet'
                elif FileHandler.is_csv_file(artifact_path):
                    output_file_type = 'csv'
            if output_file_type == 'parquet':
                FileHandler.save_dataframe_as_parquet_file(data, tmp_path, **kwargs)
            elif output_file_type == 'csv':
                FileHandler.save_dataframe_as_csv_file(data, tmp_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file type: {output_file_type}. "
                                 f"Only .parquet and .csv are supported.")

    def log_as_pickle(self, run_id: str, data, artifact_path: str, **kwargs) -> None:
        """
        Logs the provided data as a pickle file to the specified artifact path.

        The method serializes the given `data` object into a pickle file and saves it to
        the specified `artifact_path`. If a `run_id` is provided, the artifact is logged
        under that specific run context.

        Parameters:
            run_id: The identifier of the run with which to associate the artifact.
            data: The data object to be serialized and saved as a pickle file.
            artifact_path: The path where the pickle file will be saved.
        Return: None
        """
        with self._log_artifact_helper(run_id=run_id, artifact_file=artifact_path) as tmp_path:
            FileHandler.save_pickle_file(data, tmp_path, **kwargs)

    def log_dict(self, run_id: str, data: dict, artifact_path: str, **kwargs) -> None:
        """
        Logs a dictionary as an artifact to the specified run.

        This method facilitates logging a Python dictionary object as an artifact.
        The dictionary is saved as a file into a temporary path, which is then logged
        to the specified artifact path within the desired run. If no run_id is provided,
        the method operates within an active or default run.

        Parameters:
            run_id: Optional identifier for the run where the artifact will be logged.
            data: The dictionary to be logged as an artifact.
            artifact_path: The relative path within the run where the dictionary artifact will be stored.
        Return: None
        """
        with self._log_artifact_helper(run_id=run_id, artifact_file=artifact_path) as tmp_path:
            FileHandler.save_dict(data, tmp_path, **kwargs)

    def log_file(self, run_id: str, data, artifact_path: FilePath, **kwargs) -> None:
        """
        Logs and saves a file artifact to MLflow using the specified `run_id`.

        This method handles different file types for logging, such as text files, pickle files,
        dataframes (CSV or Parquet), dictionaries (JSON or YAML), and image files. It determines
        the file type based on the artifact path and calls respective file-specific methods to
        log the data as artifacts within the active MLflow run. Unsupported file types will
        raise a `ValueError`.

        Parameters:
            run_id: Unique identifier for the MLflow run to which the artifact belongs.
            data: The data to be logged, which can vary in format depending on the file type.
            artifact_path: File path of the artifact to be logged. The file type is inferred
                from the suffix of this path.
            kwargs: Additional keyword arguments required for file-specific logging,
        Return: None
        """
        if FileHandler.is_pickle_file(artifact_path):
            self.log_as_pickle(run_id, data, artifact_path, **kwargs)
        elif FileHandler.is_text_file(artifact_path):
            self.log_text(run_id, data, artifact_path)
        elif FileHandler.is_parquet_file(artifact_path):
            self.log_dataframe(run_id, data, artifact_path, **kwargs)
        elif FileHandler.is_csv_file(artifact_path):
            self.log_dataframe(run_id, data, artifact_path, output_file_type='csv', **kwargs)
        elif FileHandler.is_json_file(artifact_path) or FileHandler.is_yaml_file(artifact_path):
            self.log_dict(run_id, data, artifact_path, **kwargs)
        elif FileHandler.is_image_file(artifact_path):
            self.log_figure(run_id, data, artifact_path, save_kwargs=kwargs)
        else:
            raise ValueError(f"Unsupported file type {Path(artifact_path).suffix}.")

    def log_files(self, run_id: str,  data: dict) -> None:
        """
        Logs multiple files into a given run identified by run_id. Each file is
        added as an artifact with the filename specified in the data dictionary's
        keys and corresponding file content from its values.

        Parameters:
        run_id: Unique identifier for the run to which the files
            will be logged.
        data: A dictionary containing filenames as keys and their
            corresponding file content as values to be logged into the run.
        Returns
            The ActiveRun object associated with the specified run_id after
            logging the files.
        """
        for file_name, file_content in data.items():
            self.log_file(run_id, data=file_content, artifact_path=file_name)

    @contextmanager
    def _load_artifact_helpers(self, artifact_path: FilePath, run_id: str):
        """
        Context manager for loading an artifact as a local directory.

        This method helps in managing the temporary download of an artifact to a local
        directory during its usage and ensures proper cleanup of resources.

        Parameters:
            artifact_path: Path to the artifact within the run context to be loaded.
            run_id: Run identifier for the context in which the artifact is located.
        Return:
            Yields the path of the artifact's local directory as a pathlib.Path object.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(self.download_artifacts(run_id, str(artifact_path), temp_dir))
            yield local_path

    def load_parquet_artifact(self, artifact_path: FilePath, run_id: str, **kwargs):
        """
        Loads a parquet artifact from the given path and returns it as a DataFrame.

        This method retrieves a stored parquet artifact by handling the artifact's
        path and associated run identifier. It uses a context manager to temporally
        manage the local handling of the artifact and invokes the appropriate file
        handler for loading the parquet file as a DataFrame. Additional keyword
        arguments can be passed for configuration during the loading process.

        Parameters:
            artifact_path: Path to the artifact to load.
            run_id: Identifier for the run associated with the artifact.
            kwargs: Additional keyword arguments for loading configuration.
        Return:
            The dataframe loaded from the parquet artifact.
        """
        with self._load_artifact_helpers(artifact_path, run_id=run_id) as tmp_path:
            df = FileHandler.load_parquet_file(tmp_path, **kwargs)
            return df

    def load_csv_artifact(self, artifact_path: FilePath, run_id: str, **kwargs):
        """
        Loads a CSV artifact from a specified artifact path associated with a given run ID. This
        function retrieves the artifact, temporarily stores it, and loads it as a DataFrame
        using an appropriate file handler.
        Parameters:
            artifact_path: The path to the artifact to be loaded.
            run_id: The identifier for the run associated with the artifact.
            kwargs: Additional keyword arguments to pass to the file loading function.
        Return:
            A DataFrame object containing the data from the CSV artifact.
        """
        with self._load_artifact_helpers(artifact_path, run_id=run_id) as tmp_path:
            df = FileHandler.load_csv_file(tmp_path, **kwargs)
            return df

    def load_text_artifact(self, artifact_path: FilePath, run_id: str):
        """
        Load a text artifact file from a specific run and return its content as a string. This
        function closes the temporary file handlers after reading, ensuring that resources
        are properly managed.

        Parameters:
            artifact_path: The path to the artifact file within the run's artifacts.
                           This is the relative path from the root directory of the run's  artifact storage.
            run_id: The unique identifier for the run. Used to locate the specific run's
                    artifact directory within the artifact storage system.
        Return:
            The content of the text artifact file as a string.
        """
        with self._load_artifact_helpers(artifact_path, run_id=run_id) as tmp_path:
            return FileHandler.load_text_file(tmp_path)

    def load_pickle_artifact(self, artifact_path: FilePath, run_id: str, **kwargs):
        """
        Load a pickle artifact from a given artifact path and run ID. This method uses the
        internal helper `_load_artifact_helpers` to retrieve and manage the temporary file
        path of the stored artifact, then leverages the `FileHandler.load_pickle_file` method
        to deserialize the content of the pickle file.

        Parameters:
            artifact_path: The relative path to the artifact within the storage system.
            run_id: The unique identifier of the run associated with the artifact.
        Return:
            The deserialized object loaded from the pickle file.
        """
        with self._load_artifact_helpers(artifact_path, run_id=run_id) as tmp_path:
            return FileHandler.load_pickle_file(tmp_path, **kwargs)

    def load_dataframe(self, artifact_path: FilePath, run_id: str, file_type: Optional[str] = None, **kwargs):
        """
        Loads a DataFrame artifact from the specified path and run ID. The method supports
        loading artifacts in both 'parquet' and 'csv' file formats. If an unsupported file
        type is provided, an exception will be raised. Additional keyword arguments are
        passed along to the specific loaders.

        Parameters:
            artifact_path: The path to the artifact to load.
            run_id: The identifier of the run from which the artifact is to be loaded.
            file_type: The type of file to load. Must be either 'parquet' or 'csv'. Defaults to 'parquet'.
            kwargs: Additional arguments to pass to the specific file loader.
        Return:
            A loaded DataFrame from the specified artifact path and run ID.
        Raises:
            ValueError: If an unsupported file type is provided.
        """
        if file_type is None:
            if FileHandler.is_parquet_file(artifact_path):
                file_type = 'parquet'
            elif FileHandler.is_csv_file(artifact_path):
                file_type = 'csv'
        if file_type == 'parquet':
            return self.load_parquet_artifact(artifact_path, run_id, **kwargs)
        elif file_type == 'csv':
            return self.load_csv_artifact(artifact_path, run_id, **kwargs)
        else:
            raise ValueError(f"Unsupported file type for artifact: {artifact_path}. "
                             f"Only .parquet and .csv are supported.")

    def load_dict(self, run_id: str, artifact_path: FilePath):
        """
        Loads a dictionary object from the specified artifact path within the given run context.

        This method utilizes an artifact helper to temporarily retrieve and handle the specified
        artifact path. It ensures proper handling of resources while accessing the desired data in
        the provided `run_id` and artifact location.

        Parameters:
            run_id: The unique identifier of the run from which the artifact is being loaded.
            artifact_path: The path to the artifact file that contains the dictionary to be loaded.
        Return:
            A dictionary object loaded from the specified artifact path.
        """
        with self._load_artifact_helpers(artifact_path, run_id=run_id) as tmp_path:
            return FileHandler.load_dict(tmp_path)

    def load_file(self, run_id: str, artifact_path: FilePath, **kwargs):
        """
        Load a file artifact from a specified path and run ID, using the appropriate
        file handler based on the artifact type. Supports various file formats
        including pickle, text, parquet, CSV, JSON, and YAML. Raises an error
        if the file format is not supported.

        Parameters:
            run_id: A unique identifier for the run from which the artifact is being loaded.
            artifact_path: Path to the artifact file to be loaded.
            kwargs: Additional keyword arguments required for file-specific
                loading, applicable for certain file formats like CSV or parquet.
        Return:
            The loaded data, whose format depends on the file type.
        Raises:
             ValueError: If the file format of the artifact is unsupported.
        """
        if FileHandler.is_pickle_file(artifact_path):
            data = self.load_pickle_artifact(artifact_path, run_id, **kwargs)
        elif FileHandler.is_text_file(artifact_path):
            data = self.load_text_artifact(artifact_path, run_id)
        elif FileHandler.is_parquet_file(artifact_path) or FileHandler.is_csv_file(artifact_path):
            data = self.load_dataframe(artifact_path, run_id, **kwargs)
        elif FileHandler.is_json_file(artifact_path) or FileHandler.is_yaml_file(artifact_path):
            data = self.load_dict(run_id, artifact_path)
        else:
            raise ValueError(f"Unsupported file type for artifact: {artifact_path}.")
        return data

    def load_files(self, run_id: str, artifact_path: FilePath) -> dict:
        """
        Fetch and load artifacts associated with a specific run into memory.

        Summary:
        This method retrieves artifacts associated with a provided run ID and artifact path. The artifacts
        are downloaded into a temporary directory. Based on the file extensions, different loading methods
        are applied (e.g., '.pkl', '.txt', or '.parquet') to the files, and the loaded artifacts are stored
        in a dictionary. Unsupported file types are skipped, with a warning logged. The method then returns
        the dictionary containing the loaded artifacts.

        Parameters:
            run_id (str): The unique identifier of the run whose artifacts are being fetched.
            artifact_path (str): The path to the directory containing the artifacts.

        Returns:
        dict: A dictionary where the keys are artifact path (with extensions), and the values are the loaded
              artifact contents.
        """
        logger.info(f"Fetching artifacts for run_id: {run_id}")
        artifacts = {}
        artifacts_path_list = [i.path for i in self.list_artifacts(run_id, artifact_path) if not i.is_dir]
        for file_path in artifacts_path_list:
            artifacts[file_path] = self.load_file(run_id, file_path)
        logger.info(f"Artifacts downloaded from {artifact_path}")
        return artifacts

    def get_run_params(self, run_id: str):
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

    def get_latest_model_version(self, model_name: str):
        """
        Fetches the latest version information of a model based on the given model name.

        This function retrieves all available versions of a specified model by querying
        the client. It then determines the latest version by comparing the version
        numbers of retrieved results. If no versions are found, an empty list is returned.

        Parameters:
            model_name: Name of the model for which the latest version information needs to be fetched.
        Return:
            The latest version information of the model, or an empty list if no versions are available.
        """
        # get all model versions
        model_versions = self.search_model_versions(f"name='{model_name}'")
        # find latest version
        if model_versions:
            return max(model_versions, key=lambda x: int(x.version))
        return []
