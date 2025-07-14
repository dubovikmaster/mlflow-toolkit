import json
import yaml
import pandas as pd
from pathlib import Path
from typing import (
    Any,
    Union,
    Dict,
    TypeAlias
)
import logging
import pickle
from os import PathLike

try:
    import joblib
    import dill
except ImportError:
    joblib = None
    dill = None

logger = logging.getLogger(__name__)


class FlowListDumper(yaml.Dumper):
    pass


def represent_list_as_flow(dumper, data):
    return dumper.represent_sequence(
        yaml.resolver.BaseResolver.DEFAULT_SEQUENCE_TAG,
        data,
        flow_style=True
    )


def represent_tuple_as_flow(dumper, data):
    return dumper.represent_sequence(
        'tag:yaml.org,2002:python/tuple',
        data,
        flow_style=True
    )


FlowListDumper.add_representer(list, represent_list_as_flow)
FlowListDumper.add_representer(tuple, represent_tuple_as_flow)

FilePath: TypeAlias = Union[str, bytes, PathLike]


class FileHandler:
    """
    Handles file operations including reading from and writing to various file formats.

    This class provides utility methods for handling files, such as reading and writing
    text, JSON, YAML, CSV, Parquet, and pickle files. It includes functionality for
    validating and preparing file paths, logging operations, and raising meaningful
    errors in case of failures.

    """

    @staticmethod
    def _file_path_prepare(path: FilePath):
        """
        Prepares and validates a file path input by ensuring it is of the correct
        type. If the input is not already a Path-like object, it converts the
        input into a Path instance.

        Parameters:
            path: The input path which can be a string, bytes, or an object  implementing os.PathLike.
        Return:
            A validated and converted Path object.
        """
        if not isinstance(path, PathLike):
            path = Path(path)
        return path

    @staticmethod
    def load_pickle_file(file_path: FilePath, backend: str = 'pickle', **kwargs) -> Any:
        """
        Static method to load and deserialize an object from a pickle file. It reads binary data
        from the given file path and reconstructs the original object using Python's pickle module.
        If an error occurs during file opening or deserialization, an exception is raised after
        logging the error.

        Parameters:
            file_path (Path): The path to the pickle file to be loaded.
            backend (str): The backend to use for loading the file. Defaults to 'pickle'.
            kwargs: Additional keyword arguments to be passed to the pickle load function.

        Returns:
            Any: The deserialized object from the pickle file.

        Raises:
        Exception: If the file cannot be opened or deserialization fails, an exception is raised
        with the corresponding error logged.
        """
        logger.info(f"Loading pickle file from {file_path}")
        try:
            if backend == 'pickle':
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            elif backend == 'dill':
                if dill is None:
                    raise ImportError("dill is not installed. Please install it to use this backend.")
                with open(file_path, 'rb') as f:
                    return dill.load(f)
            elif backend == 'joblib':
                if joblib is None:
                    raise ImportError("joblib is not installed. Please install it to use this backend.")
                return joblib.load(file_path, **kwargs)
            else:
                raise ValueError(
                    f"Unsupported backend: {backend}. Supported backends are 'pickle', 'dill', and 'joblib'.")
        except Exception as e:
            logger.error(f"Failed to load pickle file. Error: {e}")
            raise

    @staticmethod
    def load_text_file(file_path: FilePath) -> str:
        """
        Loads the content of a text file and returns it as a string. This method reads
        the file using UTF-8 encoding. In case of any error during the file reading
        operation, an exception is logged and re-raised to the caller.

        Parameters:
            file_path: The path of the text file to be loaded.

        Return:
            The content of the text file as a single string.

        Raises:
             Exception: For any other unexpected errors encountered during file operations.
        """
        try:
            logger.info(f"Loading text file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load text file {file_path}: {e}")
            raise

    @staticmethod
    def load_parquet_file(file_path: FilePath, **kwargs) -> pd.DataFrame:
        """
        This static method is designed to load a Parquet file from the specified file
        path and return its content as a pandas DataFrame. It employs the pandas
        library's `read_parquet` method to load the file and allows additional keyword
        arguments for customization. Logging is implemented to provide information
        on the loading process, including both successful operations and errors.

        Parameters:
            file_path: The file path of the Parquet file to be loaded.
            kwargs: Additional keyword arguments to be passed to `pd.read_parquet`.
        Return:
            A pandas DataFrame containing the data from the loaded Parquet file.
        Raises:
            Exception: If the loading process fails, the exception is caught, logged, and re-raised.
        """
        try:
            logger.info(f"Loading Parquet file from: {file_path}")
            df = pd.read_parquet(file_path, **kwargs)
            logger.info(f"Parquet file loaded from: {file_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load Parquet file {file_path}: {e}")
            raise

    @staticmethod
    def load_csv_file(file_path: FilePath, **kwargs) -> pd.DataFrame:
        """
        Loads a CSV file into a pandas DataFrame. The function attempts to read the
        file specified by the given file path, applying any additional keyword
        arguments passed to customize the loading process.

        Parameters:
            file_path: Path to the CSV file to be loaded.
            kwargs: Additional keyword arguments passed to `pandas.read_csv`
            to customize the file reading behavior.
        Returns:
            A pandas DataFrame containing the data from the CSV file.
        Raises:
            Exception: If the CSV file cannot be loaded due to issues such as
            file not found, format mismatch, or other I/O errors.
        """
        try:
            logger.info(f"Loading CSV file from: {file_path}")
            df = pd.read_csv(file_path, **kwargs)
            logger.info(f"CSV file loaded from: {file_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load CSV file {file_path}: {e}")
            raise

    @staticmethod
    def load_dict(file_path: FilePath, **kwargs) -> Dict:
        """
        Loads a dictionary from a specified file path. The file type must be either JSON or YAML.
        If the file type is unsupported, a ValueError is raised. This method ensures that the file
        path is properly prepared and validated before attempting to read the content. The method
        also logs any errors that occur during the process.

        Parameters:
            file_path: Path to the file to be loaded as a dictionary.
            kwargs: Additional keyword arguments to be passed to the appropriate deserialization method.
        Returns:
            Dictionary containing file data. The content and structure depend on the loaded file.
        Raises:
            - ValueError: If the file type is unsupported.
            - Exception: If an error occurs during file reading or content parsing.
        """
        try:
            file_path = FileHandler._file_path_prepare(file_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                if FileHandler.is_json_file(file_path):
                    return json.load(f, **kwargs)
                if FileHandler.is_yaml_file(file_path):
                    return yaml.load(f, yaml.FullLoader)
                raise ValueError(f"Unsupported file type for {file_path}")
        except Exception as e:
            logger.error(f"Failed to load dictionary from {file_path}: {e}")
            raise

    @staticmethod
    def save_dict(data: Dict, file_path: FilePath, **kwargs) -> None:
        """
        Save a dictionary to a specified file path in either JSON or YAML format. The
        method determines the file format based on the file extension provided in the
        ``file_path`` and utilizes the appropriate serialization technique. JSON files
        are saved with indentations for readability, and YAML files are dumped with
        FlowListDumper while supporting Unicode and indentation. If the given file
        extension is not supported, an error is raised.

        Parameters:
            data: Dictionary to be saved.
            file_path: Path to the target file, including the file name and extension.
                Supported extensions are `.json`, `.yml` or `.yaml`.
            kwargs: Additional keyword arguments to be passed to the appropriate serialization method.
        Raises:
            - ValueError: If the file type in `file_path` is not supported.
            - Exception: If any other issue occurs while saving the file.
        Returns: None
        """
        try:
            file_path = FileHandler._file_path_prepare(file_path)
            with open(file_path, 'w', encoding='utf-8') as f:
                if FileHandler.is_json_file(file_path):
                    json.dump(data, f, **kwargs)
                elif FileHandler.is_yaml_file(file_path):
                    yaml.dump(data, f, Dumper=FlowListDumper, **kwargs)
                else:
                    raise ValueError(f"Unsupported file type for {file_path}")
        except Exception as e:
            logger.error(f"Failed to save dictionary to {file_path}: {e}")
            raise

    @staticmethod
    def save_pickle_file(obj, file_path: FilePath, backend: str = 'pickle', **kwargs):
        """
        Save an object to a pickle file.

        This static method takes an object and saves it as a pickle file at the specified file path.
        It logs messages to indicate success or failure.

        Parameters:
            obj: The object to be saved as a pickle file.
            file_path (Path): The directory path where the pickle file will be saved.
            backend: The backend to use for saving the file. Defaults to 'pickle'.
            kwargs: Additional keyword arguments to be passed to the pickle dump function.

        Raises:
            Exception: If an error occurs during the process of saving the file, it logs the error message
            and re-raises the exception.
        """
        logger.info(f"Saving {file_path} has begun")
        try:
            if backend == 'pickle':
                with open(file_path, 'wb') as file:
                    pickle.dump(obj, file)
            elif backend == 'dill':
                if dill is None:
                    raise ImportError("dill is not installed. Please install it to use this backend.")
                with open(file_path, 'wb') as file:
                    dill.dump(obj, file)
            elif backend == 'joblib':
                if joblib is None:
                    raise ImportError("joblib is not installed. Please install it to use this backend.")
                joblib.dump(obj, file_path, **kwargs)
            else:
                raise ValueError(
                    f"Unsupported backend: {backend}. Supported backends are 'pickle', 'dill', and 'joblib'.")
            logger.info(f"{file_path} saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save {file_path}. Error: {e}")
            raise

    @staticmethod
    def save_text_file(text: str, file_path: FilePath):
        """
        Static method that saves a given string to a text file at the specified
        file path. Logs the process and raises
        exceptions if there are any errors during the operation.

        Parameters:
            text: str
                The string content to be written to the file.
            file_path: Path
                The base directory path where the file should be saved.

        Raises:
        Exception
            If the text file cannot be written due to an error.
        """
        logger.info(f"Saving text to {file_path}")
        try:
            with open(file_path, 'w') as file:
                file.write(text)
            logger.info(f"{file_path} saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save {file_path}. Error: {e}")
            raise

    @staticmethod
    def save_dataframe_as_parquet_file(df: Union[pd.DataFrame, pd.Series], file_path: FilePath, **kwargs):
        """
        Save a DataFrame or Series object as a parquet file in the specified location using the PyArrow engine.
        The function handles both DataFrame and Series objects appropriately. In case of an error during the
        save operation, an exception is logged and re-raised.

        Parameters:
            df (pd.DataFrame | pd.Series): The data object to save, which can be either a Pandas DataFrame or
                                            Series.
            file_path (Path): The directory in which the parquet file should be saved.

        Raises:
        Exception: If the parquet file could not be saved due to any reason, the exception is logged and
                   re-raised.
        """
        logger.info(f"Saving dataframe to {file_path}")
        try:
            if isinstance(df, pd.DataFrame):
                df.to_parquet(file_path, **kwargs)
            elif isinstance(df, pd.Series):
                df.to_frame().to_parquet(file_path, **kwargs)
            logger.info(f"Dataframe saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save dataframe to {file_path}. Error: {e}")
            raise

    @staticmethod
    def save_dataframe_as_csv_file(df: Union[pd.DataFrame, pd.Series], file_path: FilePath, **kwargs):
        """
        Saves a pandas DataFrame or Series to a CSV file.

        This function allows saving the provided pandas DataFrame or Series to the
        specified file path in CSV format. Additional parameters for the `to_csv`
        method can be passed via keyword arguments. In case of an error during the
        saving process, an exception is raised after logging the error details.

        Parameters:
            df: A pandas DataFrame or Series to be saved as a CSV file.
            file_path: The destination file path where the CSV file will be saved.
            kwargs: Additional keyword arguments passed to the `to_csv` method.
        Return: None
        """
        logger.info(f"Saving dataframe to {file_path}")
        try:
            df.to_csv(file_path, **kwargs)
            logger.info(f"Dataframe saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save dataframe to {file_path}. Error: {e}")
            raise

    @staticmethod
    def is_parquet_file(file_path: FilePath) -> bool:
        """
        Checks whether the given file path corresponds to a Parquet file. This is inferred
        based on the file extensions ``.parq`` or ``.parquet``. The method supports both
        string file paths and `Path` objects. If the input is a string, it will be
        converted to a `Path` object for validation.

        Parameters:
            file_path: The path of the file to check. This can either be a string or a `Path` object.
        Return:
            ``True`` if the file has a Parquet file extension, otherwise ``False``.
        """
        file_path = FileHandler._file_path_prepare(file_path)
        return any(suffix in ['.parq', '.parquet'] for suffix in file_path.suffixes)

    @staticmethod
    def is_csv_file(file_path: FilePath) -> bool:
        """
        Determines whether the given file is a CSV file based on its suffix.

        This method checks if the provided file path ends with a `.csv` extension.
        It works by analyzing the suffixes of the file path. The function supports
        both `Path` objects and string representations of file paths. If the input
        is not a `Path` object, it will attempt to convert it using `Path`.

        Parameters:
            file_path: The path to the file, provided as a string or a `Path` object.
        Return:
            Returns `True` if the file has a `.csv` extension, otherwise `False`.
        """
        file_path = FileHandler._file_path_prepare(file_path)
        return any(suffix == '.csv' for suffix in file_path.suffixes)

    @staticmethod
    def is_image_file(file_path: FilePath) -> bool:
        """
        Checks if the given file path corresponds to an image file. The function evaluates
        whether the file's suffix matches common image file extensions, such as `.jpg`,
        `.jpeg`, `.png`, or `.bmp`.

        Parameters:
            file_path: The path to the file, provided as a string or a `Path` object.
        Return:
            True if the file is an image file; otherwise, False.
        """
        file_path = FileHandler._file_path_prepare(file_path)
        return any(suffix in ['.jpg', '.jpeg', '.png', '.bmp'] for suffix in file_path.suffixes)

    @staticmethod
    def is_text_file(file_path: FilePath) -> bool:
        """
        Checks if a given file is a text-based file. The method evaluates the file
        suffixes to determine if the file falls into the categories of `.txt`, `.md`,
        or `.html` files to confirm if it is a text file.

        Parameters:
            file_path: The path to the file, provided as a string or a `Path` object.
        Return:
            True if the file's suffix matches any of the predefined text file types; False otherwise.
        """
        file_path = FileHandler._file_path_prepare(file_path)
        return any(suffix in ['.txt', '.md', '.html'] for suffix in file_path.suffixes)

    @staticmethod
    def is_json_file(file_path: FilePath) -> bool:
        """
        Determines if the given file path corresponds to a JSON file based on its suffix.

        This method examines the suffixes of the provided file path to check if
        any match the '.json' suffix. Only valid JSON file suffixes will return
        a positive result. Input is prepared internally to ensure compatibility.

        Parameters:
            file_path: The path to the file, provided as a string or a `Path` object.
        Return:
            True if the file path indicates a JSON file, False otherwise.
        """
        file_path = FileHandler._file_path_prepare(file_path)
        return any(suffix in ['.json'] for suffix in file_path.suffixes)

    @staticmethod
    def is_yaml_file(file_path: FilePath) -> bool:
        """
        Determines if a given file path corresponds to a YAML file.

        This static method checks if the file at the given path has a
        suffix of either `.yml` or `.yaml`, indicating it is a YAML file.

        Parameters:
            file_path: The path to the file, provided as a string or a `Path` object.
        Return:
            True if the file is a YAML file, False otherwise.
        """
        file_path = FileHandler._file_path_prepare(file_path)
        return any(suffix in ['.yml', '.yaml'] for suffix in file_path.suffixes)

    @staticmethod
    def is_pickle_file(file_path: FilePath) -> bool:
        """
        Checks if the provided file path corresponds to a pickle file.

        This method evaluates the suffixes of the given file path to determine
        if it has an extension commonly associated with pickle files.

        Parameters:
            file_path: The path to the file, provided as a string or a `Path` object.
        Return:
            True if the file is a pickle file, False otherwise.
        """
        file_path = FileHandler._file_path_prepare(file_path)
        return any(suffix in ['.pkl', '.pickle', '.joblib', '.dill'] for suffix in file_path.suffixes)
