"""
Format registry: maps file suffixes to save/load handlers.

Built-in formats (parquet, csv, feather, json, yaml, pickle, text, numpy, images)
are registered at import time. Users can plug in their own formats:

    from mlflow_toolkit import register_format

    register_format(
        'excel', ['.xlsx'],
        save=lambda df, path, **kw: df.to_excel(path, **kw),
        load=lambda path, **kw: pd.read_excel(path, **kw),
    )

After that `MLflowWorker.log_file` / `load_file` / `load_files` handle the new
suffix like any built-in one.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from mlflow_toolkit.utils import (
    FileHandler,
    FilePath,
    pl,
)

__all__ = ['FormatHandler', 'register_format', 'get_format_handler', 'registered_suffixes']


@dataclass(frozen=True)
class FormatHandler:
    """
    A save/load pair for one file format.

    Attributes:
        name: Human-readable format name (used in error messages).
        suffixes: File suffixes handled by this format, e.g. ('.yml', '.yaml').
        save: Callable ``save(data, local_path, **kwargs)`` or None if the format is load-only.
        load: Callable ``load(local_path, **kwargs) -> data`` or None if the format is save-only.
    """
    name: str
    suffixes: tuple[str, ...]
    save: Callable[..., None] | None = None
    load: Callable[..., Any] | None = None


_registry: dict[str, FormatHandler] = {}


def register_format(name: str, suffixes, save: Callable[..., None] | None = None,
                    load: Callable[..., Any] | None = None, *, overwrite: bool = False) -> FormatHandler:
    """
    Register a file format so that `log_file` / `load_file` can handle its suffixes.

    Parameters:
        name: Human-readable format name.
        suffixes: Iterable of file suffixes (with or without the leading dot).
        save: Callable ``save(data, local_path, **kwargs)`` writing `data` to `local_path`.
            None makes the format load-only.
        load: Callable ``load(local_path, **kwargs) -> data`` reading `local_path`.
            None makes the format save-only.
        overwrite: Allow replacing a handler for an already-registered suffix.
    Return:
        The registered FormatHandler.
    Raises:
        ValueError: If neither `save` nor `load` is provided, or a suffix is already
            registered and `overwrite` is False.
    """
    if save is None and load is None:
        raise ValueError("At least one of 'save' or 'load' must be provided.")
    normalized = tuple(s.lower() if s.startswith('.') else f'.{s.lower()}' for s in suffixes)
    if not normalized:
        raise ValueError("At least one suffix must be provided.")
    handler = FormatHandler(name=name, suffixes=normalized, save=save, load=load)
    for suffix in normalized:
        existing = _registry.get(suffix)
        if existing is not None and not overwrite:
            raise ValueError(f"Suffix {suffix!r} is already registered for format {existing.name!r}. "
                             f"Pass overwrite=True to replace it.")
        _registry[suffix] = handler
    return handler


def get_format_handler(file_path: FilePath) -> FormatHandler | None:
    """
    Find the registered handler for a file path by its suffixes.

    Suffixes are checked from the last one backwards, so compressed names like
    ``data.csv.gz`` resolve to the 'csv' handler.

    Parameters:
        file_path: The path whose format should be resolved.
    Return:
        The matching FormatHandler, or None if no suffix is registered.
    """
    path = FileHandler._file_path_prepare(file_path)
    for suffix in reversed(path.suffixes):
        handler = _registry.get(suffix.lower())
        if handler is not None:
            return handler
    return None


def registered_suffixes() -> list[str]:
    """Return all registered file suffixes, sorted alphabetically."""
    return sorted(_registry)


def _save_feather(data, file_path: FilePath, **kwargs) -> None:
    data = FileHandler._normalize_polars(data)
    if pl is not None and isinstance(data, pl.DataFrame):
        data.write_ipc(file_path, **kwargs)
    elif isinstance(data, pd.Series):
        data.to_frame().to_feather(file_path, **kwargs)
    elif isinstance(data, pd.DataFrame):
        data.to_feather(file_path, **kwargs)
    else:
        raise TypeError(f"Unsupported dataframe type: {type(data)!r}")


def _load_feather(file_path: FilePath, backend: str | None = None, **kwargs):
    backend = FileHandler._resolve_dataframe_backend(backend)
    if backend == 'polars':
        return pl.read_ipc(file_path, **kwargs)
    return pd.read_feather(file_path, **kwargs)


def _save_npy(data, file_path: FilePath, **kwargs) -> None:
    np.save(file_path, data, **kwargs)


def _load_npy(file_path: FilePath, **kwargs):
    return np.load(file_path, **kwargs)


def _save_npz(data, file_path: FilePath, **kwargs) -> None:
    if isinstance(data, dict):
        np.savez(file_path, **data)
    else:
        np.savez(file_path, data)


def _load_npz(file_path: FilePath, **kwargs) -> dict:
    # materialize while the temp file still exists: NpzFile reads lazily
    with np.load(file_path, **kwargs) as npz:
        return dict(npz)


def _save_figure(figure, file_path: FilePath, **kwargs) -> None:
    if hasattr(figure, 'savefig'):  # matplotlib
        figure.savefig(file_path, **kwargs)
    elif hasattr(figure, 'write_image'):  # plotly
        figure.write_image(str(file_path), **kwargs)
    else:
        raise TypeError(f"Cannot save {type(figure)!r} as an image: "
                        f"expected a matplotlib or plotly figure.")


register_format('parquet', ('.parquet', '.parq'),
                save=FileHandler.save_dataframe_as_parquet_file, load=FileHandler.load_parquet_file)
register_format('csv', ('.csv',),
                save=FileHandler.save_dataframe_as_csv_file, load=FileHandler.load_csv_file)
register_format('feather', ('.feather',), save=_save_feather, load=_load_feather)
register_format('json', ('.json',), save=FileHandler.save_dict, load=FileHandler.load_dict)
register_format('yaml', ('.yml', '.yaml'), save=FileHandler.save_dict, load=FileHandler.load_dict)
register_format('pickle', ('.pkl', '.pickle', '.dill', '.joblib'),
                save=FileHandler.save_pickle_file, load=FileHandler.load_pickle_file)
register_format('text', ('.txt', '.md', '.html'),
                save=FileHandler.save_text_file, load=FileHandler.load_text_file)
register_format('numpy', ('.npy',), save=_save_npy, load=_load_npy)
register_format('numpy-archive', ('.npz',), save=_save_npz, load=_load_npz)
register_format('image', ('.jpg', '.jpeg', '.png', '.bmp', '.svg'), save=_save_figure)
