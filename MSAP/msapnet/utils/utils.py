"""
Misc functions used elsewhere
"""

import os
from argparse import ArgumentParser

import yaml
import torch
import numpy as np


def progress_bar(i: int, total: int, text: str = ""):
    """
    Terminal progress bar

    Parameters
    ----------
    i : integer
        Current progress
    total : integer
        Completion number
    text : string, default = '
        Optional text to place at the end of the progress bar
    """
    length = 50
    i += 1

    filled = int(i * length / total)
    percent = i * 100 / total
    bar_fill = "█" * filled + "-" * (length - filled)
    print(f"\rProgress: |{bar_fill}| {int(percent)}%\t{text}\t", end="")

    if i == total:
        print()


def even_length(x: torch.Tensor) -> torch.Tensor:
    """
    Returns a tensor of even length in the last
    dimension by merging the last two values

    Parameters
    ----------
    x : Tensor
        Input data

    Returns
    -------
    Tensor
        Output data with even length
    """
    if x.size(-1) % 2 != 0:
        x = torch.cat(
            (x[..., :-2], torch.mean(x[..., -2:], dim=-1, keepdim=True)), dim=-1
        )

    return x


# def data_normalization(
#         data,
#         mean: bool = True,
#         axis: int = None,
#         transform: tuple[float, float] = None) -> tuple[np.ndarray, tuple[float, float]]:
#     """
#     Transforms data either by normalising or
#     scaling between 0 & 1 depending on if mean is true or false.

#     Parameters
#     ----------
#     data : ndarray
#         Data to be normalised
#     mean : boolean, default = True
#         If data should be normalised or scaled between 0 and 1
#     axis : integer, default = None
#         Which axis to normalise over, if none, normalise over all axes
#     transform: tuple[float, float], default = None
#         If transformation values exist already

#     Returns
#     -------
#     tuple[ndarray, tuple[float, float]]
#         Transformed data & transform values
#     """
#     if axis is not None:
#         if mean and not transform:
#             transform = [torch.mean(data, dim=axis), torch.std(data, dim=axis)]
#         elif not mean and not transform:
#             transform = [
#                 torch.min(data, dim=axis),
#                 torch.max(data, dim=axis) - torch.min(data, dim=axis)
#             ]

#         if len(transform[0].shape):
#             data = (data - transform[0].unsqueeze(axis)) /\
#                 transform[1].unsqueeze(axis)
#         else:
#             data = (data - transform[0]) / transform[1]

#         return data, transform
#     else:
#         if mean and not transform:
#             transform = [torch.mean(data), torch.std(data)]
#         elif not mean and not transform:
#             transform = [
#                 torch.min(data),
#                 torch.max(data) - torch.min(data)
#             ]

#         if len(transform[0].shape):
#             data = (data - transform[0]) /\
#                 transform[1]
#         else:
#             data = (data - transform[0]) / transform[1]

#         return data, transform


def file_names(
    data_dir: str, blacklist: list[str] = None, whitelist: str = None
) -> np.ndarray:
    """
    Fetches the file names of all spectra that are in the whitelist, if not None,
    or not on the blacklist, if not None

    Parameters
    ----------
    data_dir : string
        Directory of the spectra dataset
    blacklist : list[string], default = None
        Exclude all files with substrings
    whitelist : string, default = None
        Require all files have the substring

    Returns
    -------
    ndarray
        Array of spectra file names
    """
    # Fetch all files within directory
    files = np.sort(np.array(os.listdir(data_dir)))

    # Remove all files that aren't whitelisted
    if whitelist:
        files = np.delete(files, np.char.find(files, whitelist) == -1)

    # Remove all files that are blacklisted
    for substring in blacklist:
        files = np.delete(files, np.char.find(files, substring) != -1)

    return files


def open_config(
    idx: int, config_path: str, parser: ArgumentParser = None
) -> tuple[str, dict]:
    """
    Opens the configuration file from either the provided path or through command line argument

    Parameters
    ----------
    idx : integer
        Index of the configuration file
    config_path : string
        Default path to the configuration file
    parser : ArgumentParser, default = None
        Parser if arguments other than config path are required

    Returns
    -------
    tuple[string, dictionary]
        Configuration path and configuration file dictionary
    """
    if not parser:
        parser = ArgumentParser()

    parser.add_argument(
        "--config_path",
        default=config_path,
        help="Path to the configuration file",
        required=False,
    )
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    config_path = args.config_path
    # print(config_path)

    with open(config_path, "r", encoding="utf-8") as file:
        config = list(yaml.safe_load_all(file))[idx]

    return config_path, config
