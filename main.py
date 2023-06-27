#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import pickle as pickle


DATA_PATH = Path('./data/data_acc_rot.dat')


def main() -> None:
    """Entry point of the program."""

    data = load_data(DATA_PATH)


def load_data(data_path: Path):
    """Load the pedestrian data from the specified path."""
    with open(data_path, 'rb') as data_file:
        data = pickle.load(data_file, encoding='latin1')
    return data


if __name__ == '__main__':
    main()
