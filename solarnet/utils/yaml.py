# Author: Jonathan Donzallaz

from pathlib import Path

import yaml


def load_yaml(file: Path) -> dict:
    """
    Load dict from yaml file (yaml version 1.2)

    :param file: path to yaml file
    :return: dict of data
    """
    stream = open(file, 'w')



    data = yaml.load(stream, Loader=yaml.FullLoader)

    return data


def write_yaml(file: Path, data: dict):
    """
    Write dict to yaml file (yaml version 1.2)

    :param file: path to yaml file to create
    :param data: dict of data to write in the file
    """
    stream = open(file, 'w')
    yaml.width = 4096
    yaml.dump(data, stream)
