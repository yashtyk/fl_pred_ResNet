import logging
import os
from enum import Enum
import argparse
import yaml





from solarnet.tasks.test import test
from solarnet.tasks.train import train_standard
from solarnet.tasks.test_on_two import test_new, draw

from solarnet.tasks.train_multi_deep import train_multi_deep
from solarnet.tasks.test_multi_deep import test_multi_deep

parser = argparse.ArgumentParser(description="...")
parser.add_argument("--config_path", default="/Users/yanashtyk/Documents/GitHub/ResNet/config/experiment/resnet_deep.yaml", type=str, help="The config file path")
parser.add_argument('--task', default='train', type = str, help = 'Type of task (train , train_multi_deep, test, test_multi) ')
args = parser.parse_args()



os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"





def train_command(
    config,
    verbose: bool = False
):



    print(config)

    training_type = config["training_type"]
    if training_type == "train":
        train_standard(config)

def train_multi_deep_comand(config, verbose  =False):
    train_multi_deep(config)

def test_multi_deep_command(config, verbose = False):
    test_multi_deep(config)


def test_command(
    config,
    verbose: bool = False,
):



    print(config)

    test(config, verbose)



def test_multi_command(
    config,
    verbose: bool = False
):


    print(config)

    test_new(config, verbose)


def draw_command(config):
    draw(config)









class Split(str, Enum):
    train = "train"
    val = "val"
    test = "test"







if __name__ == '__main__':
    # --- configs and constants ----------------------------------------------------------------------------------------
    with open(args.config_path) as file:
        parameters = yaml.load(file, Loader=yaml.FullLoader)
    if args.task == 'train':
        train_command(parameters)

    if args.task == 'train_multi_deep':
        train_multi_deep_comand(parameters)

    if args.task == 'test_multi_deep':
        test_multi_deep_command(parameters)

    if args.task == 'test':
        test_command(parameters)

    if args.task == 'test_multi':
        test_multi_command(parameters)


    if args.task == 'draw':
        draw(parameters)