import datetime as dt
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from solarnet.data.dataset_utils import BaseDataset
from solarnet.utils.data import train_test_split

#from solarnet.data.my_transform import  AddGaussianNoise, MyRotationTransform



import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

class SDOBenchmarkDataset_deep(BaseDataset):
    def __init__(
        self,
        csv_file: Path,
        root_folder: Path,
        channel="171",
        transform: Optional[Callable] = None,
        #target_transform: Optional[Callable] = None,
        time_steps: Union[int, List[int]] = 0,
        stage = 'fit'
    ):
        metadata = pd.read_csv(csv_file, parse_dates=["start", "end"])

        self.root_folder = root_folder
        self.channel = channel
        self.transform = transform
        #self.target_transform = target_transform

        self.time_steps_values = [0, 7 * 60, 10 * 60 + 30, 11 * 60 + 50]
        self.time_steps = time_steps if isinstance(time_steps, list) else [time_steps]

        self.setup(metadata)

    def setup(self, metadata):
        ls = []
        for i in range(len(metadata)):
            sample_metadata = metadata.iloc[i]
            target = sample_metadata["peak_flux"]
            '''
            if self.target_transform is not None and \
                isinstance(self.target_transform(target), int) and \
                self.target_transform(target) < 0:
                # Ignore sample if it is not part of a class
                continue
            '''

            sample_active_region, sample_date = sample_metadata["id"].split("_", maxsplit=1)

            paths: List[Path] = []
            paths_check = []

            for time_step in self.time_steps:
                for ch in ['magnetogram', '94', '193', '211', '335' , '131', 'continuum', '1700', '304', '171']:
                    image_date = sample_metadata["start"] + dt.timedelta(minutes=self.time_steps_values[time_step])
                    image_date_str = dt.datetime.strftime(image_date, "%Y-%m-%dT%H%M%S")
                    image_name = f"{image_date_str}__{ch}.jpg"
                    paths_check.append(Path(sample_active_region) / sample_date / image_name)

            if not all((self.root_folder / path).exists() for path in paths_check):

                continue

            for time_step in self.time_steps:
                for chn in self.channel:
                    image_date = sample_metadata["start"] + dt.timedelta(minutes=self.time_steps_values[time_step])
                    image_date_str = dt.datetime.strftime(image_date, "%Y-%m-%dT%H%M%S")
                    image_name = f"{image_date_str}__{chn}.jpg"
                    paths.append(Path(sample_active_region) / sample_date / image_name)

            ls.append((paths, target))

        self.ls = ls

    def __len__(self) -> int:
        return len(self.ls)
    def transform_func(self, flux):
        if flux >= 1e-6:
            return 1
        return 0
    def __getitem__(self, index):
        metadata = self.ls[index]
        target = metadata[1]
        images = [Image.open(self.root_folder / path) for path in metadata[0]]
        to_tensor = transforms.ToTensor()
        images = [to_tensor(image) for image in images]

        if self.transform:
            images = [self.transform(image) for image in images]

        target = self.transform_func(target)

        if not torch.is_tensor(images[0]):
            return images[0], target

        # Put images of different time steps as one image of multiple channels (time steps ~ rgb)
        image = torch.cat(images, 0)

        return image, target

    def y(self, indices: Optional[Sequence[int]] = None) -> list:
        ls = self.ls
        if indices is not None:
            ls = (self.ls[i] for i in indices)


        return [self.transform_func(y[1]) for y in ls]

        #return [y[1] for y in ls]


class SDOBenchmarkDataModule_deep(pl.LightningDataModule):
    def __init__(
        self, dataset_dir: Path,
        channel: str = '171',
        batch_size: int = 32,
        num_workers: int = 0,
        validation_size: float = 0.1,
        resize: int = 64,
        seed: int = 42,
        transform: str = 'None',

        target_transform: Callable[[float], any] = None,
        train: bool =False,
        time_steps: Union[int, List[int]] = 0,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.channel = channel
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_size = validation_size
        self.seed = seed
        if train:
            print('train')

            self.transform_train = transforms.Compose([
                    transforms.Resize(resize),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                    #transforms.RandomHorizontalFlip(0.5),
                    #transforms.RandomVerticalFlip(0.5),
                    #MyRotationTransform(),
                    #AddGaussianNoise(),


            ])
            '''
            if transform == 'noise':
                self.transform_train = transforms.Compose([
                    transforms.Resize(resize),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                    # transforms.RandomHorizontalFlip(0.5),
                    # transforms.RandomVerticalFlip(0.5),
                    # MyRotationTransform(),
                    AddGaussianNoise(),

                ])
            if transform == 'rotate':
                self.transform_train = transforms.Compose([
                    transforms.Resize(resize),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    MyRotationTransform(),
                    #AddGaussianNoise(),

                ])
            if transform == 'both':
                self.transform_train = transforms.Compose([
                    transforms.Resize(resize),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    MyRotationTransform(),
                    AddGaussianNoise(),

                ])

            '''
            self.transform_val = transforms.Compose([
                transforms.Resize(resize),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
            self.transform_test = transforms.Compose([
                transforms.Resize(resize),
                transforms.Normalize(mean=[0.5], std=[0.5]),

            ])
        else:
            print('test')
            self.transform_test = transforms.Compose([
                transforms.Resize(resize),
                transforms.Normalize(mean=[0.5], std=[0.5]),

            ])

        self.target_transform = target_transform
        self.time_steps = time_steps

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.dataset_train_all = SDOBenchmarkDataset_deep(self.dataset_dir / 'training' / 'meta_data.csv',
                                                         self.dataset_dir / 'training',
                                                          channel=self.channel,
                                                         time_steps=self.time_steps, stage = 'fit')

            self.dataset_val_all = SDOBenchmarkDataset_deep(self.dataset_dir / 'training' / 'meta_data.csv',
                                                       self.dataset_dir / 'training',
                                                        channel=self.channel,
                                                       time_steps=self.time_steps, stage = 'fit')

            #obtain training indices that will be used for validation

            num_train = len(self.dataset_train_all)
            indices = list(range(num_train))
            np.random.shuffle(indices)
            split=int(np.floor(self.validation_size*num_train))
            train_idx, valid_idx = indices[split:], indices[:split]

            #define samplers for obtaining training and validation batches

            self.train_sampler = SubsetRandomSampler(train_idx)
            self.valid_sampler = SubsetRandomSampler(valid_idx)


            self.dataset_train = Subset(self.dataset_train_all, train_idx)
            #self.dataset_val=Subset(self.dataset_val_all, valid_idx)
            #self.dataset_train, self.dataset_val = train_test_split(self.dataset_train_val, self.validation_size)

            self.dims = tuple(self.dataset_val_all[0][0].shape)

        if stage == 'test' or stage is None:
            print(stage)
            self.dataset_test = SDOBenchmarkDataset_deep(self.dataset_dir / 'test' / 'meta_data.csv',
                                                    self.dataset_dir / 'test', transform=self.transform_test,
                                                    channel=self.channel,
                                                    time_steps=self.time_steps)
            self.dims = tuple(self.dataset_test[0][0].shape)

    def train_dataloader(self):
        #return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
         return DataLoader(self.dataset_train_all, batch_size=self.batch_size, sampler=self.train_sampler, num_workers=self.num_workers)

    def val_dataloader(self):
        #return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)
         return DataLoader(self.dataset_val_all, batch_size=self.batch_size, sampler=self.valid_sampler, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers)
