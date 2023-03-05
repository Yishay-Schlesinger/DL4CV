import torch
import numpy as np
import copy
import random
from torchvision import datasets
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.auto import tqdm
from torchvision.transforms import ToTensor
from data_utils import lift_image, naive_discretize_to_lattice


class DatasetLoader:
    def __init__(self, dataset_name, data_root, train_batch_size, test_batch_size, device,
                 train_transform_fn, test_transform_fn, train_split=None, test_split=None, **kwargs):
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.train_split = train_split
        self.test_split = test_split
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_transform_fn = train_transform_fn
        self.test_transform_fn = test_transform_fn
        self.device = device
        self.data_loader_kwargs = kwargs.get('data_loader_kwargs',
                                             {'num_workers': 1, 'pin_memory': True} if self.device == 'cuda' else {})
        self.train_targets = kwargs.get('train_targets', None)
        self.test_targets = kwargs.get('test_targets', None)

        if len(self.train_batch_size) > 1: assert len(self.train_batch_size) == len(self.train_targets)
        if len(self.test_batch_size) > 1: assert len(self.test_batch_size) == len(self.test_targets)
        if self.train_targets is None: assert len(self.train_batch_size) == 1
        if self.test_targets is None: assert len(self.test_batch_size) == 1

    def get_data_loaders(self):
        if self.dataset_name == 'MNIST':
            train_dataset = datasets.MNIST(root=self.data_root, train=True, download=True,
                                           transform=self.train_transform_fn)
            test_dataset = datasets.MNIST(root=self.data_root, train=False, download=True,
                                          transform=self.test_transform_fn)
        elif self.dataset_name == 'CelebA':
            # train_dataset = datasets.CelebA(root=self.data_root, split='train', download=True,
            #                                 transform=self.train_transform_fn)
            # test_dataset = datasets.CelebA(root=self.data_root, split='test', download=True,
            #                                transform=self.test_transform_fn)
            raise ValueError('CelebA not yet supported')
        else:
            raise ValueError('Unsupported dataset name: {}'.format(self.dataset_name))

        train_loaders = self.create_loaders(train_dataset, self.train_targets, self.train_batch_size)
        test_loaders = self.create_loaders(test_dataset, self.test_targets, self.test_batch_size)
        return train_loaders, test_loaders

    def create_loaders(self, full_dataset, given_targets, batch_sizes):
        if len(batch_sizes) == 1:       # uncontrolled batch distribution
            if given_targets is None:   # take all targets
                loader = DataLoader(full_dataset, batch_size=batch_sizes[0], shuffle=True, **self.data_loader_kwargs)
            else:
                mask = [target in given_targets for target in full_dataset.targets]
                indices = [i for i, m in enumerate(mask) if m]
                for i, t in enumerate(given_targets):       # rename the target classes (e.g {0, 2, 7} -> {0, 1, 2})
                    full_dataset.targets[full_dataset.targets == t] = i
                data_subset = torch.utils.data.Subset(full_dataset, indices)
                loader = DataLoader(data_subset, batch_size=batch_sizes[0], shuffle=True, **self.data_loader_kwargs)
        else:
            og_targets = full_dataset.targets.detach().clone()
            for i, t in enumerate(given_targets):
                full_dataset.targets[full_dataset.targets == t] = i
            sampler = TargetBatchSampler(full_dataset, given_targets, og_targets, batch_sizes)
            loader = DataLoader(full_dataset, batch_sampler=sampler, **self.data_loader_kwargs)
        return loader


class TargetBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, target_classes, og_targets, num_samples_per_class):
        self.dataset = dataset
        self.target_classes = target_classes
        self.og_targets = og_targets
        self.num_samples_per_class = num_samples_per_class
        masks = [[target == given_target for target in self.og_targets]
                 for given_target in self.target_classes]
        indices_selections = [[i for i, m in enumerate(mask) if m] for mask in masks]
        self.class_indices = {given_target: indices_selections[i] for i, given_target in enumerate(self.target_classes)}

    def __iter__(self):
        used_indices = {given_target: [] for given_target in self.target_classes}
        i = 0
        while True:
            if i == self.__len__():
                break
            try:
                batch = []
                for c, n in zip(self.target_classes, self.num_samples_per_class):
                    all_data_indices = set(self.class_indices[c]) - set(used_indices[c])
                    n = min(n, len(all_data_indices))
                    data_indices = random.sample(all_data_indices, n)
                    batch.extend(data_indices)
                    used_indices[c].extend(data_indices)
                random.shuffle(batch)
                i += 1
                yield batch
            except StopIteration:
                break

    def __len__(self):
        num_batches = sum([len(self.class_indices[c]) for c in self.target_classes]) // sum(self.num_samples_per_class)
        return num_batches
