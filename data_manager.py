import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


class Sharding(Dataset):
    """A sharding of a dataset"""

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DatasetCreator(object):
    """Dataset creator loads and divides dataset"""

    def __init__(self, num_clients, dataset_name, partition_method, seed=5):
        random.seed(seed)  # fix seed for reproduction
        self.n_clients = num_clients
        self.dt_name = dataset_name
        self.pt_method = partition_method
        self.shardings = []
        self.training_data, self.test_data = self.__get_dataset()
        self.__divide_dataset()

    def get_loader(self, client_id=0, batch_sz=32, is_test=False, pin_memory=True, time_out=0, num_workers=0):
        """Get DataLoader object."""
        if not is_test:
            drop_last = shuffle = True
            sharding = Sharding(self.training_data, self.shardings[client_id])
            loader = DataLoader(sharding, batch_sz, shuffle=shuffle, pin_memory=pin_memory, timeout=time_out,
                                num_workers=num_workers, drop_last=drop_last)
        else:
            index = [_ for _ in range(len(self.test_data))]
            non_sharding = Sharding(self.test_data, index)
            loader = DataLoader(non_sharding, batch_sz, shuffle=False, pin_memory=pin_memory, timeout=time_out,
                                num_workers=num_workers, drop_last=False)
        return loader

    def get_training_data_len(self, client_id=0):
        return len(self.shardings[client_id])

    def get_test_data_len(self):
        return len(self.test_data)

    def __divide_dataset(self):
        if self.pt_method == 'uniform':
            data_len = len(self.training_data)
            train_indexes = list(range(data_len))

            random.shuffle(train_indexes)

            part_len = int(1. / self.n_clients * len(train_indexes))

            for _ in range(self.n_clients):
                self.shardings.append(train_indexes[0:part_len])
                train_indexes = train_indexes[part_len:]

    def __get_dataset(self):
        training_dataset = test_dataset = None
        training_transform, test_transform = self.__get_transform()

        if self.dt_name == 'CIFAR10':
            training_dataset = datasets.CIFAR10(
                root='data/CIFAR10',
                train=True,
                download=True,
                transform=training_transform
            )
            test_dataset = datasets.CIFAR10(
                root='data/CIFAR10',
                train=False,
                download=True,
                transform=test_transform
            )

        return training_dataset, test_dataset

    def __get_transform(self):
        if self.dt_name == 'CIFAR10':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        return transform_train, transform_test
