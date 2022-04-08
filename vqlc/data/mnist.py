from importlib import import_module

import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision

def get_loader(args, kwargs):
    loader_train = None

    if not args.test_only:
        transform_train = transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
        (0.1307,), (0.3081,))
            ])

        loader_train = DataLoader(
            datasets.MNIST(
                root=args.dir_data,
                download=True,
                train=True,
                transform=transform_train),
            batch_size=args.batch_size * args.n_GPUs, shuffle=True, **kwargs
        )

    transform_test = transforms.Compose([
        torchvision.transforms.Resize(32),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
        (0.1307,), (0.3081,))
        ])

    loader_test = DataLoader(
        datasets.MNIST(
            root=args.dir_data,
            train=False,
            download=True,
            transform=transform_test),
        batch_size=256 * args.n_GPUs, shuffle=False, **kwargs
    )

    return loader_train, loader_test
