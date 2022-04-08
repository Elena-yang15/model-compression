from importlib import import_module

import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms

def get_loader(args, kwargs):
    norm_mean=[x / 255 for x in [109.9, 109.7, 113.8]]
    norm_std=[x / 255 for x in [50.1, 50.6, 50.8]]
    loader_train = None

    if not args.test_only:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)])

        loader_train = DataLoader(
            datasets.SVHN(
                root=args.dir_data,
                download=True,
                split='train',
                transform=transform_train),
            batch_size=args.batch_size * args.n_GPUs, shuffle=True, **kwargs
        )

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)])

    loader_test = DataLoader(
        datasets.SVHN(
            root=args.dir_data,
            split='test',
            download=True,
            transform=transform_test),
        batch_size=256 * args.n_GPUs, shuffle=False, **kwargs
    )

    return loader_train, loader_test
