from pathlib import Path
import urllib.request
from zipfile import ZipFile

from kaggle.api.kaggle_api_extended import KaggleApi
import progressbar
import torch
from torchvision import transforms


# calculate mean and standard deviation of features
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    """ normalize features """
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


def calc_feat_flatten_mean_std(feat):
    """ takes 3D features (C, H, W), return mean and std of array within channels """
    assert (feat.size()[0] == 3)  # make usre there are 3 dimentions
    assert (isinstance(feat, torch.FloatTensor))  # make usre its a tensor
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def adjust_learning_rate(optimizer, iteration_count, lr_rate, lr_decay):
    """Imitating the original implementation"""
    lr = lr_rate / (1.0 + lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_transform():
    """ resize image and convert to tensor """
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


# progressbar for download
pbar = None


def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None


def download_data():
    """ Download the datasets """
    # download content images
    Path("data/content").mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve("http://images.cocodataset.org/zips/train2017.zip",
                               "data/train2017.zip", show_progress)
    content = ZipFile("data/train2017.zip")
    content.extractall(path="data")

    # download style images from kaggle
    Path("data/style").mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.competition_download_file('painter-by-numbers', 'train.zip', 'data')
    style = ZipFile("data/train.zip")
    style.extractall(path="data")
