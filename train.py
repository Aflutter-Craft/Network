import argparse
import glob
import os

from PIL import Image
from PIL import ImageFile
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import tqdm

from net.net import *
from utils.flat_dataset import FlatFolderDataset
from utils.sampler import InfiniteSamplerWrapper
from utils.utils import *

cudnn.benchmark = True  # benchmark mode is faster
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device('cuda')  # use GPU for training

# argument parser (makes life easier)
parser = argparse.ArgumentParser(
    description="training arguments", prog="train")

# Basic options
parser.add_argument('--content_dir', type=str, default='data/content',
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, default='data/style',
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='pretrained/vgg.pth')

# training options
parser.add_argument('--save_dir', default='models',
                    help='Directory to save the model')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--style_weight', type=float, default=3.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--n_threads', type=int, default=2)
parser.add_argument('--save_model_interval', type=int, default=1000)
parser.add_argument('--start_iter', type=float, default=0)

# make sure the datasets have been downloaded
if (not os.path.exists("data/content") or not os.path.exists("data/style")):
    print("Downloading Data...")
    download_data()

# parse args
args = parser.parse_args('')


# init vgg and decoder from defined arch
decoder = decoder_arch
vgg = vgg_arch


vgg.load_state_dict(torch.load(args.vgg))  # load pretrained vgg network
vgg = nn.Sequential(*list(vgg.children())[:44])  # extract layers until relu5_1

# create and config main net
network = Net(vgg, decoder, args.start_iter)
network.train()  # set network to training mode
network.to(device)  # move network to GPU

# get image transformations
content_tf = train_transform()
style_tf = train_transform()

# load style and content images (__iter__ is used to iterate them)
content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

# make training datasets
content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

# set up optimizer (optimizing both networks losses)
optimizer = optim.Adam([
    {'params': network.decoder.parameters()},
    {'params': network.transform.parameters()}], lr=args.lr)

# load optimizer state if not training from the begining
if(args.start_iter > 0):
    optimizer.load_state_dict(torch.load(
        'optimizer_iter_' + str(args.start_iter) + '.pth'))

# trainnig loop
for i in tqdm(range(args.start_iter, args.max_iter)):
    # decay learning rate
    adjust_learning_rate(optimizer, iteration_count=i, lr_rate=args.lr,
                         lr_decay=args.lr_decay)
    # get input images
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)
    # get losses for both images
    loss_c, loss_s, l_identity1, l_identity2 = network(
        content_images, style_images)
    loss_c = args.content_weight * loss_c  # lambda_c * loss
    loss_s = args.style_weight * loss_s  # lambda_s *loss
    # total loss, (50 and 1 are numbers proven to improve the results)
    loss = loss_c + loss_s + (l_identity1 * 50) + (l_identity2 * 1)

    optimizer.zero_grad()  # reset gradients
    loss.backward()  # calculate gradients
    optimizer.step()  # next epoch

    # save models and optimizer states based on specificed interval (cuda makes this complicated)
    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:

        # create dir
        if not os.path.exists(args.save_dir):
            os.system("mkdir " + args.save_dir)

        # delete all previously saved models except last one
        for f in glob.glob(args.save_dir + '/*.pth'):
            if (f'{(i + 1) - 1000}') not in f:
                os.remove(f)

        state_dict = decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/decoder_iter_{:d}.pth'.format(args.save_dir,
                                                       i + 1))
        state_dict = network.transform.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/transformer_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
        state_dict = optimizer.state_dict()
        torch.save(state_dict,
                   '{:s}/optimizer_iter_{:d}.pth'.format(args.save_dir,
                                                         i + 1))
