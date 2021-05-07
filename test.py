import argparse
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision.utils import save_image
from utils.utils import train_transform
from net.net import *

device = torch.device('cuda' if torch.cuda.is_available()  # type: ignore
                      else 'cpu')

parser = argparse.ArgumentParser()

# Basic options
parser.add_argument('--content', type=str, default='',
                    help='File path to the content image')
parser.add_argument('--style', type=str, default='',
                    help='File path to the style image')
parser.add_argument('--steps', type=str, default=1)
parser.add_argument('--vgg', type=str, default='models/vgg.pth')
parser.add_argument('--decoder', type=str, default='models/decoder.pth')
parser.add_argument('--transformer', type=str,
                    default='models/transformer.pth')

# Additional options
parser.add_argument('--save_ext', default='.png',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')

# Advanced options
parser.add_argument('--alpha', default=0.8,
                    help='Style content tradeoff, more means more style and less means preserve more content, max = 1.0')

args = parser.parse_args('')


# extract style and content features
def feat_extractor(vgg, content, style):
    # extract used layers from vgg network
    enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
    enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
    enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
    enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
    enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1

    # move everything to GPU
    enc_1.to(device)
    enc_2.to(device)
    enc_3.to(device)
    enc_4.to(device)
    enc_5.to(device)

    # extract content features
    Content4_1 = enc_4(enc_3(enc_2(enc_1(content))))
    Content5_1 = enc_5(Content4_1)
    # extract style features
    Style4_1 = enc_4(enc_3(enc_2(enc_1(style))))
    Style5_1 = enc_5(Style4_1)

    return Content4_1, Content5_1, Style4_1, Style5_1


# perform style transfer
def style_transfer(vgg, decoder, samodule, content, style, alpha=1):
    assert (0.0 <= alpha <= 1.0)  # make sure alpha value is valid

    # move samodule and decoder to gpu
    samodule.to(device)
    decoder.to(device)

    # get features for both style and content
    Content4_1, Content5_1, Style4_1, Style5_1 = feat_extractor(
        vgg, content, style)

    Fccc = samodule(Content4_1, Content4_1, Content5_1, Content5_1)

    # get final image features
    feat = samodule(Content4_1, Style4_1, Content5_1, Style5_1)
    # change final image according to alpha values
    feat = feat * alpha + Fccc * (1 - alpha)
    # return decoded final image
    return decoder(feat)


if __name__ == "__main__":

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    decoder = decoder_arch
    samodule = StyleAttentionalNet(in_dim=512)
    vgg = vgg_arch

    # set to evalaution mode (faster)
    decoder.eval()
    samodule.eval()
    vgg.eval()

    # load saved model states
    decoder.load_state_dict(torch.load(args.decoder))
    samodule.load_state_dict(torch.load(args.samodule))
    vgg.load_state_dict(torch.load(args.vgg))

    # load images from widgets and convert to RBG (in case its a png)
    trans = train_transform()
    content = trans(Image.open(args.content).convert('RGB'))
    style = trans(Image.open(args.style).convert('RGB'))

    # transfer images to GPU and add batch dimention
    style = style.to(device).unsqueeze(0)
    content = content.to(device).unsqueeze(0)

    # stylization loop (dont calculate gradients because this is inference and we have no use for them)
    with torch.no_grad():
        # get stylized image
        output = style_transfer(vgg, decoder, samodule,
                                content, style, args.alpha)

        # extract name from style and content names
        style_name = args.style.split('.')[0]
        content_name = args.content.split('.')[0]

        # save final image
        output_name = f'{args.output}/{content_name}_stylized_{style_name}_{args.alpha}{args.save_ext}'
        save_image(output, output_name)
