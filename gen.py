import math
import argparse
import torch
import torchvision
from torch.autograd import Variable

from .model import Discriminator, Generator

parser = argparse.ArgumentParser()
parser.add_argument('--gen-num', type=int, default=64, metavar='N',
                    help='the number of generated picture(s) you will get, must be a perfect square (default: 64)')
parser.add_argument('--gen-count', type=int, default=512, metavar='N',
                    help='the number of generated picture(s) (default: 512)')
parser.add_argument('--dim', type=int, default=100, metavar='D',
                    help='the dimension of random noise in your model (default: 100)')
parser.add_argument('--channel', type=int, default=8, metavar='N',
                    help='intermediate channel of model (default: 8)')
parser.add_argument('--g-model-path', type=str, default='checkpoint/gen.pth',
                    help='the path of generator\'s state dict (default: checkpoint/gen.pth)')
parser.add_argument('--d-model-path', type=str, default='checkpoint/dis.pth',
                    help='the path of discriminator\'s state dict (default: checkpoint/dis.pth)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='A',
                    help='negative slope of leaky relu (default: 0.2)')
parser.add_argument('--images-path', type=str, default='pic.png',
                    help='the path of generated images (default: pic.png)')
args = parser.parse_args()

D = Discriminator(args.channel, args.alpha)
G = Generator(args.dim, args.channel)

D.eval()
G.eval()

D.load_state_dict(torch.load(args.d_model_path))
G.load_state_dict(torch.load(args.g_model_path))

assert args.gen_count >= args.gen_num
assert int(math.sqrt(args.gen_num)) ** 2 == args.gen_num

noises = torch.randn(args.gen_count, args.dim, 1, 1)
noises = Variable(noises, volatile=True)

generated_images = G(noises)
scores = D(generated_images).data

indexes = scores.topk(args.gen_num)[1]
results = []
for i in indexes:
    results.append(generated_images.data[i])

torchvision.utils.save_image(torch.stack(results), args.image_path,normalize=True, range=(-1, 1))
