import math
import argparse
import visdom
import torch
import torchvision
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from model import Discriminator, Generator
class Gan():
    def __init__(self):
        self.train_parser = argparse.ArgumentParser()
        self.train_parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='batch size for training (default: 64)')
        self.train_parser.add_argument('--dataset', default='./data/MNIST', type=str)
        self.train_parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 300)')
        self.train_parser.add_argument('--g-lr', type=float, default=2e-4, metavar='LR-G',
                    help='learning rate of generator (default: 2e-4)')
        self.train_parser.add_argument('--d-lr', type=float, default=2e-4, metavar='LR-D',
                    help='learning rate of discriminator (default: 2e-4)')
        self.train_parser.add_argument('--beta1', type=float, default=0.5, metavar='B1',
                    help='beta1 of Adam optimizer (default: 0.5)')
        self.train_parser.add_argument('--beta2', type=float, default=0.999, metavar='B2',
                    help='beta2 of Adam optimizer (default: 0.999)')
        self.train_parser.add_argument('--dim', type=int, default=100, metavar='D',
                    help='the dimension of random noise (default: 100)')
        self.train_parser.add_argument('--plot-per-epoch', type=int, default=40, metavar='PPE',
                    help='interval between plotting generated pictures (default: 40)')
        self.train_parser.add_argument('--save-per-epoch', type=int, default=40, metavar='SPE',
                    help='interval between saving model (default: 40)')
        self.train_parser.add_argument('--channel', type=int, default=64, metavar='N',
                    help='intermediate channel of model (default: 64)')
        self.train_parser.add_argument('--train-d-every', type=int, default=1, metavar='N',
                    help='the interval between training deterministic model (default: 1)')
        self.train_parser.add_argument('--train-g-every', type=int, default=2, metavar='N',
                    help='the interval between training generated model (default: 2)')
        self.train_parser.add_argument('--alpha', type=float, default=0.2, metavar='A',
                    help='negative slope of leaky relu (default: 0.2)')
        self.train_parser.add_argument('--env', type=str, default='face', metavar='N',
                    help='the name of visdom\'s environment (default : face)')

        self.args = self.train_parser.parse_args()
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")



         # transform
        self.train_transform = transforms.Compose(
            [transforms.ToTensor()])
        self.test_transform = transforms.Compose(
            [transforms.ToTensor()])

    def train(self,args):
            model_path='checkpoint/'
            data_path='./data/MNIST'
            vis=visdom.Visdom(env=args.env)
            dataset = args.dataset
            trainset = torchvision.datasets.MNIST(root=dataset, train=True,
                                              download=True, transform=self.train_transform)
            dataloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset),
                                             shuffle=True)
            G=Generator(args.dim,args.channel)
            D=Discriminator(args.channel,args.alpha)

            G_optimizer =optim.Adam(G.parameters(),lr=args.g_lr,betas=(args.beta1,args.beta2))
            D_optimizer=optim.Adam(D.parameters(),lr=args.d_lr,betas=(args.beta1,args.beta2))
            criterion=torch.nn.BCELoss()

            true_labels=Variable(torch.ones(args.batch_size))
            fake_labels=Variable(torch.zeros(args.batch_size))

            gen_vector=Variable(torch.randn(args.batch_size,args.dim,1,1))

            train_d_times=0
            train_g_times=0

            for epoch in range(args.epochs):
                print('epoch {0}'.format(epoch+1))

                for batch_ix,(img,_) in enumerate(dataloader):
                    print("a new batch!")
                    images=Variable(img)
                    if batch_ix% args.train_d_every ==0:
                        D_optimizer.zero_grad()

                        output=D(images)
                        real_loss=criterion(output,true_labels)
                        real_loss.backward()

                        noise=Variable(torch.randn(args.batch_size,args.dim,1,1))
                        fake_images=G(noise).detach()
                        output=D(fake_images)
                        fake_loss=criterion(output,fake_labels)
                        fake_loss.backward()

                        D_optimizer.step()

                        loss=real_loss+fake_loss

                        #visualize error
                        vis.line(win='D_error',
                                 X=torch.Tensor([train_d_times]),
                                 Y=loss.data,
                                 update=None if train_d_times==0 else 'append'
                                 )
                        train_d_times+=1

                    if batch_ix % args.train_g_every==0:
                        G_optimizer.zero_grad()

                        noise=Variable(torch.randn(args.batch_size,args.dim,1,1))
                        fake_images=G(noise)
                        output=D(fake_images)
                        fake_loss=criterion(output,true_labels)
                        fake_loss.backward()

                        G_optimizer.step()
                        vis.line(win='G_error',
                                 X=torch.Tensor(train_g_times),
                                 Y=fake_loss.data,
                                 update=None if train_g_times==0 else 'append'
                                 )
                        train_g_times+=1

                    if (epoch+1) %args.save_per_epoch==0:
                        torch.save(D.state_dict(),model_path+'D_epoch_{0}.pth'.format(epoch))
                        torch.save(G.state_dict(),model_path+'G_epoch_{0}.pth'.format(epoch))

                    if (epoch+1) % args.plot_per_epoch ==0:
                        images=G(gen_vector)
                        n_row=int(math.sqrt(args.batch_size))
                        torchvision.utils.save_image(images.data[: n_row * n_row], model_path + '%d_.png'.format(epoch), normalize=True, range=(-1, 1), nrow=n_row)
            print("train finished")           





