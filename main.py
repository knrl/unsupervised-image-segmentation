import argparse
import cv2
import os
import numpy as np
from skimage import segmentation
from skimage import filters
import torch.nn.init

import matplotlib.pyplot as plt
from skimage.filters import sobel
from scipy import ndimage as ndi
from skimage import exposure

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# CNN model
class MyNet(nn.Module):
    def __init__(self,input_dim):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, args.nChannel, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(args.nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(args.nConv-1):
            self.conv2.append( nn.Conv2d(args.nChannel, args.nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(args.nChannel) )
        self.conv3 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(args.nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(args.nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.up(x)
        return x

class UNet(nn.Module):
    def __init__(self, input_dim):
        super(UNet, self).__init__()

        n1 = args.nChannel
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(input_dim, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv6 = nn.Conv2d(filters[0], args.nChannel, kernel_size=1, stride=1, padding=0)

        # self.Conv7 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0)
        self.Bn3 = nn.BatchNorm2d(args.nChannel)

    def forward(self, x):
        e1 = self.Conv1(x)
    
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        # e4 = self.Maxpool3(e3)
        # e4 = self.Conv4(e4)

        # e5 = self.Maxpool4(e4)
        # e5 = self.Conv5(e5)
        # d5 = self.Up5(e5)
        # d5 = torch.cat((e4, d5), dim=1)
        # d5 = self.Up_conv5(d5)

        # d4 = self.Up4(e4)
        # d4 = torch.cat((e3, d4), dim=1)
        # d4 = self.Up_conv4(d4)

        d3 = self.Up3(e3)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv6(d2)
        out = self.Bn3(out)

        return out

def load_image(input, image_size):
    im = cv2.imread(input)
    im = cv2.resize(im, image_size)
    if (args.equalize):
        im[0] = exposure.equalize_hist(im[0])
        im[1] = exposure.equalize_hist(im[1])
        im[2] = exposure.equalize_hist(im[2])
    data = torch.from_numpy(np.array([im.transpose( (2, 0, 1) ).astype('float32')/255.]))
    if use_cuda:
        data = data.cuda()
    return im, data

def get_model(input_size, model_name='mynet'):
    if (model_name == 'mynet'):
        model = MyNet(input_size)
    elif (model_name == 'unet'):
        model = UNet(input_size)
    
    if use_cuda:
        model.cuda()
    return model

def train_model(data, model, image_shape, lr, maxIter, l_inds):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    label_colours = np.random.randint(255,size=(100,3))

    for batch_idx in range(maxIter):
        # forwarding
        optimizer.zero_grad()
        output = model( data )[ 0 ]
        output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )
        _, target = torch.max( output, 1 )
        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))
        if args.visualize:
            im_target_rgb = np.array([label_colours[ c % 100 ] for c in im_target])
            im_target_rgb = im_target_rgb.reshape(image_shape).astype( np.uint8 )

            # cv2.imshow("output", im_target_rgb)
            # cv2.waitKey(10)

        # superpixel refinement
        for i in range(len(l_inds)):
            labels_per_sp = im_target[ l_inds[ i ] ]
            u_labels_per_sp = np.unique( labels_per_sp )
            hist = np.zeros( len(u_labels_per_sp) )
            for j in range(len(hist)):
                hist[ j ] = len( np.where( labels_per_sp == u_labels_per_sp[ j ] )[ 0 ] )
            im_target[ l_inds[ i ] ] = u_labels_per_sp[ np.argmax( hist ) ]
        
        target = torch.from_numpy(im_target.reshape(-1))
        if use_cuda:
            target = target.cuda()

        target = Variable( target )
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        #############################
        if (batch_idx+1 % 10 == 0):
            optimizer.param_groups[0]['lr'] = 0.01
        if (batch_idx+1 % 30 == 0):
            optimizer.param_groups[0]['lr'] = 0.001
        if (batch_idx+1 % 70 == 0):
            optimizer.param_groups[0]['lr'] = 0.0001
        #############################

        # print (batch_idx, '/', maxIter, ':', nLabels, loss.data[0])
        print (batch_idx, '/', maxIter, ':', nLabels, loss.item())

    return label_colours

def test_model(input, model, label_colours, image_size):
    if not os.path.isdir(input):
        path = input
        im, data = load_image(path, image_size)
        if use_cuda:
            data = data.cuda()

        output = model( data )[ 0 ]
        output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )
        _, target = torch.max( output, 1 )
        im_target = target.data.cpu().numpy()
        im_target_rgb = np.array([label_colours[ c % 100 ] for c in im_target])
        im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )
        cv2.imwrite(args.resultPath + path.split('/')[-1], im_target_rgb)

        print(args.resultPath + path.split('/')[-1])
    else:
        for image in os.listdir(input):
            path = os.path.join(input, image)
            im, data = load_image(path, image_size)
            if use_cuda:
                data = data.cuda()

            output = model( data )[ 0 ]
            output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )
            _, target = torch.max( output, 1 )
            im_target = target.data.cpu().numpy()
            im_target_rgb = np.array([label_colours[ c % 100 ] for c in im_target])
            im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )

            # cv2.imshow("output", im_target_rgb)
            # cv2.waitKey(2000)
            cv2.imwrite(args.resultPath + image + ".png", im_target_rgb)

def main(args, model_name='mynet'):
    # parameters
    image_size = (256, 256)
    N = 3

    print("Is directory: ", args.isDirectory)
    print("Loading image...")

    # init once
    model = get_model(N, model_name)
    model.train()

    # load image
    if (args.isDirectory):
        for image in os.listdir(args.trainInput):
            path = os.path.join(args.trainInput, image)
            im, data = load_image(path, image_size)
            data     = Variable(data)

            # slic
            labels   = segmentation.slic(im, compactness=args.compactness, n_segments=args.num_superpixels)
            labels   = labels.reshape(im.shape[0]*im.shape[1])
            u_labels = np.unique(labels)
            l_inds   = []
            for i in range(len(u_labels)):
                l_inds.append(np.where( labels == u_labels[i] )[0])

            # train
            label_colours = train_model(data, model, im.shape, args.lr, args.maxIter, l_inds)
    elif (args.combine == True):
        im, data = load_image(args.trainInput, image_size)
        data     = Variable(data)

        # slic
        labels   = segmentation.slic(im, compactness=args.compactness, n_segments=args.num_superpixels)
        labels   = labels.reshape(im.shape[0]*im.shape[1])
        u_labels = np.unique(labels)
        l_inds   = []
        for i in range(len(u_labels)):
            l_inds.append(np.where( labels == u_labels[i] )[0])

        # train
        label_colours = train_model(data, model, im.shape, args.lr, args.maxIter , l_inds)

        model_unet = get_model(N, 'unet')
        model_unet.train()
        # copy model's conv weights to model unet
        list(model_unet.state_dict().items())[0][1].data[:] = list(model.state_dict().items())[0][1].data[:]

        # train
        label_colours = train_model(data, model_unet, im.shape, args.lr, args.maxIter, l_inds)
    else:
        im, data = load_image(args.trainInput, image_size)
        data     = Variable(data)

        # slic
        labels   = segmentation.slic(im, compactness=args.compactness, n_segments=args.num_superpixels)
        labels   = labels.reshape(im.shape[0]*im.shape[1])
        u_labels = np.unique(labels)
        l_inds   = []
        for i in range(len(u_labels)):
            l_inds.append(np.where( labels == u_labels[i] )[0])

        # train
        label_colours = train_model(data, model, im.shape, args.lr, args.maxIter, l_inds)

    # test
    test_model(args.testInput, model, label_colours, image_size)

if (__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
    parser.add_argument('--nChannel',        metavar='N',       default=100,   type=int,   help='number of channels')
    parser.add_argument('--maxIter',         metavar='T',       default=10,    type=int,   help='number of maximum iterations')
    parser.add_argument('--minLabels',       metavar='minL',    default=3,     type=int,   help='minimum number of labels')
    parser.add_argument('--lr',              metavar='LR',      default=0.1,   type=float, help='learning rate')
    parser.add_argument('--nConv',           metavar='M',       default=2,     type=int,   help='number of convolutional layers')
    parser.add_argument('--nEpochs',         metavar='E',       default=2,     type=int,   help='number of epochs')
    parser.add_argument('--num_superpixels', metavar='K',       default=10000, type=int,   help='number of superpixels')
    parser.add_argument('--compactness',     metavar='C',       default=100,   type=float, help='compactness of superpixels')
    parser.add_argument('--visualize',       metavar='1 or 0',  default=1,     type=int,   help='visualization flag')
    parser.add_argument('--isDirectory',     metavar='D',       default=False, type=bool,  help='directory')
    parser.add_argument('--trainInput',      metavar='FILENAME',help='input image file name', required=True)
    parser.add_argument('--testInput',       metavar='FILENAME',help='input image file name', required=True)
    parser.add_argument('--resultPath',      metavar='FILENAME',help='results file name',     required=True)
    parser.add_argument('--model',           metavar='NM',      help='model name', required=False)
    parser.add_argument('--combine',         metavar='CM',      default=False, type=bool,  help='combine model', required=False)
    parser.add_argument('--equalize',        metavar='EQ',      default=True,  type=bool,  help='equalization on/off', required=False)
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    print("CUDA is available: ", use_cuda)

    main(args, args.model)

"""
Example usage:
python3 src/main.py --trainInput '/<path>/src/demo_test_data/2092.jpg' --testInput '/<path>/src/demo_test_data/2092.jpg' --resultPath /<path>/results/demo_results/ --combine True --model mynet --nChannel 50 --lr 0.1 --maxIter 80
"""
