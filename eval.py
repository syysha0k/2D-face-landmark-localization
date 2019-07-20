'''Eval Face Landmark Pytorch.'''
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from models import *
from util.utils import progress_bar
from util.data_load import FacialKeypointsDataset, Rescale, Normalize, ToTensor
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch Face Landmark Training')
parser.add_argument('--test_list', type=str, help='test file')
parser.add_argument('--test_dataset', type=str, help='test image dataset')
parser.add_argument('--test-batch-size', default=32, type=int)
parser.add_argument('--record', type=str, help='Where to save result record')
parser.add_argument('--pth', type=str, help='The test weigth')
args = parser.parse_args()

print('** Eval Model')
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'


print(' ==> Preparing data ...')
data_transform = transforms.Compose([
    Rescale((224, 224)),
    Normalize(),
    ToTensor(),
])
val_dataset = FacialKeypointsDataset(
    csv_file=args.test_list,
    root_dir=args.test_dataset,
    transform=data_transform
)
val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, num_workers=4,
    shuffle=False, pin_memory=True)

print(' ==> The number of validset is {}'.format(len(val_dataset)))
print('==> Building model..')
net = mobilenet_1()
#net = torchvision.models.vgg16()
#net.classifier._modules['6'] = nn.Linear(4096, 136)
checkpoint = torch.load(args.pth)
net.load_state_dict(checkpoint['net'])
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
f = open(args.record, 'w')
def eval():
    net.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            print(' ** eval: {}'.format(batch_idx))
            images = data['image']
            names = data['name']
            length = len(names)
            images = images.type(torch.FloatTensor)
            inputs = images.to(device)
            outputs = net(inputs)
            #print((outputs[0].cpu().numpy() * 50 + 100).astype(np.int32))
            #exit(0)
            for index in range(length):
                f.write(names[index])
                f.write(',')
                rec = outputs[index].cpu().numpy() * 224.0 
                rec.astype(np.int32).tolist()
                rec = [str(ele) for ele in rec.astype(np.int32)]
                keypoint = ' '.join(rec)
                f.write(keypoint)
                f.write('\n')
    f.close()

if __name__ == '__main__':
    eval()





