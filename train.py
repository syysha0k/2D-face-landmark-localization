'''Train Face Landmark Pytorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from models import *
from util.utils import progress_bar
from util.data_load import FacialKeypointsDataset, Rescale, Normalize, ToTensor
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch Face Landmark Training')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--train_list', type=str, help='train file')
parser.add_argument('--val_list', type=str, help='test file')
parser.add_argument('--train_dataset', type=str, help='train image dataset')
parser.add_argument('--val_dataset', type=str, help='valid image dataset')
parser.add_argument('--batch-size', default=256, type=int)
parser.add_argument('--val-batch-size', default=128, type=int)
parser.add_argument('--save_path', type=str, help='The pat to save model')
args = parser.parse_args()

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
data_transform = transforms.Compose([
    Rescale((224, 224)),
    Normalize(),
    ToTensor()
])

train_dataset = FacialKeypointsDataset(
    csv_file=args.train_list,
    root_dir=args.train_dataset, 
    transform=data_transform
)
val_dataset = FacialKeypointsDataset(
    csv_file=args.val_list,
    root_dir=args.val_dataset,
    transform=data_transform
)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4,
    shuffle=True, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=4,
    shuffle=False, pin_memory=True)

print(' ==> The number of trainset is {}'.format(len(train_dataset)))
print(' ==> The number of validset is {}'.format(len(val_dataset)))

# Model
print('==> Building model..')
net = mobilenet_1()
#net = torchvision.models.vgg16(pretrained=True)
#net.classifier._modules['6'] = nn.Linear(4096, 136)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

#criterion = nn.MSELoss()
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(params = net.parameters(), lr = args.lr)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        images = data['image']
        key_pts = data['keypoints']
        key_pts = key_pts.view(key_pts.size(0), -1)
        key_pts = key_pts.type(torch.FloatTensor)
        images = images.type(torch.FloatTensor)
        inputs = images.to(device)
        targets = key_pts.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        #\outputs = outputs.view(outputs.size()[0], 68, -1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if batch_idx % 10 == 9:
            print('EPOCH: {}, BATCH: {}, AVG LOSS: {}'.format(epoch + 1, batch_idx + 1, running_loss))
            running_loss = 0.0
#        train_loss += loss.item()
#        _, predicted = outputs.max(1)
#        total += targets.size(0)
#        correct += predicted.eq(targets).sum().item()

#        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

pred = open('pred.txt', 'w')
lab = open('lab.txt', 'w')
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0


    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            images = data['image']
            key_pts = data['keypoints']
            key_pts = key_pts.view(key_pts.size(0), -1)
            key_pts = key_pts.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)
            inputs = images.to(device)
            targets = key_pts.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()

            if epoch == 99:
                print('*************')
                for index in range(len(data['name'])):
                    name = data['name'][index]
                    rec_pred = outputs[index].cpu().numpy() * 224
                    rec_pred.astype(np.int32).tolist()
                    rec_pred = [str(ele) for ele in rec_pred]
                    keypoint_pred = ' '.join(rec_pred)
                    keypoint_pred = name + ',' + keypoint_pred
                    pred.write(keypoint_pred)
                    pred.write('\n')
                    print(keypoint_pred)

                    rec_lab = key_pts.numpy() * 224
                    rec_lab = rec_lab[index].astype(np.int32).tolist()
                    rec_lab = [str(ele) for ele in rec_lab]
                    keypoint_lab = ' '.join(rec_lab)
                    keypoint_lab = name + ',' + keypoint_lab
                    lab.write(keypoint_lab)
                    lab.write('\n')
                    print(keypoint_lab)
                pred.close()
                lab.close()
        print('==> TEST EPOCH: {}, LOSS: {}'.format(epoch, test_loss))
            #print('AVG LOSS: {}'.format(test_loss))
#            _, predicted = outputs.max(1)
#            total += targets.size(0)
#            correct += predicted.eq(targets).sum().item()
#
#            progress_bar(batch_idx, len(testloader), 'Loss: %.3f' %(test_loss/(batch_idx+1)))

    # Save checkpoint.
    # acc = 100.*correct/total
    if test_loss != 0:
        print('Saving..')
        state = {
            'epoch': epoch,
            'net': net.state_dict(),
             #'optimizer': optimizer.state_dict()
        }
        model_save_path = args.save_path
        if not os.path.isdir(model_save_path):
            os.mkdir(model_save_path)
        torch.save(state, './' + model_save_path + '/ckpt_ep{}_loss{}.pth'.format(epoch, test_loss))
        



#def net_sample_output():
#     for i, sample in enumerate(train_loader):
#         images = sample['image']
#         key_pts = sample['keypoints']
#         images = images.type(torch.FloatTensor)
#         output_pts = net(images)
#         output_pts = output_pts.view(output_pts.size()[0], 68, -1)
#         if i == 0:
#             return images, output_pts, key_pts
#     
for epoch in range(start_epoch, start_epoch+100):
    train(epoch)
    test(epoch)





