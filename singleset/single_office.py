"""
SingleSet for office caltech 10
"""
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
import torchvision
import torchvision.transforms as transforms
import time
from nets.models import AlexNet
import argparse
from utils.data_utils import OfficeDataset
import numpy as np
import random



def train(model, data_loader, optimizer, loss_fun, device):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
        
        loss.backward()
        optimizer.step()

    return loss_all / len(data_loader), correct/total

def test(model, data_loader, loss_fun, device):
    model.eval()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct/total


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 4
    np.random.seed(seed)
    torch.manual_seed(seed)     
    torch.cuda.manual_seed_all(seed) 
    random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help='whether to log')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--batch', type = int, default= 32, help ='batch size')
    parser.add_argument('--save_path', type = str, default='../checkpoint/office', help='path to save the checkpoint')
    parser.add_argument('--data', type = str, default= 'dslr', help='[amazon | caltech | dslr | webcam]')
    args = parser.parse_args()

    exp_folder = 'single_office'

    args.save_path = os.path.join(args.save_path, exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, '{}'.format(args.data))
    
    log = args.log
    if log:
        log_path = os.path.join('../logs/office/', exp_folder)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logfile = open(os.path.join(log_path,'{}.log'.format(args.data)), 'a')
        logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        logfile.write('===Setting===\n')
        logfile.write('    lr: {}\n'.format(args.lr))
        logfile.write('    batch: {}\n'.format(args.batch))
        logfile.write('    dataset: {}\n'.format(args.data))
        logfile.write('    epochs: {}\n'.format(args.epochs))

    # setup data
    transform_office = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.ToTensor(),
    ])
    
    data_base_path = '../data'
    min_data_len = 5e8
    for site in ['amazon', 'caltech', 'dslr', 'webcam']:
        trainset = OfficeDataset(data_base_path, site, transform=transform_office)
        if min_data_len > len(trainset):
            min_data_len = len(trainset)
    
    val_len = int(min_data_len * 0.4)
    min_data_len = int(min_data_len * 0.5)
    
    trainset = OfficeDataset(data_base_path, args.data, transform=transform_office)
    testset = OfficeDataset(data_base_path, args.data , transform=transform_test, train=False)
    # subset sample
    valset   = torch.utils.data.Subset(trainset, list(range(len(trainset)))[-val_len:])
    trainset = torch.utils.data.Subset(trainset, list(range(min_data_len)))

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
    
    # setup model
    model = AlexNet().to(device)
    loss_fun = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=args.lr)

    best_acc = 0
    best_epoch = 0
    start_epoch = 0
    N_EPOCHS = args.epochs

    for epoch in range(start_epoch,start_epoch+N_EPOCHS):
        train_loss, train_acc = train(model, train_loader, optimizer, loss_fun, device)
        print('Epoch: [{}/{}] | Train Loss: {:.4f} | Train Acc: {:.4f}'.format(epoch, N_EPOCHS ,train_loss, train_acc))
        if log:
            logfile.write('Epoch: [{}/{}] | Train Loss: {:.4f} | Train Acc: {:.4f}\n'.format(epoch, N_EPOCHS ,train_loss, train_acc))

        val_loss, val_acc = test(model, val_loader, loss_fun, device)
        print('Val site-{} | Val Loss: {:.4f} | Val Acc: {:.4f}'.format(args.data, val_loss, val_acc))
        if log:
            logfile.write('Val site-{} | Val Loss: {:.4f} | Val Acc: {:.4f}\n'.format(args.data, val_loss, val_acc))
            logfile.flush()

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            print(' Saving the best checkpoint to {}...'.format(SAVE_PATH))
            torch.save({
                'model': model.state_dict(),
                'best_epoch': best_epoch,
                'best_acc': best_acc,
                'epoch': epoch
            }, SAVE_PATH)
            print('Best site-{} | Epoch:{} | Test Acc: {:.4f}'.format(args.data, best_epoch, best_acc))
            if log:
                logfile.write('Best site-{} | Epoch:{} | Test Acc: {:.4f}\n'.format(args.data, best_epoch, best_acc))
            
            _, test_acc = test(model, test_loader, loss_fun, device)
            print('Test site-{} | Epoch:{} | Test Acc: {:.4f}'.format(args.data, best_epoch, test_acc))
            if log:
                logfile.write('Test site-{} | Epoch:{} | Test Acc: {:.4f}\n'.format(args.data, best_epoch, test_acc))

    
    if log:
        logfile.flush()
        logfile.close()