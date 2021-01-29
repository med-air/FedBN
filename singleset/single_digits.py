"""
Singleset on benchmark exp.
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
from nets.models import DigitModel
import argparse
from utils import data_utils

def prepare_data():
    # Prepare data
    transform_mnist = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_svhn = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_usps = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_synth = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_mnistm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # MNIST
    mnist_trainset     = data_utils.DigitsDataset(data_path="../data/MNIST", channels=1, percent=args.percent, train=True,  transform=transform_mnist)
    mnist_testset      = data_utils.DigitsDataset(data_path="../data/MNIST", channels=1, percent=args.percent, train=False, transform=transform_mnist)
    # SVHN
    svhn_trainset      = data_utils.DigitsDataset(data_path='../data/SVHN', channels=3, percent=args.percent,  train=True,  transform=transform_svhn)
    svhn_testset       = data_utils.DigitsDataset(data_path='../data/SVHN', channels=3, percent=args.percent,  train=False, transform=transform_svhn)
    # USPS
    usps_trainset      = data_utils.DigitsDataset(data_path='../data/USPS', channels=1, percent=args.percent,  train=True,  transform=transform_usps)
    usps_testset       = data_utils.DigitsDataset(data_path='../data/USPS', channels=1, percent=args.percent,  train=False, transform=transform_usps)
    # Synth Digits
    synth_trainset     = data_utils.DigitsDataset(data_path='../data/SynthDigits/', channels=3, percent=args.percent,  train=True,  transform=transform_synth)
    synth_testset      = data_utils.DigitsDataset(data_path='../data/SynthDigits/', channels=3, percent=args.percent,  train=False, transform=transform_synth)
    # MNIST-M
    mnistm_trainset     = data_utils.DigitsDataset(data_path='../data/MNIST_M/', channels=3, percent=args.percent,  train=True,  transform=transform_mnistm)
    mnistm_testset      = data_utils.DigitsDataset(data_path='../data/MNIST_M/', channels=3, percent=args.percent,  train=False, transform=transform_mnistm)


    if args.data.lower() == 'mnist':
        train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=args.batch, shuffle=True)
        test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=args.batch, shuffle=False)
    elif args.data.lower() == 'svhn':
        train_loader = torch.utils.data.DataLoader(svhn_trainset, batch_size=args.batch, shuffle=True)
        test_loader = torch.utils.data.DataLoader(svhn_testset, batch_size=args.batch, shuffle=False)
    elif args.data.lower() == 'usps':
        train_loader = torch.utils.data.DataLoader(usps_trainset, batch_size=args.batch, shuffle=True)
        test_loader = torch.utils.data.DataLoader(usps_testset, batch_size=args.batch, shuffle=False)
    elif args.data.lower() == 'synth':
        train_loader = torch.utils.data.DataLoader(synth_trainset, batch_size=args.batch, shuffle=True)
        test_loader = torch.utils.data.DataLoader(synth_testset, batch_size=args.batch, shuffle=False)
    elif args.data.lower() == 'mnistm':
        train_loader = torch.utils.data.DataLoader(mnistm_trainset, batch_size=args.batch, shuffle=True)
        test_loader = torch.utils.data.DataLoader(mnistm_testset, batch_size=args.batch, shuffle=False)
    else:
        raise ValueError('Unknown dataset')
    return train_loader, test_loader

def train(data_loader, optimizer, loss_fun, device):
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        optimizer.zero_grad()

        data = data.to(device).float()
        target = target.to(device).long()
        output = model(data)
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss.backward()
        optimizer.step()

    return loss_all / len(data_loader), correct/total

def test(data_loader,site, loss_fun, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device).float()
            target = target.to(device).long()
            output = model(data)
            test_loss += loss_fun(output, target).item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()
            total += target.size(0)

    test_loss /= len(data_loader)
    correct /= total
    print(' {} | Test loss: {:.4f} | Test acc: {:.4f}'.format(site, test_loss, correct))

    if log:
        logfile.write(' {} | Test loss: {:.4f} | Test acc: {:.4f}\n'.format(site, test_loss, correct))
    

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--percent', type = float, default= 0.1, help ='percentage of dataset to train')
    parser.add_argument('--log', action='store_true', help='whether to log')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--batch', type = int, default= 32, help ='batch size')
    parser.add_argument('--data', type = str, default= 'svhn', help='[svhn | usps | synth | mnistm | mnist]')
    parser.add_argument('--save_path', type = str, default='../checkpoint/digits', help='path to save the checkpoint')
    args = parser.parse_args()

    exp_folder = 'singleset_digits'

    args.save_path = os.path.join(args.save_path, exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path,'SingleSet_{}'.format(args.data))

    log = args.log
    if log:
        log_path = os.path.join('../logs/digits', exp_folder)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logfile = open(os.path.join(log_path,'SingleSet_{}.log'.format(args.data)), 'a')
        logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        logfile.write('===Setting===\n')
        logfile.write('    lr: {}\n'.format(args.lr))
        logfile.write('    batch: {}\n'.format(args.batch))
        logfile.write('    dataset: {}\n'.format(args.data))
        logfile.write('    epochs: {}\n'.format(args.epochs))

    model = DigitModel().to(device)
    loss_fun = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    train_loader, test_loader = prepare_data()

    for epoch in range(args.epochs):
        print(f"Epoch: {epoch}" )
        loss,acc = train(train_loader, optimizer, loss_fun, device)
        print(' {} | Train loss: {:.4f} | Train acc : {:.4f}'.format(args.data, loss,acc))

        if log:
            logfile.write('Epoch Number {}\n'.format(epoch))
            logfile.write('Train loss: {:.4f} and accuracy : {:.4f}\n'.format(loss, acc))
            logfile.flush()

        test(test_loader, args.data, loss_fun, device)

    print(' Saving the best checkpoint to {}...'.format(SAVE_PATH))
    torch.save({
        'model': model.state_dict(),
        'epoch': epoch
    }, SAVE_PATH)

    if log:
        logfile.flush()
        logfile.close()
