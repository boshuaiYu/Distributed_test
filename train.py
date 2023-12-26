import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.transforms import transforms

import argparse
import time
import random
import numpy as np

from models.model import TestConvModel


def setup_seed(seed: int) -> None:
    random.seed(seed)  # 正常计算的随机种子
    np.random.seed(seed)  # numpy计算的随机种子
    torch.manual_seed(seed)  # torch计算的随机种子
    torch.cuda.manual_seed(seed)  # 单GPU计算的随机种子
    torch.cuda.manual_seed_all(seed)  # 多GPU计算的随机种子

    torch.backends.cudnn.deterministic = True  # 固定cuda的随机数种子，每次返回的卷积结果确定
    torch.backends.cudnn.benchmark = False  # 关闭benchmark加速效果，保证可复现性

def train():
    model = TestConvModel(in_channels=1, num_classes=10)
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                                 transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batches,
                                               shuffle=True,
                                               num_workers=4,
                                               pin_memory=True,
                                               sampler=None)
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            output = model(images)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1,
                                                                         len(train_loader), loss.item()))
    print('训练时间:', (time.time() - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Original Training")
    parser.add_argument("-e", "--epochs", default=2, type=int, metavar="N", help="Epochs of training")
    parser.add_argument("-b", "--batches", default=16, type=int, metavar="N", help="Batches of training")
    parser.add_argument("-l", "--lr", default=1e-3, type=float, metavar="N", help="Learning rate of training")
    parser.add_argument("-s", "--seed", default=100, type=int, metavar="N", help="Seed of training")
    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    train()
