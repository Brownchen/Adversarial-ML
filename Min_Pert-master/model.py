# -*- coding: utf-8 -*-
# @Time    : 2020/9/9 10:13
# @Author  : CM
# @File    : model.py
# @Software: PyCharm

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torch.optim as optim
from tqdm import *
from torchvision import transforms
import matplotlib.pyplot as plt



import os

class Basic_MNIST(nn.Module):
    def __init__(self):
        super(Basic_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5,1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 50, 5,1)
        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 50 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# load data, 定义数据转换格式
# mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x : x.resize_(28*28))])
# # 导入数据，定义数据接口
# traindata = torchvision.datasets.MNIST(root="./mnist", train=True, download=True, transform=mnist_transform)
# testdata  = torchvision.datasets.MNIST(root="./mnist", train=False, download=True, transform=mnist_transform)
# trainloader = torch.utils.data.DataLoader(traindata, batch_size=256, shuffle=True, num_workers=0)
# testloader = torch.utils.data.DataLoader(testdata, batch_size=256, shuffle=True, num_workers=0)
def get_mnist_loader(bs=1, size=(28, 28), test=True):
    if (test != True):

        imageset = torchvision.datasets.MNIST(root='./data', train=True,
                                              download=True, transform=transforms.Compose([transforms.ToTensor()]))
        imageloader = torch.utils.data.DataLoader(imageset, batch_size=bs,
                                                  shuffle=False, num_workers=0)
    else:
        imageset = torchvision.datasets.MNIST(root='./data', train=False,
                                              download=True, transform=transforms.Compose([transforms.ToTensor()]))
        imageloader = torch.utils.data.DataLoader(imageset, batch_size=bs,
                                                  shuffle=False, num_workers=0)

    return imageloader, imageset



net = Basic_MNIST()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-04)

trainloader, trainset = get_mnist_loader(256,False)

#Training stage
num_epoch = 200
# for epoch in tqdm(range(num_epoch)):
#     losses = 0.0
#     for i, data in enumerate(trainloader):
#         inputs, labels = data
#         inputs, labels = Variable(inputs.view(-1,1,28,28)), Variable(labels)
#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = loss_function(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         losses += loss.data.item()
#     print("*****************当前平均损失为{}*****************".format(losses/2000.0))

net = torch.load('mnist_net_conv.pkl') # 加载模型

testloader, testset = get_mnist_loader(256)
#evaluate stage
correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images.view(-1,1,28,28)))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print("预测准确率为：{}/{}".format(correct, total))


# 保存整个网络
# torch.save(net, 'mnist_net_conv.pkl')
# # 保存网络中的参数, 速度快，占空间少
# torch.save(net.state_dict(),'mnist_net_conv_param.pkl')
#针对上面一般的保存方法，加载的方法分别是：
# model_dict=torch.load(PATH)
# model_dict=model.load_state_dict(torch.load(PATH))

net = torch.load('mnist_net_conv.pkl') # 加载模型

index = 100 # 选择测试样本
image = testset.data[index].float()
label = testset.targets[index].float()
print('label:', label)

outputs = net(Variable(image.view(-1,1,28,28)))
# temp = torch.argmax(outputs.data,0)
predicted = torch.max(outputs.data,1)[1]
print('预测值为：{}'.format(predicted[0]))

# image.resize_(28,28)
# img = transforms.ToPILImage()(image)
# plt.imshow(img, cmap='gray')
# plt.show()



