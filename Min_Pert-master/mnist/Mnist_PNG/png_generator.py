# -*- coding: utf-8 -*-
# @Time    : 2020/9/14 16:23
# @Author  : CM
# @File    : png_generator.py
# @Software: PyCharm

import torch
from torchvision import transforms
import torchvision
import random
import numpy as np
import matplotlib.pyplot as plt


# load data, 定义数据转换格式
mnist_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)), transforms.Lambda(lambda x : x.resize_(28*28))])
# 导入数据，定义数据接口
traindata = torchvision.datasets.MNIST(root="D:\Adversary Samples\Min_Pert-copy\mnist", train=True, download=False, transform=mnist_transform)
testdata  = torchvision.datasets.MNIST(root="D:\Adversary Samples\Min_Pert-copy\mnist", train=False, download=False, transform=mnist_transform)
trainloader = torch.utils.data.DataLoader(traindata, batch_size=256, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testdata, batch_size=256, shuffle=True, num_workers=0)

#从各类中随机挑选出测试样本
class_samples = []
# count_class = 5
for i in range(10):
    a = testdata.data[testdata.targets == i]
    class_samples.append(a)#[(0, 1, 2, 3, 4),:,:]
#
count_class = 5
index = []
for i in range(10):
    index.append(random.sample(range(0, len(class_samples[i])), count_class))
#print(index)
samples_by_class = []#np.zeros(shape=[10,5],dtype=np.float32)
for i in range(10):
    samples_by_class.append(class_samples[i][(index[i]),:,:].numpy().astype(np.float32))

# print(samples_by_class)
for i in range(10):
    for j in range(5):
        image = samples_by_class[i][j,:,:]
        image.reshape(28, 28)
        img = transforms.ToPILImage()(image)
        plt.imshow(img,cmap='gray')
        plt.imsave('label{}_{}.png'.format(str(i),str(j)),img,cmap='gray')
        plt.show()
