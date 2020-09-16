# -*- coding: utf-8 -*-
# @Time    : 2020/9/7 15:59
# @Author  : CM
# @File    : test_mnist.py
# @Software: PyCharm

import numpy as np
from numpy import mat
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import transforms
from tqdm import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import copy
from torch.autograd.gradcheck import zero_gradients
from approaches import deepfool, boundary_attack
from PIL import Image

from art.attacks.evasion import BoundaryAttack
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist,to_categorical

#Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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


def mnist_show(image):
    image.resize_(28,28)
    img = transforms.ToPILImage()(image)
    plt.imshow(img, cmap='gray')
    plt.show()

# test_transforms = transforms.Compose([
#      transforms.ToTensor(),
#      transforms.Normalize((0.1307,), (0.3081,))
# ])

# load data, 定义数据转换格式
mnist_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)), transforms.Lambda(lambda x : x.resize_(28*28))])
# 导入数据，定义数据接口
traindata = torchvision.datasets.MNIST(root="./mnist", train=True, download=True, transform=mnist_transform)
testdata  = torchvision.datasets.MNIST(root="./mnist", train=False, download=True, transform=mnist_transform)
trainloader = torch.utils.data.DataLoader(traindata, batch_size=256, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testdata, batch_size=256, shuffle=True, num_workers=0)


# 展示原始图片
# index = 100
# image = testdata[index][0]
# label = testdata[index][1]
# print(label)
# image.resize_(28,28)
# img = transforms.ToPILImage()(image)
# plt.imshow(img, cmap='gray')

# 展示图片
# index = 100
# batch = iter(testloader).next() #将testloader转换为迭代器
# image = batch[0][index]
# label = batch[1][index]
# print(label)
# image.resize_(28,28)
# img = transforms.ToPILImage()(image)
# plt.title('Original image')
# plt.imshow(img, cmap='gray')

#初始化模型、选择代价函数及优化器
# net = Net()
# loss_function = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-04)


#Training stage
# num_epoch = 50
# for epoch in tqdm(range(num_epoch)):
#     losses = 0.0
#     for data in trainloader:
#         inputs, labels = data
#         #inputs, labels = Variable(inputs), Variable(labels)
#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = loss_function(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         losses += loss.data.item()
#     print("*****************当前平均损失为{}*****************".format(losses/2000.0))


#evaluate stage
# correct = 0
# total = 0
# for data in testloader:
#     images, labels = data
#     outputs = net(Variable(images))
#     _, predicted = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     correct += (predicted == labels).sum()
# print("预测准确率为：{}/{}".format(correct, total))
#
#
# # 保存整个网络
# torch.save(net, 'mnist_net_all.pkl')
# # 保存网络中的参数
# torch.save(net.state_dict(),'mnist_net_param.pkl')


#inference stage
net = torch.load('mnist_net_conv.pkl') # 加载模型
# net = torch.load('mnist_net_all.pkl') # 加载模型

# 选择测试样本
# index = 100
# image = testdata[index][0]
# label = testdata[index][1]
# samples_by_class = np.load('samples_by_class.npz')['arr_0']
# samples_by_class = torch.from_numpy(samples_by_class.astype(np.float32))
# for i in range(10):
#     for j in range(5):
#         mnist_show(samples_by_class[i,j,:,:])

# outputs = net(Variable(samples_by_class[6,1,:,:].reshape(1,1,28,28)))#.view(1,1,28,28)
# predicted = torch.max(outputs.data,1)[1]
# print('预测值为：{}'.format(predicted[0]))



'''DeepFool'''
# net = torch.load('mnist_net_all.pkl') # 加载模型
# net = torch.load('mnist_net_conv.pkl') # 加载模型

def unnormal(sample, mean, std):
    sample = sample.reshape(28, 28)
    sample = sample * std + mean
    sample = sample.reshape(1, 1, 28, 28)
    return sample

# def test_deepFool(test_samples,net,label):
#     # index = 100 # 选择测试样本
#     # plt.imshow(test_samples.view(28,28), cmap='gray')
#     image = Variable(test_samples.resize_(1,1,28,28), requires_grad=True)
#     # label = torch.tensor([testdata[index][1]])
#     print('label:',label)
#
#     r, loop_i, label_orig, label_pert, pert_image = deepfool(image, net)
#
#     if label_orig != label_pert:
#         MSE = np.mean(((pert_image.detach().numpy() - test_samples.detach().numpy()) ** 2), dtype=np.float32)
#
#         with torch.no_grad():
#             # print(pert_image.shape)
#             outputs = net(pert_image.view(1,1,28,28))
#         #     outputs = net(pert_image.data.resize_(1,784))
#             predicted = torch.max(outputs.data,1)[1]
#             print('攻击后预测值为：{}'.format(predicted[0]))
#             print('\n')
#             pert_image = unnormal(pert_image, 0.1307, 0.3081)
#             pert_im = pert_image.view(28,28)
#             #pert_image.data.resize_(28,28)
#             plt.figure(figsize=(5,5))
#             img = transforms.ToPILImage()(pert_im)
#             plt.title('Predicted as class {}, MSE: {:.6f}'.format(predicted[0], MSE))
#             plt.imshow(img, cmap='gray')#, cmap='gray'
#             plt.show()
#
#             return pert_image
#
# for label in range(10):
#     for time in range(5):
#         img = Image.open('mnist/Mnist_PNG/label{}_{}.png'.format(label,time)).convert('L')
#         image = transforms.ToTensor()(img)
#         plt.figure(figsize=(5,5))
#         plt.imshow(image.reshape(28,28),cmap='gray')
#         outputs = net(Variable(image.reshape(1,1,28,28)))
#         predicted = torch.max(outputs.data,1)[1]
#         print('初始预测值为：{}'.format(predicted[0]))
#         pert_image = test_deepFool(image, net, label)
#         plt.imsave('mnist/Mnist_PNG/DeepFool_images/Perturbed image{}_{}.png'.format(label,time),pert_image.reshape(28,28),cmap='gray')
#         # perturbation = pert_image - img.tensor()
#         # plt.imshow(perturbation.reshape(28,28),cmap='gray')
#         # plt.imsave('mnist/Mnist_PNG/DeepFool_images/Perturbation{}_{}.png'.format(label,time),perturbation.reshape(28,28),cmap='gray')
#         # plt.show()




'''boundary attack'''

# initial_image = Image.open('mnist/Mnist_PNG/label{}_{}.png'.format(0,0)).convert('L')
# target_image = Image.open('mnist/Mnist_PNG/label{}_{}.png'.format(6,2)).convert('L')
# adv_image = boundary_attack(np.array(initial_image),np.array(target_image))
# classifier = torch.load('mnist_net_conv.pkl')
# adv_sample = mpimg.imread('images\mnist_step\_20200910_142129_20200910_125407.png')
# attack_class = np.argmax(classifier(torch.from_numpy(adv_sample).reshape(-1, 1, 28, 28)).detach().numpy())
# print(attack_class)


'''boundary attack'''
# def sample(label):
#     # _, (x_test, y_test) = mnist.load_data()
#     # x_test, y_test = testdata.data, testdata.targets#[0], testdata[1]
#     while True:
#         choice = np.random.choice(len(y_test))
#         if (y_test[choice] == label).all():
#             return x_test[choice]

#target
initial_image = Image.open('mnist/Mnist_PNG/label{}_{}.png'.format(0,0)).convert('L')
target_image = Image.open('mnist/Mnist_PNG/label{}_{}.png'.format(6,2)).convert('L')

# initial_img = transforms.ToTensor()(initial_image)
# target_img = transforms.ToTensor()(target_image)
initial_img = np.array(initial_image)
target_img = np.array(target_image)
# plt.figure(figsize=(5,5))
# plt.imshow(image.reshape(28,28),cmap='gray')
# outputs = net(Variable(initial_img.reshape(1,1,28,28)))
# predicted = torch.max(outputs.data,1)[1]
# print('初始预测值为：{}'.format(predicted[0]))
pert_image = boundary_attack(initial_img, target_img)
plt.figure(figsize=(5,5))
plt.imshow(pert_image.reshape(28,28),cmap='gray')
plt.show()
# plt.imsave('mnist/Mnist_PNG/DeepFool_images/Perturbed image{}_{}.png'.format(label,time),pert_image.reshape(28,28),cmap='gray')
# perturbation = pert_image - img.tensor()
# plt.imshow(perturbation.reshape(28,28),cmap='gray')
# plt.imsave('mnist/Mnist_PNG/DeepFool_images/Perturbation{}_{}.png'.format(label,time),perturbation.reshape(28,28),cmap='gray')
# plt.show()



# untargeted
# def boundary_attack_art(initial_image, net, label):
#
#     model = net
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-04)
#
#     # Step 3: Create the ART classifier
#
#     classifier = PyTorchClassifier(model=model,  loss=criterion,
#                                    optimizer=optimizer, input_shape=(1, 28, 28), nb_classes=10,device_type='cpu')
#
#     # Step 4: Train the ART classifier
#
#     # classifier.fit(x_train, y_train, batch_size=64, nb_epochs=3)
#
#     # Step 5: Evaluate the ART classifier on benign test examples
#
#     predictions = classifier.predict(initial_image.reshape(1,1,28,28))
#     # accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
#     # print('Accuracy on benign test examples: {}%'.format(accuracy * 100))
#     # initial_image = sample(to_categorical([9], 10))#.numpy().astype(np.float32)#.float()
#     # target_image = sample(to_categorical([1], 10))
#     # plt.figure(figsize=(5, 5))
#     # plt.title('Initial image')
#     # plt.imshow(initial_image[0,:,:], cmap='gray')
#     # plt.show()
#     #.view(1,28,28).numpy().astype(np.float32)#.float()
#
#     attack = BoundaryAttack(estimator=classifier, targeted=False, max_iter=0, delta=0.05, epsilon=1.)
#     iter_step = 200
#     x_adv = initial_image[0,:,:].numpy() #np.array([initial_image[0,:,:]])
#     # plt.title('Initial image')
#     # plt.imshow(x_adv[0].squeeze(0), cmap='gray')
#     # plt.show()
#     # L2 = []
#     for i in range(40):
#         x_adv = attack.generate(x=initial_image.reshape(1,1,28,28).numpy(), y=None, x_adv_init=x_adv.reshape(1,1,28,28))
#
#         # clear_output()
#         # MSE = np.mean(((x_adv[0] - initial_image.reshape(1,1,28,28).detach().numpy()) ** 2), dtype=np.float32)
#         # temp = np.reshape(x_adv[0] - initial_image.reshape(1,1,28,28).numpy(), [-1])
#         L2_error = np.linalg.norm(np.reshape(x_adv[0] - initial_image.reshape(1,1,28,28).numpy(), [-1])) / 784
#         print("Adversarial image at step %d." % (i * iter_step), "L2 error", L2_error,
#               "and class label %d." % np.argmax(classifier.predict(x_adv.reshape(1,1,28,28))[0]))
#         if i%10 == 0:
#             plt.figure(figsize=(5, 5))
#             plt.title('Predicted label: {}, L2 error: {:.5f}, Step {}'.format(np.argmax(classifier.predict(x_adv.reshape(1,1,28,28))[0]),
#                     L2_error, (i * iter_step)))
#             plt.imshow(x_adv[0][0,:,:], cmap='gray')
#             plt.show()
#
#         if hasattr(attack, 'curr_delta') and hasattr(attack, 'curr_epsilon'):
#             attack.max_iter = iter_step
#             attack.delta = attack.curr_delta
#             attack.epsilon = attack.curr_epsilon
#         else:
#             break
#         # L2.append(L2_error)
#
#     return x_adv, L2_error

# img = Image.open('mnist/Mnist_PNG/label{}_{}.png'.format(8,3)).convert('L')
# image = transforms.ToTensor()(img)
# plt.figure(figsize=(5,5))
# plt.title('Initial image')
# plt.imshow(image.reshape(28,28),cmap='gray')
# plt.show()
# outputs = net(Variable(image.reshape(1,1,28,28)))
# predicted = torch.max(outputs.data,1)[1]
# print('初始预测值为：{}'.format(predicted[0]))
# pert_image, L2_error = boundary_attack_art(image, net, 8)
# # L2.append(L2_error)
# pert_img = unnormal(pert_image, 0.1307, 0.3081)
# plt.figure(figsize=(5,5))
# plt.title('Perturbed image, L2-distance; {:.5f}'.format(L2_error))
# plt.imshow(pert_img.reshape(28, 28), cmap='gray')
# plt.show()

# for label in range(10):
#     L2 = []
#     for time in range(5):
#         img = Image.open('mnist/Mnist_PNG/label{}_{}.png'.format(label,time)).convert('L')
#         image = transforms.ToTensor()(img)
#         plt.figure(figsize=(5,5))
#         plt.title('Initial image')
#         plt.imshow(image.reshape(28,28),cmap='gray')
#         plt.show()
#         outputs = net(Variable(image.reshape(1,1,28,28)))
#         predicted = torch.max(outputs.data,1)[1]
#         print('初始预测值为：{}'.format(predicted[0]))
#         pert_image, L2_error = boundary_attack_art(image, net, label)
#         L2.append(L2_error)
#         pert_img = unnormal(pert_image, 0.1307, 0.3081)
#         plt.figure(figsize=(5,5))
#         plt.title('Perturbed image, L2-distance; {:.5f}'.format(L2_error))
#         plt.imshow(pert_img.reshape(28, 28), cmap='gray')
#         plt.show()
#     print('The mean squared L2-distance of class {} is {}'.format(label,np.mean(L2)))
#     print('********************************************************************************')
        # plt.imsave('mnist/Mnist_PNG/DeepFool_images/Perturbed image{}_{}.png'.format(label,time),pert_image.reshape(28,28),cmap='gray')
        # perturbation = pert_image - img.tensor()
        # plt.imshow(perturbation.reshape(28,28),cmap='gray')
        # plt.imsave('mnist/Mnist_PNG/DeepFool_images/Perturbation{}_{}.png'.format(label,time),perturbation.reshape(28,28),cmap='gray')
        # plt.show()