# -*- coding: utf-8 -*-
# @Time    : 2020/9/7 16:00
# @Author  : CM
# @File    : approaches.py
# @Software: PyCharm

import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
from torch.autograd.gradcheck import zero_gradients
from PIL import Image
import time
import datetime
import os
from torchvision import transforms
# from tqdm import *
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    f_image = net.forward(image).data.numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]
    label = I[0]

    input_shape = image.data.squeeze().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0
    # max_iter = 50
    # overshoot = 0.0

    x = Variable(pert_image, requires_grad=True)
    fs = net.forward(x)
    fs_list = [fs[0][I[k]] for k in range(len(I))]
    k_i = label

    while k_i == label and loop_i < max_iter:
        pert = np.inf
        fs[0][I[0]].backward(retain_graph=True)
        orig_grad = x.grad.data.numpy().copy()

        for k in range(len(I)):
            zero_gradients(x)
            fs[0][I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.numpy().copy()

            w_k = cur_grad - orig_grad
            f_k = (fs[0][I[k]] - fs[0][I[0]]).data.numpy()

            pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())
            if pert_k < pert:
                pert = pert_k
                w = w_k
        r_i = (pert + 1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot)
        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.numpy().flatten())
        loop_i += 1
    r_tot = (1 + overshoot) * r_tot

    return r_tot, loop_i, label, k_i, pert_image

def boundary_attack(initial_sample,target_sample):
    # classifier = ResNet50(weights='imagenet')
    classifier = torch.load('mnist_net_conv.pkl')  # 加载模型

    writer = SummaryWriter('log')
    # initial_sample = preprocess('images/original/awkward_moment_seal.png')
    # target_sample = preprocess('images/original/bad_joke_eel.png')
    # folder = time.strftime('%Y%m%d_%H%M%S', datetime.datetime.now().timetuple())
    # os.makedirs(os.path.join("images", folder))
    # draw(np.copy(initial_sample), classifier, folder)
    mnist_show(initial_sample,'initial sample', fontsize=15)
    mnist_show(target_sample,'target_sample', fontsize=15, step=-1)
    attack_class = np.argmax(classifier(torch.from_numpy(initial_sample).float().reshape(-1, 1, 28, 28)).detach().numpy())#6
    target_class = np.argmax(classifier(torch.from_numpy(target_sample).float().reshape(-1, 1, 28, 28)).detach().numpy())#2

    adversarial_sample = initial_sample
    n_steps = 0
    n_calls = 0
    epsilon = 1.
    delta = 0.5

    # Move first step to the boundary
    while True:
        trial_sample = adversarial_sample + forward_perturbation(epsilon * get_diff(adversarial_sample, target_sample), adversarial_sample, target_sample)
        prediction = classifier(torch.from_numpy(trial_sample).reshape(-1, 1, 28, 28)).detach().numpy()
        n_calls += 1
        temp = np.argmax(prediction)
        if np.argmax(prediction) == attack_class:
            adversarial_sample = trial_sample
            break
        else:
            epsilon *= 0.9
    while True:
        print("Step #{}...".format(n_steps))
        print("\tDelta step...")
        d_step = 0
        while True:
            d_step += 1
            print("\t#{}".format(d_step))
            trial_samples = []
            for i in np.arange(10):
                trial_sample = adversarial_sample + orthogonal_perturbation(delta, adversarial_sample, target_sample)
                trial_samples.append(trial_sample)
            predictions = []
            for i in np.arange(10):
                prediction = classifier(torch.from_numpy(trial_samples[i]).float().reshape(-1, 1, 28, 28)).detach().numpy()
                predictions.append(prediction.squeeze(0))
            # predictions = classifier(torch.from_numpy(np.array(trial_samples)).reshape(-1, 1, 28, 28))
            n_calls += 10
            # temp = np.array(predictions)
            predictions = np.argmax(np.array(predictions), axis=1)
            d_score = np.mean(predictions == attack_class)
            if d_score > 0.0:
                if d_score < 0.3:
                    delta *= 0.9
                elif d_score > 0.6:
                    delta /= 0.9
                adversarial_sample = np.array(trial_samples)[np.where(predictions == attack_class)[0][0]]
                break
            else:
                delta *= 0.9
        print("\tEpsilon step...")
        e_step = 0
        epsilon = np.linalg.norm(adversarial_sample - target_sample).astype(np.float32)
        while True:
            e_step += 1
            print("\t#{}".format(e_step))
            trial_sample = adversarial_sample + forward_perturbation(epsilon * get_diff(adversarial_sample, target_sample), adversarial_sample, target_sample)
            # prediction = classifier(trial_sample.reshape(-1, 1, 28, 28))
            prediction = classifier(torch.from_numpy(trial_sample).float().reshape(-1, 1, 28, 28)).detach().numpy()
            n_calls += 1
            if np.argmax(prediction) == attack_class:
                adversarial_sample = trial_sample
                # epsilon /= 0.5
                break
            # elif e_step > 500:
            #         break
            else:
                epsilon *= 0.9
        n_steps += 1
        chkpts = [1, 5, 10, 50, 100, 500, 1000]
        if (n_steps in chkpts) or (n_steps % 500 == 0):
            print("{} steps".format(n_steps))
            draw(np.copy(adversarial_sample), classifier)
            # mnist_show(np.copy(adversarial_sample).astype(np.float32), str(n_steps))
        diff = np.mean(get_diff(adversarial_sample, target_sample))
        if diff <= 1e-3 or n_steps > 30000:
            print("{} steps".format(n_steps))
            MSE = np.mean((adversarial_sample - initial_sample) ** 2,dtype=np.float32)

            print("Mean Squared Error: {}".format(MSE))
            draw(np.copy(adversarial_sample), classifier)
            break
        print("Mean Squared Error: {}".format(diff))
        L2 = np.linalg.norm(np.reshape(adversarial_sample - initial_sample.reshape(1, 1, 28, 28), [-1])) / 784
        writer.add_scalar('MSE', diff, global_step=n_steps)
        title = 'step:{}, L2 distance:{}'.format(str(n_steps), L2)
        if n_steps % 2000 == 0:
            mnist_show(np.copy(adversarial_sample).astype(np.float32), title,step=n_steps)
        print("Calls: {}".format(n_calls))
        print("Attack Class: {}".format(attack_class))
        print("Target Class: {}".format(target_class))
        print("Adversarial Class: {}".format(np.argmax(prediction)))


def orthogonal_perturbation(delta, prev_sample, target_sample):
    prev_sample = prev_sample.reshape(1, 28, 28)
    # Generate perturbation
    perturb = np.random.randn(1, 28, 28)
    # perturb /= np.linalg.norm(perturb)
    perturb /= get_diff(perturb, np.zeros_like(perturb))
    # perturb *= delta * np.linalg.norm(target_sample - prev_sample)
    perturb *= delta * np.mean(get_diff(target_sample, prev_sample))
    # Project perturbation onto sphere around target
    diff = (target_sample - prev_sample).astype(np.float32)
    # diff /= np.linalg.norm(diff)
    diff /= get_diff(target_sample, prev_sample)
    diff = diff.reshape(28, 28)
    temp = np.dot(perturb, diff)
    perturb -= np.dot(perturb, diff) * diff
    # Check overflow and underflow
    overflow = (prev_sample + perturb) - np.ones_like(prev_sample) * (255)
    perturb -= overflow * (overflow > 0)
    underflow = np.zeros_like(prev_sample) - (prev_sample + perturb)
    perturb += underflow * (underflow > 0)
    return perturb

def forward_perturbation(epsilon, prev_sample, target_sample):
    perturb = (target_sample - prev_sample).astype(np.float32)
    # perturb /= np.linalg.norm(target_sample - prev_sample)
    perturb /= get_diff(target_sample, prev_sample)
    perturb *= epsilon
    return perturb

# def get_converted_prediction(sample, classifier):
#     sample = sample.reshape(224, 224, 3)
#     mean = [103.939, 116.779, 123.68]
#     sample[..., 0] += mean[0]
#     sample[..., 1] += mean[1]
#     sample[..., 2] += mean[2]
#     sample = sample[..., ::-1].astype(np.uint8)
#     sample = sample.astype(np.float32).reshape(1, 224, 224, 3)
#     sample = sample[..., ::-1]
#     mean = [103.939, 116.779, 123.68]
#     sample[..., 0] -= mean[0]
#     sample[..., 1] -= mean[1]
#     sample[..., 2] -= mean[2]
#     label = decode_predictions(classifier.predict(sample), top=1)[0][0][1]
#     return label

def draw(sample, name, step=0):
    sample = sample.reshape(28, 28).astype(np.uint8)
    sample = Image.fromarray(sample)
    # name =  time.strftime('_%Y%m%d_%H%M%S_', datetime.datetime.now().timetuple()) + str(step)
    filepath = os.path.join("images/mnist_step", "step_{}.png".format(step))
    sample.save(filepath)
    return filepath


# def preprocess(sample_path):
# 	img = image.load_img(sample_path, target_size=(224, 224))
# 	x = image.img_to_array(img)
# 	x = np.expand_dims(x, axis=0)
# 	x = preprocess_input(x)
# 	return x

# def get_diff(sample_1, sample_2):
#     return np.mean((sample_1 - sample_2) ** 2,dtype=np.float32)#.astype(np.float32)
def get_diff(sample_1, sample_2):
    sample_1 = sample_1.reshape(1, 28, 28)
    sample_2 = sample_2.reshape(1, 28, 28)
    diff = []
    for i, channel in enumerate(sample_1):
        diff.append(np.linalg.norm((channel - sample_2[i])))#.astype(np.float32)
    return np.array(diff)

def mnist_show(image,title, step=0, fontsize=12):
    # image.resize_(28,28)
    draw(np.copy(image), step)
    img = transforms.ToPILImage()(image.reshape(28, 28))
    plt.title(title, fontsize=fontsize)
    plt.imshow(img, cmap='gray')
    plt.show()