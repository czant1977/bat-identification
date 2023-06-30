# coding: utf-8
"""
通过实现Grad-CAM学习module中的forward_hook和backward_hook函数
"""
import cv2
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as function
import torchvision.transforms as transforms
import natsort

from model import *


def img_transform(img_in, transform):
    """
    将img进行预处理，并转换成模型输入所需的形式—— B*C*H*W
    :param transform:
    :param img_in:
    :param img_roi: np.array
    :return:
    """
    img = img_in.copy()
    img = Image.fromarray(np.uint8(img))
    img = transform(img)
    img = img.unsqueeze(0)  # C*H*W --> B*C*H*W
    return img


def img_preprocess(img_in):
    """
    读取图片，转为模型可读的形式
    :param img_in: ndarray, [H, W, C]
    :return: PIL.image
    """
    img = img_in.copy()
    img = cv2.resize(img, (224, 224))
    img = img[:, :, ::-1]  # BGR --> RGB
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    img_input = img_transform(img, transform)
    return img_input


def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())


def farward_hook(module, input, output):
    fmap_block.append(output)


def show_cam_on_image(img, mask, out_dir, image_name, predict, real, number):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    name_cam_img = number + "_cam_" + predict + "-" + real + ".jpg"
    name_raw_img = number + "_raw_" + predict + "-" + real + ".jpg"

    path_cam_img = os.path.join(out_dir, name_cam_img)
    path_raw_img = os.path.join(out_dir, name_raw_img)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cv2.imwrite(path_cam_img, np.uint8(255 * cam))
    cv2.imwrite(path_raw_img, np.uint8(255 * img))
    # print("predict: {}, real: {}".format(predict, real))


def comp_class_vec(ouput_vec, index=None):
    """
    计算类向量
    :param ouput_vec: tensor
    :param index: int，指定类别
    :return: tensor
    """
    if not index:
        index = np.argmax(ouput_vec.cpu().data.numpy())
    else:
        index = np.array(index)
    index = index[np.newaxis, np.newaxis]
    index = torch.from_numpy(index)
    one_hot = torch.zeros(1, 7).scatter_(1, index, 1) # one_hot = torch.zeros(1, 8).scatter_(1, index, 1)
    one_hot.requires_grad = True
    class_vec = torch.sum(one_hot * output)  # one_hot = 11.8605

    return class_vec


def gen_cam(feature_map, grads):
    """
    依据梯度和特征图，生成cam
    :param feature_map: np.array， in [C, H, W]
    :param grads: np.array， in [C, H, W]
    :return: np.array, [H, W]
    """
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam shape (H, W)

    weights = np.mean(grads, axis=(1, 2))  #

    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= np.min(cam)
    cam /= np.max(cam)

    return cam


if __name__ == '__main__':
    # 修改GPU卡号
    torch.cuda.set_device(0)

    classes = ("1", "2", "3", "4", "5", "6", "7")
    classes_number = [1, 2, 3, 4, 5, 6, 7]

    # 相关参数
    ######################################################
    # 0: EfficientNet
    # 1: ResNet50
    # 2: VGG16
    # 3: MobileNetV2
    # 4: SqueezeNet-1.0
    # 5: GoogleNet
    # 6: InceptionV3
    # 7: ShuffleNetV2-1.0
    # 8: AlexNet
    #####################################################
    model = 5
    net_path = os.path.join("./", "weights_7_classes", "other_best", "GoogleNet_9014.pth")
    BASE_DIR = "./Rhinolophid_part_7_classes/test"

    # 加载模型
    net = torch.load(net_path)  # 模型加载路径
    net = net.cpu()
    net.eval()  # 测试模式

    # 注册Hook
    if model == 0:
        net._conv_head.register_forward_hook(farward_hook)
        net._conv_head.register_backward_hook(backward_hook)
    elif model == 1:
        net.layer4[2].conv3.register_forward_hook(farward_hook)
        net.layer4[2].conv3.register_backward_hook(backward_hook)
    elif model == 2:
        net.features[40].register_forward_hook(farward_hook)
        net.features[40].register_backward_hook(backward_hook)
    elif model == 3:
        net.features[18][0].register_forward_hook(farward_hook)
        net.features[18][0].register_backward_hook(backward_hook)
    elif model == 4:
        # net.features[12].expand3x3.register_forward_hook(farward_hook)
        # net.features[12].expand3x3.register_backward_hook(backward_hook)
        net.classifier[1].register_forward_hook(farward_hook)
        net.classifier[1].register_backward_hook(backward_hook)
    elif model == 5:
        net.inception5b.branch4[1].conv.register_forward_hook(farward_hook)
        net.inception5b.branch4[1].conv.register_backward_hook(backward_hook)
    elif model == 6:
        net.Mixed_7c.branch_pool.conv.register_forward_hook(farward_hook)
        net.Mixed_7c.branch_pool.conv.register_backward_hook(backward_hook)
    elif model == 7:
        net.conv5[0].register_forward_hook(farward_hook)
        net.conv5[0].register_backward_hook(backward_hook)
    elif model == 8:
        net.features[12].register_forward_hook(farward_hook)
        net.features[12].register_backward_hook(backward_hook)

    # 预测错误数列表
    error_list = list()
    # 真实标签列表
    real_idx_list = list()
    # 预测值列表
    pre_idx_list = list()

    # 遍历8个文件夹
    for now_class in classes:
        error_count = 0
        error_class = list()

        class_path = os.path.join(BASE_DIR, now_class)

        output_dir = os.path.join("./", "gradCAM_result", now_class)
        # 获取所有图片
        image_list = os.listdir(class_path)
        # 排序
        image_list = natsort.natsorted(image_list)

        # 这一块代码用于填写真实标签列表
        number = classes.index(now_class)
        temp = [number + 1] * len(image_list)
        real_idx_list.extend(temp)

        # 遍历整个文件
        for i in range(len(image_list)):
            image_name = image_list[i]
            image_path = os.path.join(class_path, image_name)

            fmap_block = list()
            grad_block = list()

            # 图片读取
            img = cv2.imread(image_path, 1)  # H*W*C
            img_input = img_preprocess(img)  # 将图片转化为模型可读的形式

            # forward
            output = net(img_input)
            # 预测值
            idx = np.argmax(output.cpu().data.numpy())
            # _, idx = torch.max(output.data, 1)
            
            # 这一块代码用于填写预测值列表
            pre_idx_list.append(idx + 1)
            
            # backward
            net.zero_grad()
            class_loss = comp_class_vec(output)
            class_loss.backward()

            # 生成cam
            grads_val = grad_block[0].cpu().data.numpy().squeeze()
            fmap = fmap_block[0].cpu().data.numpy().squeeze()
            cam = gen_cam(fmap, grads_val)

            # 保存cam图片
            img_show = np.float32(cv2.resize(img, (224, 224))) / 255
            # show_cam_on_image(img_show, cam, out_dir=output_dir, image_name=image_name, predict=classes[idx],
            #                   real=now_class, number=str(i + 1))

            if classes[idx] != now_class:
                error_count += 1
                error_class.append(classes[idx])

        print('Class: {}, Error Classes: '.format(now_class), error_class)
        error_list.append(error_count)

    print(real_idx_list)
    print(pre_idx_list)

print('-' * 30)
print("Error list: ", error_list)
