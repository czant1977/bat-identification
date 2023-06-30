from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
import os
import gc

# 如果使用上面的Git工程的话这样导入
# from efficientnet.model import EfficientNet
# 如果使用pip安装的Efficient的话这样导入
from model_ca_original import *
from utils import efficientnet

# some parameters
use_gpu = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
data_dir = './Rhinolophid_part'
batch_size = 64
lr = 0.001 # 初始学习率
momentum = 0.9 # SGD优化器参数
num_epochs = 200 # 训练轮次
input_size = 224 # 输入图像大小
class_num = 8 # 分类类别
net_name = 'efficientnet-b0'
net = 0
weight_decay = 0.0001 # 权重衰减

# 文件参数
GPU = 3 # 使用的GPU卡号
# torch.backends.cudnn.enabled = False  # 禁用 cudnn 加速
running_train = True
using_ca = True
ca_path = 'weights/efficientnet-b0pth'
eff_path = 'weights/efficientnet-b0pth'

# 学习率调节器参数
milestones = [150, 200] # 学习率更新的epoch
gamma = 0.5 # 学习率衰减


def loaddata(data_dir, batch_size, set_name, shuffle):
    data_transforms = {
        'train': transforms.Compose([
            # transforms.Resize(input_size),
            # brightness：亮度 hue：色调 contrast：对比度
            # transforms.ColorJitter(brightness=0.3, hue=0.3, contrast=0.3),
            # transforms.CenterCrop(input_size),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            # transforms.Resize(input_size),
            # transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in [set_name]}
    # num_workers=0 if CPU else =1
    dataset_loaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                      batch_size=batch_size,
                                                      shuffle=shuffle, num_workers=0) for x in [set_name]}  # 原 num_workers=1
    data_set_sizes = len(image_datasets[set_name])
    return dataset_loaders, data_set_sizes


# 模型、损失函数、优化器、外部学习率调节器、训练次数
def train_model(model_ft, criterion, optimizer, lr_scheduler, num_epochs=50):
    train_sign = 150
    print_sign = True
    train_loss = []  # 训练损失值
    since = time.time()
    best_model_wts = model_ft.state_dict()
    best_acc = 0.0  # 最高准确度
    model_ft.train(True)

    # 定义学习率修改器
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=gamma)

    for epoch in range(num_epochs):
        # 加载数据，返回值：数据集、图片大小
        dset_loaders, dset_sizes = loaddata(data_dir=data_dir, batch_size=batch_size, set_name='train', shuffle=True)

        if print_sign:
            print('Loading Data Size: ', dset_sizes)
            print_sign = False

        print('-' * 25)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # optimizer = lr_scheduler(optimizer, epoch, init_lr=lr)  # 随时更新学习率
        scheduler.step()

        print('LR is set to [{}]'.format(optimizer.state_dict()['param_groups'][0]['lr']))

        running_loss = 0.0  # 运行损失
        running_corrects = 0
        count = 0

        for data in dset_loaders['train']:
            inputs, labels = data
            labels = torch.squeeze(labels.type(torch.LongTensor))
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            outputs = model_ft(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs.data, 1)  # 最大值及其下标

            # 训练神经网络
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() # 更新权重

            count += 1
            if count % 30 == 0 or outputs.size()[0] < batch_size:
                # print('Epoch:{}: loss:{:.3f}'.format(epoch, loss.item()))
                train_loss.append(loss.item())

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dset_sizes
        epoch_acc = running_corrects.double() / dset_sizes

        print('Loss: {:.4f} Acc: {:.4f}'.format(
            epoch_loss, epoch_acc))

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model_ft.state_dict()  # 模型最佳权重的 state_dict
        if epoch_acc > 0.999 and train_sign !=0:
            train_sign -= 1
        elif epoch_acc > 0.999 and train_sign == 0:
            break

        # 测试集精度
        test_model(model_ft, criterion)
        torch.save(model_ft, './weights_test/' + 'ca' + str(epoch) + 'pth')
        model_ft.train(True)

        # 每个关键节点 epoch 清除一次内存
        gc.collect()
        torch.cuda.empty_cache()

    # 训练完成

    # 模型保存路径
    model_out_path = './weights_final/' + net_name + 'pth'
    # 保存最后训练模型
    torch.save(model_ft, model_out_path)

    final_model_wts = model_ft.state_dict()

    # 将最佳训练权重替换到当前模型
    model_ft.load_state_dict(best_model_wts)
    # 模型的保存路径
    model_out_path = './weights/' + net_name + 'pth'
    # 保存模型
    torch.save(model_ft, model_out_path)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return train_loss, best_model_wts, final_model_wts


def test_model(model, criterion):
    model.eval()  # 进入测试模式
    running_loss = 0.0
    running_corrects = 0
    cont = 0
    outPre = []
    outLabel = []
    dset_loaders, dset_sizes = loaddata(data_dir=data_dir, batch_size=8, set_name='test', shuffle=False)
    for data in dset_loaders['test']:
        inputs, labels = data
        labels = torch.squeeze(labels.type(torch.LongTensor))
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)  # 按列输出最大值及其索引

        loss = criterion(outputs, labels)
        if cont == 0:
            outPre = outputs.data.cpu()
            outLabel = labels.data.cpu()
        else:
            outPre = torch.cat((outPre, outputs.data.cpu()), 0)
            outLabel = torch.cat((outLabel, labels.data.cpu()), 0)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        cont += 1
    print('Loss: {:.4f} Acc: {:.4f}'.format(running_loss / dset_sizes,
                                            running_corrects.double() / dset_sizes))


def exp_lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=10):
    """Decay learning rate by a f#            model_out_path ="./model/W_epoch_{}.pth".format(epoch)
#            torch.save(model_W, model_out_path) actor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.8**(epoch // lr_decay_epoch))
    print('LR is set to {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


if running_train:
    # 训练并测试

    # 修改GPU卡号
    torch.cuda.set_device(GPU)

    stage_B0 = 16
    channel_list_B0 = [32, 96, 144, 144, 240, 240, 480, 480, 480, 672, 672, 672, 1152, 1152, 1152, 1152]
    num_squeezed_channels_list = [8, 4, 6, 6, 10, 10, 20, 20, 20, 28, 28, 28, 48, 48, 48, 48]

    channel_dict = {0: channel_list_B0}
    stage_dict = {0: stage_B0}

    pth_map = {
        'efficientnet-b0': 'efficientnet-b0-355c32eb.pth',
        'efficientnet-b1': 'efficientnet-b1-f1951068.pth',
        'efficientnet-b2': 'efficientnet-b2-8bb594d6.pth',
        'efficientnet-b3': 'efficientnet-b3-5fb5a3c3.pth',
        'efficientnet-b4': 'efficientnet-b4-6ed6700e.pth',
        'efficientnet-b5': 'efficientnet-b5-b6417697.pth',
        'efficientnet-b6': 'efficientnet-b6-c76e70fd.pth',
        'efficientnet-b7': 'efficientnet-b7-dcc49843.pth',
    }

    # 构建模型
    blocks_args, global_params = efficientnet(width_coefficient=1.0, depth_coefficient=1.0,
                                              image_size=224, num_classes=8)
    model_ft = EfficientNet(blocks_args=blocks_args, global_params=global_params)

    if using_ca:
        for i in range(stage_dict[net]):
            model_ft._blocks[i]._se_reduce = CoordAtt(channel_dict[net][i], channel_dict[net][i], mip=num_squeezed_channels_list[i])
            print('Add CA to EfficientNet!')

    # 修改全连接层的层数
    num_ftrs = model_ft._fc.in_features
    model_ft._fc = nn.Linear(num_ftrs, class_num)
    # 修改Dropout层
    model_ft._dropout = nn.Dropout(0.2)

    # # 离线加载预训练，需要事先下载好
    # model_ft = EfficientNet.from_name(net_name)
    # net_weight = 'eff_weights/' + pth_map[net_name]
    # state_dict = torch.load(net_weight)
    # model_ft.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss()
    if use_gpu:
        model_ft = model_ft.cuda()
        criterion = criterion.cuda()

    # 使用SGD优化器
    # optimizer = optim.SGD((model_ft.parameters()), lr=lr, momentum=momentum, weight_decay=0.0004)

    # 使用Adam优化器
    optimizer = optim.Adam(model_ft.parameters(), lr=lr, weight_decay=weight_decay)

    # 开始训练
    train_loss, best_model_wts, final_model_wts = train_model(model_ft, criterion, optimizer=optimizer, lr_scheduler=exp_lr_scheduler, num_epochs=num_epochs)

    # 开始测试
    print('-' * 25)
    print('Test Accuracy:')

    print('The best model test Accuracy:')
    # 权重替换
    model_ft.load_state_dict(best_model_wts)
    test_model(model_ft, criterion)

    print('The final model test Accuracy:')
    # 权重替换
    model_ft.load_state_dict(final_model_wts)
    test_model(model_ft, criterion)

else:
    # 仅测试
    print('-' * 25)
    print('Test Accuracy:')
    criterion = nn.CrossEntropyLoss()
    if using_ca:
        model = torch.load(ca_path)  # 模型加载路径
        print("Using EfficientNet-CA model: {}".format(ca_path))
    else:
        model = torch.load(eff_path)
        print("Using original EfficientNet model: {}".format(eff_path))

    if use_gpu:
        model = model.cuda()
        criterion = criterion.cuda()

    test_model(model, criterion)
