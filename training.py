
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
import os
import gc
import random
import numpy as np
from efficientnet_pytorch import EfficientNet as EfficientNet_Baseline 
from utils import efficientnet

# 如果使用上面的Git工程的话这样导入
# from efficientnet.model import EfficientNet
# 如果使用pip安装的Efficient的话这样导入
from model import EfficientNet as EfficientNet_CA
from model import *
from efficientnet_model import EfficientNet as EfficientNet_original
from SE import SE

# some parameters
use_gpu = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
data_dir = './Rhinolophid_part_7_classes' # ./Rhinolophid_part_7_classes
batch_size = 16
lr = 5E-4
lrf = 0.01
momentum = 0.9
num_epochs = 700
input_size = 224
class_num = 7
net_name = 'efficientnet-b0'
net = 0 # 选择网络的版本
weight_decay = 5E-2 # 权重衰减 0.0004、0.05

# 文件参数
GPU = 0 # 使用的GPU卡号

# torch.backends.cudnn.enabled = False  # 禁用 cudnn 加速
running_train = False
baseline_using_pre_training = False
using_ca = False
ca_use_pre_training = False

ca_path = ''
eff_path = ''

# 学习率调节器参数
milestones = [200, 300, 400, 450, 500, 550, 600, 650] # 学习率更新的epoch
gamma = 0.5 # 学习率衰减


def loaddata(data_dir, batch_size, set_name, shuffle):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(random.randint(180, 224)),
            transforms.Resize(input_size),
            # brightness：亮度 hue：色调 contrast：对比度 saturation：饱和度
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2, saturation=0.2),
            # transforms.CenterCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.RandomAffine(degrees=20, translate=(0.08, 0.08), shear=(-20, 20)),
            # transforms.RandomPerspective(distortion_scale=random.uniform(0.1, 0.2), p=1),
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
                                                      shuffle=shuffle, num_workers=1) for x in [set_name]}  # 原 num_workers=1
    data_set_sizes = len(image_datasets[set_name])
    return dataset_loaders, data_set_sizes


# 模型、损失函数、优化器、学习率调节器、训练次数
def train_model(model_ft, criterion, optimizer, lr_scheduler, num_epochs=50):
    print_sign = True
    best_test_acc = 0
    train_loss = []  # 训练损失值
    since = time.time()
    best_model_wts = model_ft.state_dict()
    best_acc = 0.0  # 最高准确度
    model_ft.train(True)

    # 定义学习率修改器
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=gamma)

    lf = lambda x: ((1 + math.cos(x * math.pi / num_epochs)) / 2) * (1 - lrf) + lrf  # cosine
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(num_epochs):
        # 加载数据，返回值：数据集、图片大小
        dset_loaders, dset_sizes = loaddata(data_dir=data_dir, batch_size=batch_size, set_name='train', shuffle=True)

        if print_sign:
            print('Loading Data Size [{}]'.format(dset_sizes))
            print_sign = False

        print('-' * 25)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # optimizer = lr_scheduler(optimizer, epoch, lr, 20)  # 随时更新学习率
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
            optimizer.step()

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

        # 测试集精度
        test_acc = test_model(model_ft, criterion)

        if test_acc >= best_test_acc:
            if test_acc >= 0.85:
                torch.save(model_ft, './weights_7_classes/best/' + 'ca_' + str(epoch) + '.pth')
            best_test_acc = test_acc
            print('The New Best test Acc: [{:.4f}] !'.format(best_test_acc))

        model_ft.train(True)

        # 每个关键节点 epoch 清除一次内存
        gc.collect()
        torch.cuda.empty_cache()

    # 训练完成

    # 模型保存路径
    model_out_path = './weights_7_classes/final/' + 'ca_' + str(epoch) + '.pth'
    # 保存最后训练模型
    torch.save(model_ft, model_out_path)

    final_model_wts = model_ft.state_dict()

    # 将最佳训练权重替换到当前模型
    model_ft.load_state_dict(best_model_wts)
    # 模型保存路径
    # model_out_path = './weights_best/' + net_name + '.pth'
    # 保存最佳模型
    # torch.save(model_ft, model_out_path)

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
    int = 0
    dset_loaders, dset_sizes = loaddata(data_dir=data_dir, batch_size=2, set_name='test', shuffle=False)

    # print("Loading Data [{}]".format(dset_sizes))

    for data in dset_loaders['test']:
        inputs, labels = data
        labels = torch.squeeze(labels.type(torch.LongTensor))
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        outputs = model(inputs)
        
        _, preds = torch.max(outputs.data, 1)  # 按行寻找最大值及其索引
        
        if int == 0:
            out = torch.tensor(preds)
            int = 1
        else:
            out = torch.cat((out, preds), dim=0)

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
    
    out = [ i + 1 for i in out.tolist()]
    
    print(outPre.tolist())
    print(out)

    return running_corrects.double() / dset_sizes


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

    # CA的默认设置为（基准取值）：      [8, 8, 8, 8, 8, 8, 15, 15, 15, 21, 21, 21, 36, 36, 36, 36]
    ca_squeezed_channels_list = [8, 8, 8, 8, 8, 8, 15, 15, 15, 21, 21, 21, 36, 36, 36, 36]
    # SE的压缩通道数为：                [8, 4, 6, 6,10,10, 20, 20, 20, 28, 28, 28, 48, 48, 48, 48]
    num_squeezed_channels_list_change = [8, 8, 8, 8, 8, 8, 16, 16, 16, 24, 24, 24, 48, 48, 48, 48]

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

    if using_ca:
        if ca_use_pre_training:
            path = ''
            model_ft = torch.load(path)  # 模型加载路径
            print("Using EfficientNet-CA pre-training")
        else:
            # 构建基线模型
            blocks_args, global_params = efficientnet(width_coefficient=1.0, depth_coefficient=1.0,
                                                      image_size=input_size, num_classes=class_num)
            model_ft = EfficientNet_CA(blocks_args=blocks_args, global_params=global_params)

            # 常规CA，将SE修改为CA
            for i in range(stage_B0):
                model_ft._blocks[i]._se_reduce = CoordAtt(channel_list_B0[i], channel_list_B0[i],
                                                          use_reduction=False, reduction=32, mip=num_squeezed_channels_list[i])
                # model_ft._blocks[i]._se_reduce = nn.Identity()

            print("Add CA to EfficientNet!")
    else:
        if baseline_using_pre_training:
            # 从本地文件中加载预训练模型
            model_ft = EfficientNet.from_pretrained(net_name)
            # model_ft = EfficientNet_Baseline.from_pretrained('efficientnet-b0')
            print("Using the EfficientNet baseline having pretraining!")

            # 重置SE的权重
            for i in range(stage_B0):
                model_ft._blocks[i]._se_reduce = SE(channel_list_B0[i], num_squeezed_channels_list[i], input_size)

            print("Reset the SE of EfficientNet!")
        else:
            # 构建基线模型
            blocks_args, global_params = efficientnet(width_coefficient=1.0, depth_coefficient=1.0,
                                                      image_size=input_size, num_classes=class_num)
            model_ft = EfficientNet_original(blocks_args=blocks_args, global_params=global_params)

            # 重置SE的权重
            # for i in range(stage_B0):
            #     model_ft._blocks[i]._se_reduce = SE(channel_list_B0[i], num_squeezed_channels_list[i], input_size)

            print("Using the EfficientNet baseline non-pretraining!")

    # 修改全连接层的层数
    num_ftrs = model_ft._fc.in_features
    model_ft._fc = nn.Linear(num_ftrs, class_num)
    # 修改Dropout层
    model_ft._dropout = nn.Dropout(0.2)

    # model_ft = torch.nn.DataParallel(model_ft, device_ids=[0, 1])

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
    # optimizer = optim.SGD((model_ft.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # 使用AdamW优化器
    optimizer = optim.AdamW(model_ft.parameters(), lr=lr, weight_decay=weight_decay)

    train_loss, best_model_wts, final_model_wts = train_model(model_ft, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs)

    # 测试
    print('-' * 25)
    print('Test Accuracy:')

    print('The best model test Accuracy:')
    # 权重替换
    model_ft.load_state_dict(best_model_wts)
    test_model(model_ft, criterion)

    print('The final model test Accuracy:')
    model_ft.load_state_dict(final_model_wts)
    test_model(model_ft, criterion)

else:
    # 修改GPU卡号
    torch.cuda.set_device(GPU)

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
