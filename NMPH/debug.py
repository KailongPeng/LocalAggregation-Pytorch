'''
Pre-activation ResNet in PyTorch.
Adapted from https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

os.chdir('/gpfs/milgram/project/turk-browne/projects/LocalAggregation-Pytorch')
BATCH_NORM_MOMENTUM = 0.003
Kailong = True


def batch_norm(num_features):
    return nn.BatchNorm2d(num_features, momentum=BATCH_NORM_MOMENTUM)


# create layer norm
def layer_norm(normalized_shape):
    return nn.LayerNorm(normalized_shape)
# def layer_norm(channels):
#     # Assuming channels is an integer representing the number of channels
#     return nn.LayerNorm((channels,))


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, enable_shortcut=False, config=None):
        super(PreActBlock, self).__init__()
        if config.layer_norm:
            self.norm1 = layer_norm((in_planes,))
        else:
            self.norm1 = batch_norm(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        if stride != 1 or in_planes != self.expansion * planes or enable_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = batch_norm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        print(f"50: x.shape={x.shape}")
        out = self.relu1(self.norm1(x))
        print(f"52: x.shape={x.shape}")
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x

        out = self.conv1(out)
        print(f"56: x.shape={x.shape}")
        out = self.conv2(self.relu2(self.norm2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, config=None):
        super(PreActBottleneck, self).__init__()
        if config.layer_norm:
            self.norm1 = layer_norm((in_planes,))
        else:
            self.norm1 = batch_norm(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        if config.layer_norm:
            self.norm2 = layer_norm((planes,))
        else:
            self.norm2 = batch_norm(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if config.layer_norm:
            self.norm3 = layer_norm((planes,))
        else:
            self.norm3 = batch_norm(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.norm1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.norm2(out)))
        out = self.conv3(F.relu(self.norm3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=128, input_channels=3, config=None):
        super(PreActResNet, self).__init__()
        self.features_secondLastLayer = None
        self.features_lastLayer = None
        self.in_planes = 64

        # (input channel # = 3, output channel # = 64)
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.padding = torch.nn.ConstantPad2d((0, 1, 0, 1), 0.)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.config = config
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)  # conv2_x
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # conv3_x
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # conv4_x
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # conv5_x

        if self.config.layer_norm:
            self.norm = layer_norm((512,))
        else:
            self.norm = batch_norm(512)
        self.relu = nn.ReLU(inplace=True)

        self.linear = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        first_flag = True
        for stride in strides:
            if first_flag:
                layers.append(block(self.in_planes, planes, stride, enable_shortcut=True, config=self.config))
                first_flag = False
            else:
                layers.append(block(self.in_planes, planes, stride, config=self.config))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.padding(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.norm(out)
        out = self.relu(out)

        # follows the TF implementation to replace avgpool with mean
        # Ref: https://github.com/neuroailab/LocalAggregation
        out = torch.mean(out, dim=(2, 3))
        if Kailong:
            self.features_secondLastLayer = out.detach().cpu().numpy()
        out = self.linear(out)  # out.shape = torch.Size([9, 128])
        if Kailong:
            self.features_lastLayer = out.detach().cpu().numpy()
        return out


# defining 5 types of PreActResNet, with different number of layers: 18, 34, 50, 101, 152
def PreActResNet18(num_classes=128, config=None):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes, config=config)


def PreActResNet34(num_classes=128, config=None):
    return PreActResNet(PreActBlock, [3, 4, 6, 3], num_classes, config=config)


def PreActResNet50(num_classes=128, config=None):
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3], num_classes, config=config)


def PreActResNet101(num_classes=128, config=None):
    return PreActResNet(PreActBottleneck, [3, 4, 23, 3], num_classes, config=config)


def PreActResNet152(num_classes=128, config=None):
    return PreActResNet(PreActBottleneck, [3, 8, 36, 3], num_classes, config=config)


import os
import sys
import copy
import json
import logging
import numpy as np
from tqdm import tqdm
from itertools import product, chain
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torchvision.transforms.functional as tf
import torch.backends.cudnn as cudnn

from src.utils.utils import \
    save_checkpoint as save_snapshot, \
    copy_checkpoint as copy_snapshot, \
    AverageMeter, adjust_learning_rate, exclude_bn_weight_bias_from_weight_decay
from src.utils.setup import print_cuda_statistics
from src.models.preact_resnet import PreActResNet18
from src.datasets.imagenet import ImageNet
from src.objectives.localagg import LocalAggregationLossModule, MemoryBank, Kmeans
from src.objectives.instance import InstanceDiscriminationLossModule
from src.utils.tensor import l2_normalize

import time
from termcolor import colored
from src.utils.tensor import repeat_1d_tensor
import pdb


class BaseAgent(object):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Agent")

        self._set_seed()  # set seed as early as possible

        self._load_datasets()
        self.train_loader, self.train_len = self._create_dataloader(self.train_dataset, shuffle=True)
        self.val_loader, self.val_len = self._create_dataloader(self.val_dataset, shuffle=False)

        self._choose_device()
        self._create_model()
        self._create_optimizer()

        self.current_epoch = 0
        self.current_iteration = 0
        self.current_val_iteration = 0

        # we need these to decide best loss
        self.current_loss = 0
        self.current_val_metric = 0
        self.best_val_metric = 0
        self.iter_with_no_improv = 0

        try:  # hack to handle different versions of TensorboardX
            self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir)
        except:
            self.summary_writer = SummaryWriter(logdir=self.config.summary_dir)

    def _set_seed(self):
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

    def _choose_device(self):
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda
        self.manual_seed = self.config.seed
        if self.cuda: torch.cuda.manual_seed(self.manual_seed)

        if self.cuda:
            if not isinstance(self.config.gpu_device, list):
                self.config.gpu_device = [self.config.gpu_device]
            num_gpus = len(self.config.gpu_device)
            self.multigpu = num_gpus > 1 and torch.cuda.device_count() > 1

            if not self.multigpu:  # e.g. just 1 GPU
                gpu_device = self.config.gpu_device[0]
                self.logger.info("User specified 1 GPU: {}".format(gpu_device))
                self.device = torch.device("cuda")
                torch.cuda.set_device(gpu_device)
            else:
                gpu_devices = ','.join([str(_gpu_id) for _gpu_id in self.config.gpu_device])
                self.logger.info("User specified {} GPUs: {}".format(
                    num_gpus, gpu_devices))
                self.device = torch.device("cuda")

            self.gpu_devices = self.config.gpu_device
            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

    def _load_datasets(self):
        raise NotImplementedError

    def _create_dataloader(self, dataset, shuffle=True):
        dataset_size = len(dataset)
        loader = DataLoader(dataset, batch_size=self.config.optim_params.batch_size,
                            shuffle=shuffle, pin_memory=True,
                            num_workers=self.config.data_loader_workers)

        return loader, dataset_size

    def _create_model(self):
        raise NotImplementedError

    def _create_optimizer(self):
        raise NotImplementedError

    def load_checkpoint(self, filename):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        raise NotImplementedError

    def save_checkpoint(self, filename="checkpoint.pth.tar", is_best=False):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        raise NotImplementedError

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.train()
            self.cleanup()
        except KeyboardInterrupt as e:
            self.logger.info("Interrupt detected. Saving data...")
            self.backup()
            self.cleanup()
            raise e

    def train(self):
        """
        Main training loop
        :return:
        """
        for epoch in range(self.current_epoch, self.config.num_epochs):
            adjust_learning_rate(epoch=epoch, opt_params=self.config.optim_params, optimizer=self.optim)
            self.current_epoch = epoch
            self.train_one_epoch()
            if (self.config.validate and
                    epoch % self.config.optim_params.validate_freq == 0):
                self.validate()  # validate every now and then
            self.save_checkpoint()

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        raise NotImplementedError

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        raise NotImplementedError

    def backup(self):
        """
        Backs up the model upon interrupt
        """
        self.summary_writer.export_scalars_to_json(os.path.join(self.config.summary_dir, "all_scalars.json".format()))
        self.summary_writer.close()

        self.logger.info("Backing up current version of model...")
        self.save_checkpoint(filename='backup.pth.tar')

    def finalise(self):
        """
        Do appropriate saving after model is finished training
        """
        self.summary_writer.export_scalars_to_json(os.path.join(self.config.summary_dir, "all_scalars.json".format()))
        self.summary_writer.close()

        self.logger.info("Saving final versions of model...")
        self.save_checkpoint(filename='final.pth.tar')

    def cleanup(self):
        """
        Undo any global changes that the Agent may have made
        """
        if self.multigpu:
            del os.environ['CUDA_VISIBLE_DEVICES']


class ImageNetFineTuneAgent(BaseAgent):
    """
    Given a pretrained ResNet module, we take the layer
    prior to the final pooling (512, 7, 7) size and learn
    a linear layer on top.

    @param config: DotMap
                   configuration settings
    """

    def __init__(self, config):
        super(ImageNetFineTuneAgent, self).__init__(config)
        self.config = config

        # self.resnet = nn.Sequential(*list(self.resnet.module.children())[:-1])
        # self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # # freeze all of these parameters and only learn the last layer
        # self.resnet = self.resnet.eval()
        # for param in self.resnet.parameters():
        #     param.requires_grad = False

        self.val_acc = []
        self.train_loss = []
        self.train_extra = []

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        cudnn.benchmark = True

    def _load_image_transforms(self):
        image_size = self.config.data_params.image_size
        train_transforms = transforms.Compose([
            # these are borrowed from
            # https://github.com/zhirongw/lemniscate.pytorch/blob/master/main.py
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        if self.config.data_params.ten_crop:
            test_transforms = transforms.Compose([
                transforms.Resize(256),  # FIXME: hardcoded for 224 image size
                transforms.TenCrop(image_size),
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])(crop) for crop in crops])),
            ])
        else:
            test_transforms = transforms.Compose([
                transforms.Resize(256),  # FIXME: hardcoded for 224 image size
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        return train_transforms, test_transforms

    def _load_datasets(self):
        train_transforms, test_transforms = self._load_image_transforms()

        train_dataset = ImageNet(train=True, image_transforms=train_transforms)
        val_dataset = ImageNet(train=False, image_transforms=test_transforms)

        self.config.data_params.n_channels = 3
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    # def _create_model(self):
    #     assert self.config.data_params.image_size == 224
    #     model = nn.Linear(512 * 7 * 7, 128)
    #     model = model.to(self.device)
    #     if self.multigpu:
    #         model = nn.DataParallel(model)
    #     self.model = model

    def _create_model(self):
        if self.config.model_params.resnet_version == 'preact-resnet18':
            # model = PreActResNet18()
            self.model = PreActResNet18(num_classes=self.config.model_params.out_dim, config=self.config)
        else:
            raise NotImplementedError

        self.model = self.model.cuda()
        if self.multigpu:
            self.model = nn.DataParallel(self.model)

        load_pretrained_model = False
        if load_pretrained_model:
            filename = os.path.join(self.config.trained_agent_exp_dir, 'checkpoints', 'model_best.pth.tar')
            checkpoint = torch.load(filename, map_location='cpu')
            model_state_dict = checkpoint['model_state_dict']
            self.model.load_state_dict(model_state_dict)

    def _set_models_to_eval(self):
        self.model = self.model.eval()

    def _set_models_to_train(self):
        self.model = self.model.train()

    def _create_optimizer(self):
        self.optim = torch.optim.SGD(self.model.parameters(),
                                     lr=self.config.optim_params.learning_rate,
                                     momentum=self.config.optim_params.momentum,
                                     weight_decay=self.config.optim_params.weight_decay)

    def train_one_epoch(self):
        Kailong = True
        num_batches = self.train_len // self.config.optim_params.batch_size
        tqdm_batch = tqdm(total=num_batches,
                          desc="[Epoch {}, lr {}]".format(self.current_epoch, self.optim.param_groups[0]['lr']))

        self._set_models_to_train()  # turn on train mode
        epoch_loss = AverageMeter()

        for batch_i, (_, images, labels) in enumerate(self.train_loader):
            batch_size = images.size(0)

            # cast elements to CUDA
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits = self.model(images)

            if Kailong:
                # get the weights of the model
                weights_previous = self.model.module.linear.weight.data.clone().to(self.device)
                # print shape of weights_previous
                print(f'weights_previous.shape = {weights_previous.shape}')
                # self.model.module.linear.weight.data.shape = torch.Size([128, 512])  (#output channel, #input channel)
                # weights_previous.shape = torch.Size([128, 512])

            # with torch.no_grad():
            #     embeddings = self.resnet(images)
            #     embeddings = embeddings.view(batch_size, -1)
            #
            # logits = self.model(embeddings)
            print(f"logits.shape = {logits.shape}")  # logits.shape = torch.Size([9, 128])
            print(f"labels.shape = {labels.shape}")  # labels.shape = torch.Size([9])
            print(f"logits[:2, :] = {logits[:2, :]}")
            print(f"labels[:2] = {labels[:2]}")
            loss = self.criterion(logits, labels)  # self.criterion = nn.CrossEntropyLoss().to(self.device)  # Compute the loss using CrossEntropyLoss

            self.optim.zero_grad()
            loss.backward()  # Backpropagate the gradients
            self.optim.step()  # Update the model parameters using the optimizer

            # Update metrics and monitoring
            epoch_loss.update(loss.item(), batch_size)  # Update the average loss
            tqdm_batch.set_postfix({"Loss": epoch_loss.avg})  # Update the progress bar

            self.summary_writer.add_scalars("epoch/loss", {'loss': epoch_loss.val},
                                            self.current_iteration)  # Write the loss to TensorBoard
            self.train_loss.append(epoch_loss.val)  # Save the loss
            self.current_iteration += 1  # Update the iteration count
            tqdm_batch.update()  # Update the progress bar

            if Kailong:
                # get the weights of the model
                weights_current = self.model.module.linear.weight.data.clone()
                # print shape of weights_current
                print(f'weights_current.shape = {weights_current.shape}')

                # compute the difference between the weights
                weights_difference = weights_current - weights_previous
                weights_previous = weights_current.clone()

                # print shape of weights_difference
                print(f'weights_difference.shape = {weights_difference.shape}')

                # print __dir__
                # print(f"self.model.module.__dir__()={self.model.module.__dir__()}")
                # self.__dir__()
                #   ['config', 'logger', 'train_dataset', 'val_dataset', 'train_ordered_labels', 'train_loader', 'train_len', 'val_loader', 'val_len', 'is_cuda', 'cuda', 'manual_seed', 'multigpu', 'device', 'gpu_devices', 'model', 'optim', 'current_epoch', 'current_iteration', 'current_val_iteration', 'current_loss', 'current_val_metric', 'best_val_metric', 'iter_with_no_improv', 'summary_writer', 'memory_bank', 'cluster_labels', 'loss_fn', 'km', 'parallel_helper_idxs', 'val_acc', 'train_loss', 'train_extra', 'first_iteration_kmeans', '__module__', '__init__', '_init_memory_bank', 'load_memory_bank', '_load_memory_bank', 'get_memory_bank', '_get_memory_bank', '_get_loss_func', '_init_cluster_labels', '_init_loss_function', '_load_image_transforms', '_load_datasets', '_create_model', '_set_models_to_eval', '_set_models_to_train', '_create_optimizer', 'train_one_epoch', 'validate', 'load_checkpoint', 'copy_checkpoint', 'save_checkpoint', '__doc__', '_set_seed', '_choose_device', '_create_dataloader', 'run', 'train', 'backup', 'finalise', 'cleanup', '__dict__', '__weakref__', '__repr__', '__hash__', '__str__', '__getattribute__', '__setattr__', '__delattr__', '__lt__', '__le__', '__eq__', '__ne__', '__gt__', '__ge__', '__new__', '__reduce_ex__', '__reduce__', '__subclasshook__', '__init_subclass__', '__format__', '__sizeof__', '__dir__', '__class__']

                # save the activation of the last layer
                activation_lastLayer = self.model.module.features_lastLayer
                print(
                    f'activation_lastLayer.shape = {activation_lastLayer.shape}')  # activation_lastLayer.shape = (9, 128)

                # save the activation of the second last layer
                activation_secondLastLayer = self.model.module.features_secondLastLayer  # self.model.module.layer4[1].relu2
                print(
                    f'activation_secondLastLayer.shape = {activation_secondLastLayer.shape}')  # activation_secondLastLayer.shape = (9, 512)

                # create a folder to save the weights_difference
                weights_difference_folder = f'/gpfs/milgram/scratch60/turk-browne/kp578/LocalAgg/{self.config.exp_name_kailong}/weights_difference/numpy/'
                if not os.path.exists(weights_difference_folder):
                    os.makedirs(weights_difference_folder)
                np.save(f'{weights_difference_folder}/weights_difference_epoch{self.current_epoch}_batch_i{batch_i}.npy',
                        weights_difference.cpu().numpy())
                np.save(f'{weights_difference_folder}/activation_lastLayer_epoch{self.current_epoch}_batch_i{batch_i}.npy',
                        np.asarray(activation_lastLayer))
                np.save(f'{weights_difference_folder}/activation_secondLastLayer_epoch{self.current_epoch}_batch_i{batch_i}.npy',
                        np.asarray(activation_secondLastLayer))
                print(colored(
                    f'weights_difference saved to {weights_difference_folder}weights_difference_epoch{self.current_epoch}_batch_i{batch_i}.pth.tar',
                    'red'))

        self.current_loss = epoch_loss.avg
        tqdm_batch.close()

    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def validate(self):
        num_batches = self.val_len // self.config.optim_params.batch_size
        tqdm_batch = tqdm(total=num_batches, desc="[Val]")

        self._set_models_to_eval()
        num_correct = 0.
        num_total = 0.
        top1 = AverageMeter()

        with torch.no_grad():
            for _, images, labels in self.val_loader:
                batch_size = images.size(0)

                if self.config.data_params.ten_crop:
                    _, ncrops, c, h, w = images.size()
                    images = images.view(-1, c, h, w)

                images = images.to(self.device)
                labels = labels.to(self.device)

                # embeddings = self.resnet(images)
                #
                # if self.config.data_params.ten_crop:
                #     embeddings = embeddings.view(batch_size, ncrops, -1).mean(1)
                # else:
                #     embeddings = embeddings.view(batch_size, -1)
                #
                # logits = self.model(embeddings)
                logits = self.model(images)
                acc = self.accuracy(logits, labels)[0]
                top1.update(acc.item(), batch_size)

                tqdm_batch.set_postfix({"Val Accuracy": top1.avg / 100.0})
                tqdm_batch.update()

        self.summary_writer.add_scalars("Val/accuracy",
                                        {'accuracy': top1.avg / 100.0},
                                        self.current_val_iteration)

        self.current_val_iteration += 1
        self.current_val_metric = top1.avg / 100.0

        # save if this was the best validation accuracy
        # (important for model checkpointing)
        if self.current_val_metric >= self.best_val_metric:  # NOTE: >= for accuracy
            self.best_val_metric = self.current_val_metric

        tqdm_batch.close()

        # store the validation metric from every epoch
        self.val_acc.append(self.current_val_metric)

        return self.current_val_metric

    def load_checkpoint(self, filename, checkpoint_dir=None,
                        load_model=True, load_optim=True, load_epoch=True):
        checkpoint_dir = checkpoint_dir or self.config.checkpoint_dir
        filename = os.path.join(checkpoint_dir, filename)
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location='cpu')

            if load_epoch:
                self.current_epoch = checkpoint['epoch']
                self.current_iteration = checkpoint['iteration']
                self.current_val_iteration = checkpoint['val_iteration']

            if load_model:
                model_state_dict = checkpoint['model_state_dict']
                self.model.load_state_dict(model_state_dict)

                resnet_state_dict = checkpoint['resnet_state_dict']
                self.model.load_state_dict(resnet_state_dict)

            if load_optim:
                optim_state_dict = checkpoint['optim_state_dict']
                self.optim.load_state_dict(optim_state_dict)

                if not self.optim.param_groups[0]['lr'] == self.config.optim_params.learning_rate:
                    self.logger.info('Change optim lr from %f to %f' % (self.optim.param_groups[0]['lr'],
                                                                        self.config.optim_params.learning_rate))
                    for param_group in self.optim.param_groups:
                        param_group['lr'] = self.config.optim_params.learning_rate

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {}) "
                             "with val acc = {}\n"
                             .format(filename, checkpoint['epoch'], checkpoint['iteration'], checkpoint['val_acc']))

        except OSError as e:
            self.logger.info("Checkpoint doesnt exists: [{}]".format(filename))
            raise e

    def save_checkpoint(self, filename="checkpoint.pth.tar"):
        out_dict = {
            'resnet_state_dict': self.model.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'train_loss': np.array(self.train_loss),
            'loss': self.current_loss,
            'val_iteration': self.current_val_iteration,
            'val_metric': self.current_val_metric,
            'val_acc': np.array(self.val_acc),
            'config': self.config,
        }

        # if we aren't validating, then every time we save is the
        # best new epoch!
        is_best = ((self.current_val_metric == self.best_val_metric) or
                   not self.config.validate)
        save_snapshot(out_dict, is_best, filename=filename,
                      folder=self.config.checkpoint_dir)
        # self.copy_checkpoint()

    def copy_checkpoint(self, filename="checkpoint.pth.tar"):
        if self.current_epoch % self.config.copy_checkpoint_freq == 0:
            copy_snapshot(
                filename=filename, folder=self.config.checkpoint_dir,
                copyname='checkpoint_epoch{}.pth.tar'.format(self.current_epoch),
            )


from src.utils.setup import process_config


config_path = "/gpfs/milgram/project/turk-browne/projects/LocalAggregation-Pytorch/config/imagenet_ft_layerNorm.json"
config = process_config(config_path)
# ImageNetFineTuneAgent_ = ImageNetFineTuneAgent(config)
#
#
# ImageNetFineTuneAgent_.run()  # ImageNetFineTuneAgent_.train_one_epoch()
#
model = PreActResNet18(num_classes=128, config=config)
