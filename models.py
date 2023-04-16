from torch.autograd import Function
import torch.nn as nn
import torch
from torch import autograd
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


def create_model(args):
    if args.model == 'DANN':
        return DANN(310, args.hidden_dim, 4, 2, args.lamda)
    elif args.model == 'SADA':
        return SADA(310, args.hidden_dim, 3, 2, args.lamda, args.triplet_weight)
    elif args.model == 'MLP':
        return MLP(310, args.hidden_dim, 4)
    elif args.model == 'IRM':
        return IRM(310, args.hidden_dim, 3, args.penalty_weight)
    elif args.model == 'REx':
        return REx(310, args.hidden_dim, 3, args.variance_weight)
    elif args.model == 'WGANGen':
        return WGANGen(64, 310, 128)
    elif args.model == 'ADDA':
        return ADDA(310, args.hidden_dim, 3, 1, 10)
    else:
        raise ValueError("Unknown model type!")


# conventional deep learning models
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.criterion = nn.CrossEntropyLoss()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.label_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, num_labels)
        )

    def forward(self, input_data):
        feature_mapping = self.feature_extractor(input_data)
        class_output = self.label_classifier(feature_mapping)
        return class_output

    def compute_loss(self, data):
        x, y = data
        class_output = self.forward(x)
        loss = self.criterion(class_output, y)
        return loss


# domain generalization models
class IRM(MLP):
    def __init__(self, input_dim, hidden_dim, num_labels, penalty_weight):
        super(IRM, self).__init__(input_dim, hidden_dim, num_labels)
        self.penalty_weight = penalty_weight

    def penalty(self, logits, y):
        scale = torch.ones((1, self.num_labels)).to(y.device).requires_grad_()
        loss = self.criterion(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad ** 2)

    def compute_loss(self, data):
        x, y = data
        class_output = self.forward(x)
        loss = self.criterion(class_output, y)
        penalty = self.penalty(class_output, y)
        return loss + self.penalty_weight * penalty

class REx(MLP):
    def __init__(self, input_dim, hidden_dim, num_labels, variance_weight):
        super(REx, self).__init__(input_dim, hidden_dim, num_labels)
        self.variance_weight = variance_weight
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def compute_loss(self, data):
        x, y = data
        class_output = self.forward(x)
        loss = self.criterion(class_output, y)
        loss_mean = torch.mean(loss)
        loss_var = torch.var(loss)
        return loss_mean + self.variance_weight * loss_var


# domain adaptation models
class ReverseLayer(Function):
    @staticmethod
    def forward(ctx, x, lamda):
        ctx.lamda = lamda
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lamda
        return output, None

# class DANN(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_labels, num_domains, lamda):
#         super(DANN, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.num_labels = num_labels
#         self.num_domains = num_domains
#         self.lamda = lamda
#         self.criterion = nn.CrossEntropyLoss()
#
#         self.feature_extractor = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU()
#         )
#
#         self.label_classifier = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim//2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim//2, hidden_dim//2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim//2, num_labels)
#         )
#
#         self.domain_classifier = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim//2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim//2, hidden_dim//2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim//2, num_domains)
#         )
#
#     def forward(self, input_data):
#         feature_mapping = self.feature_extractor(input_data)
#         reverse_feature = ReverseLayer.apply(feature_mapping, self.lamda)
#         class_output = self.label_classifier(feature_mapping)
#         domain_output = self.domain_classifier(reverse_feature)
#         return class_output, domain_output
#
#     def compute_loss(self, source_data, target_data):
#         inputs, class_labels, domain_source_labels = source_data
#         pred_class_label, pred_domain_label = self.forward(inputs)
#         class_loss = self.criterion(pred_class_label, class_labels)
#         domain_source_loss = self.criterion(pred_domain_label, domain_source_labels)
#
#         inputs, domain_target_labels = target_data
#         _, pred_domain_label = self.forward(inputs)
#         domain_target_loss = self.criterion(pred_domain_label, domain_target_labels)
#
#         domain_loss = domain_source_loss + domain_target_loss
#         return class_loss, domain_loss


class DANN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels, num_domains, lamda, momentum=0.5):
        super().__init__()
        self.criterion = nn.NLLLoss()
        self.feature_extractor = nn.Sequential(
            nn.Linear(310, 256),
            nn.BatchNorm1d(num_features=256, momentum=momentum),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(num_features=128, momentum=momentum),
            nn.Dropout(),
            nn.ReLU()
        )

        self.class_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(num_features=64, momentum=momentum),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(num_features=32, momentum=momentum),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.LogSoftmax(dim=1)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.BatchNorm1d(32, momentum=momentum),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input_data, lamda=0):
        feature = self.feature_extractor(input_data)
        reverse_feature = ReverseLayer.apply(feature, lamda)
        class_pred = self.class_classifier(feature)
        domain_pred = self.domain_classifier(reverse_feature)
        return class_pred, domain_pred

    def compute_loss(self, source_data, target_data, lamda = 0.5):
        inputs, class_labels, domain_source_labels = source_data
        pred_class_label, pred_domain_label = self.forward(inputs, lamda)
        class_loss = self.criterion(pred_class_label, class_labels)
        domain_source_loss = self.criterion(pred_domain_label, domain_source_labels)

        inputs, domain_target_labels = target_data
        _, pred_domain_label = self.forward(inputs, lamda)
        domain_target_loss = self.criterion(pred_domain_label, domain_target_labels)

        domain_loss = domain_source_loss + domain_target_loss
        return class_loss, domain_loss


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.5)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out, negative_slope=0.5)
        return out


class ResNet18(nn.Module):

    def __init__(self, block, num_blocks, num_classes=4):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(5, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 4)

    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18_():
    return ResNet18(BasicBlock, [2, 2, 2, 2]).to('cuda')
