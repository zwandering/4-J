from torch.autograd import Function
import torch.nn as nn
import torch
from torch import autograd
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


def create_model(args):
    if args.model == 'DANN':
        return DANN(310, args.hidden_dim, 4, 2, args.lamda, momentum=0.5)
    elif args.model == 'MLP':
        return MLP(310, args.hidden_dim, 4)
    elif args.model == 'tca':
        return MLP(50, args.hidden_dim, 4)
    elif args.model == 'IRM':
        return IRM(310, args.hidden_dim, 4, args.penalty_weight)
    elif args.model == 'ADDA':
        return ADDA(310, args.hidden_dim, 4, 1, 10)
    else:
        raise ValueError("Unknown model type!")


# conventional deep learning models
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels, momentum=0.5):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.criterion = nn.CrossEntropyLoss()

        self.feature_extractor = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.BatchNorm1d(num_features=256, momentum=momentum),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(num_features=128, momentum=momentum),
            nn.Dropout(),
            nn.ReLU()
        )

        self.label_classifier = nn.Sequential(
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

    def forward(self, input_data, lamda=0):
        feature_mapping = self.feature_extractor(input_data)
        class_output = self.label_classifier(feature_mapping)
        return class_output, None

    def compute_loss(self, data):
        x, y = data
        class_output, _ = self.forward(x)
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
        class_output, _ = self.forward(x)
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


class DANN_resnet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels, num_domains, lamda, momentum=0.5):
        super().__init__()
        self.criterion = nn.NLLLoss()
        self.feature_extractor = ResNet18_feature_extractor(BasicBlock, [2, 2, 2, 2])

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


class ADDA:
    def __init__(self, input_dim, hidden_dim, num_labels, num_domains, lamda):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.num_domains = num_domains
        self.lamda = lamda
        self.criterion = nn.CrossEntropyLoss()

        self.srcMapper = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, input_dim),
        )

        self.tgtMapper = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, input_dim),
        )

        self.Classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, num_labels)
        )

        self.Discriminator = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_domains)
        )

    def pretrain_forward(self, input_data):
        src_mapping = self.srcMapper(input_data)
        src_class = self.Classifier(src_mapping)
        return src_class

    def pretrain_loss(self, train_data):
        x, y = train_data
        src_class = self.pretrain_forward(x)
        loss = self.criterion(src_class, y)
        return loss

    def discriminator_loss(self, src_x, tgt_x):
        src_mapping = self.srcMapper(src_x)
        tgt_mapping = self.tgtMapper(tgt_x)
        batch_size = src_mapping.size(0)
        alpha = torch.rand(batch_size, 1).to(src_mapping.device)
        alpha = alpha.expand(src_mapping.size())
        interpolates = alpha * src_mapping + (1 - alpha) * tgt_mapping
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = self.Discriminator(interpolates)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(interpolates.device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        pred_tgt, pred_src = self.Discriminator(tgt_mapping), self.Discriminator(src_mapping)
        loss_discriminator = pred_tgt.mean() - pred_src.mean() + self.lamda * gradient_penalty
        return loss_discriminator

    def tgt_loss(self, tgt_x):
        tgt_mapping = self.tgtMapper(tgt_x)
        pred_tgt = self.Discriminator(tgt_mapping)
        loss_tgt = -pred_tgt.mean()
        return loss_tgt


class FeatureExtractor(nn.Module):

    def __init__(self, track_running_stats, momentum):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(310, 256),
            nn.BatchNorm1d(256, track_running_stats=track_running_stats, momentum=momentum),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128, track_running_stats=track_running_stats, momentum=momentum),
            nn.Dropout(),
            nn.ReLU()
        )

    def forward(self, input_data):
        return self.net(input_data)


class Classifier(nn.Module):

    def __init__(self, track_running_stats, momentum):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64, track_running_stats=track_running_stats, momentum=momentum),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32, track_running_stats=track_running_stats, momentum=momentum),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, input_data):
        return self.net(input_data)


class Discriminator(nn.Module):

    def __init__(self, track_running_stats, momentum):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 32),
            nn.BatchNorm1d(32, track_running_stats=track_running_stats, momentum=momentum),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, input_data):
        return self.net(input_data)


class SADA(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels, num_domains, lamda, triplet_weight, momentum=0.5):
        super(SADA, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.num_domains = num_domains
        self.lamda = lamda
        self.triplet_weight = triplet_weight
        self.criterion = nn.CrossEntropyLoss()

        self.feature_extractor = nn.Sequential(
            nn.Linear(310, 256),
            nn.BatchNorm1d(num_features=256, momentum=momentum),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(num_features=128, momentum=momentum),
            nn.Dropout(),
            nn.ReLU()
        )

        self.label_classifier = nn.Sequential(
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

    def forward(self, input_data, lamda):
        _, class_output, domain_output = self.record_forward(input_data, lamda)
        return class_output, domain_output

    def record_forward(self, input_data, lamda):
        feature_mapping = self.feature_extractor(input_data)
        reverse_feature = ReverseLayer.apply(feature_mapping, lamda)
        class_output = self.label_classifier(feature_mapping)
        domain_output = self.domain_classifier(reverse_feature)
        return feature_mapping, class_output, domain_output

    def separability_loss(self, labels, latents, imbalance_parameter=1):
        criteria = nn.modules.loss.CosineEmbeddingLoss()
        loss_up = 0
        one_cuda = torch.ones(1).cuda()
        mean = torch.mean(latents, dim=0).cuda().view(1, -1)
        loss_down = 0
        for i in range(self.num_labels):
            indecies = labels.eq(i)
            mean_i = torch.mean(latents[indecies], dim=0).view(1, -1)
            if str(mean_i.norm().item()) != 'nan':
                for latent in latents[indecies]:
                    loss_up += criteria(latent.view(1, -1), mean_i, one_cuda)
                loss_down += criteria(mean, mean_i, one_cuda)
        loss = (loss_up / loss_down) * imbalance_parameter
        return loss

    def pseudo_labeling(self, pred_class_label, m=.8):
        pred_class_label = F.softmax(pred_class_label, dim=1)
        pred_class_prob, pred_class_label = torch.max(pred_class_label, dim=1)
        indices = pred_class_prob > m
        pseudo_label = pred_class_label[indices]
        _, counts = np.unique(pseudo_label.cpu().numpy(), return_counts=True)
        if counts.shape[0] == 0:
            return False
        else:
            mi = np.min(counts)
            if len(counts) < 10:
                mi = 0
            ma = np.max(counts)
            return indices, pseudo_label, (mi + 1) / (ma + 1)

    def compute_loss(self, source_data, target_data, lamda):
        source_inputs, source_labels, domain_source_labels = source_data
        target_inputs, domain_target_labels = target_data

        latent_source, pred_class_label, pred_domain_label = self.record_forward(source_inputs, lamda)
        source_class_loss = self.criterion(pred_class_label, source_labels).mean()
        source_entropy = Categorical(logits=pred_class_label).entropy()
        source_domain_loss = (
                    (torch.ones_like(source_entropy) + source_entropy.detach() / self.num_labels) * self.criterion(
                pred_domain_label, domain_source_labels)).mean()

        latent_target, pred_class_label, pred_domain_label = self.record_forward(target_inputs, lamda)
        target_entropy = Categorical(logits=pred_class_label).entropy()
        target_domain_loss = ((torch.ones_like(target_entropy) + target_entropy.detach() / self.num_labels) * self.criterion(pred_domain_label, domain_target_labels)).mean()

        sep_loss = 0.
        data = self.pseudo_labeling(pred_class_label)
        if data:
            indices, pseudo_labels, imbalance_parameter = data
            latent_target = latent_target[indices, :]
            sep_loss = self.separability_loss(torch.cat((source_labels, pseudo_labels)),
                                              torch.cat((latent_source, latent_target)),
                                              imbalance_parameter)

        return source_class_loss, (source_domain_loss + target_domain_loss) * self.lamda +  sep_loss * self.triplet_weight


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


class ResNet18_feature_extractor(nn.Module):

    def __init__(self, block, num_blocks, num_classes=4):
        super(ResNet18_feature_extractor, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(5, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 128)

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
