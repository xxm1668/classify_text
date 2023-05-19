import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, class_num, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_num = class_num

    def forward(self, predict, target):
        pt = F.softmax(predict, dim=1)  # softmmax获取预测概率
        class_mask = F.one_hot(target, self.class_num)  # 获取target的one hot编码
        ids = target.view(-1, 1)
        alpha = self.alpha[ids.data.view(-1)]  # 注意，这里的alpha是给定的一个list(tensor
        # ),里面的元素分别是每一个类的权重因子
        probs = (pt * class_mask).sum(1).view(-1, 1)  # 利用onehot作为mask，提取对应的pt
        log_p = probs.log()
        # 同样，原始ce上增加一个动态权重衰减因子

        loss = -alpha * (torch.pow((1 - probs.cpu()), self.gamma)) * log_p.cpu()

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


# 定义标签平滑损失函数
class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1 - smoothing

    def forward(self, x, target):
        target = target.unsqueeze(1)
        one_hot_target = torch.zeros_like(x)
        one_hot_target.scatter_(1, target, 1)
        one_hot_target = one_hot_target * self.confidence + (1 - one_hot_target) * self.smoothing / (
                self.num_classes - 1)
        logprobs = torch.nn.functional.log_softmax(x, dim=1)
        return -(one_hot_target * logprobs).sum(dim=1).mean()


# 定义Soft F1 Loss
class SoftF1Loss(torch.nn.Module):
    def __init__(self, num_classes):
        super(SoftF1Loss, self).__init__()
        self.num_classes = num_classes

    def forward(self, y_pred, y_true):
        # y_pred: (batch_size, num_classes)
        # y_true: (batch_size,)

        one_hot = torch.zeros_like(y_pred).scatter(1, y_true.unsqueeze(1), 1)
        # one_hot: (batch_size, num_classes)

        tp = (y_pred * one_hot).sum(dim=0)
        fp = ((1 - one_hot) * y_pred).sum(dim=0)
        fn = (one_hot * (1 - y_pred)).sum(dim=0)

        precision = tp / (tp + fp + 1e-16)
        recall = tp / (tp + fn + 1e-16)

        f1 = 2 * precision * recall / (precision + recall + 1e-16)
        f1 = torch.nan_to_num(f1)

        loss = 1 - f1.mean()

        return loss
