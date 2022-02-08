import functools
import torch

from .cnn.densenet import DenseNet121, DenseNet161, DenseNet169, DenseNet201, densenet_cifar
from .cnn.dpn import DPN26, DPN92
from .cnn.efficientnet import EfficientNetB0
from .cnn.googlenet import GoogLeNet
from .cnn.lenet import LeNet
from .cnn.mobilenet import MobileNet
from .cnn.mobilenetv2 import MobileNetV2
from .cnn.pnasnet import PNASNetA, PNASNetB
from .cnn.preact_resnet import PreActResNet18, PreActResNet34, PreActResNet50, PreActResNet101, PreActResNet152
from .cnn.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .cnn.resnext import ResNeXt29_2x64d, ResNeXt29_4x64d, ResNeXt29_8x64d, ResNeXt29_32x4d
from .cnn.senet import SENet18
from .cnn.shufflenet import ShuffleNetG2, ShuffleNetG3
from .cnn.shufflenetv2 import ShuffleNetV2
from .cnn.vgg import VGG

from utils.dataset import DataLoader

def partialclass(cls, *args, **kwds):

    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwds)

    return NewCls

cnn_models = {
    'densenet121': functools.partial(DenseNet121, num_classes=num_classes),
    'densenet161': functools.partial(DenseNet161, num_classes=num_classes),
    'densenet169': functools.partial(DenseNet169, num_classes=num_classes),
    'densenet201': functools.partial(DenseNet201, num_classes=num_classes),
    'densenet_cifar': functools.partial(densenet_cifar, num_classes=num_classes),
    'dpn26': functools.partial(DPN26, num_classes=num_classes),
    'dpn92': functools.partial(DPN92, num_classes=num_classes),
    'efficientnetb0': functools.partial(EfficientNetB0, num_classes=num_classes),
    'googlenet': partialclass(GoogLeNet, num_classes=num_classes),
    'lenet': partialclass(LeNet, num_classes=num_classes),
    'mobilenet': partialclass(MobileNet, num_classes=num_classes),
    'mobilenetv2': partialclass(MobileNetV2, num_classes=num_classes),
    'pnasnet_a': functools.partial(PNASNetA, num_classes=num_classes),
    'pnasnet_b': functools.partial(PNASNetB, num_classes=num_classes),
    'preactresnet18': functools.partial(PreActResNet18, num_classes=num_classes),
    'preactresnet34': functools.partial(PreActResNet34, num_classes=num_classes),
    'preactresnet50': functools.partial(PreActResNet50, num_classes=num_classes),
    'preactresnet101': functools.partial(PreActResNet101, num_classes=num_classes),
    'preactresnet152': functools.partial(PreActResNet152, num_classes=num_classes),
    'resnet18': functools.partial(ResNet18, num_classes=num_classes),
    'resnet34': functools.partial(ResNet34, num_classes=num_classes),
    'resnet50': functools.partial(ResNet50, num_classes=num_classes),
    'resnet101': functools.partial(ResNet101, num_classes=num_classes),
    'resnet152': functools.partial(ResNet152, num_classes=num_classes),
    'resnext29_2x64d': functools.partial(ResNeXt29_2x64d, num_classes=num_classes),
    'resnext29_4x64d': functools.partial(ResNeXt29_4x64d, num_classes=num_classes),
    'resnext29_8x64d': functools.partial(ResNeXt29_8x64d, num_classes=num_classes),
    'resnext29_32x4d': functools.partial(ResNeXt29_32x4d, num_classes=num_classes),
    'senet18': functools.partial(SENet18, num_classes=num_classes),
    # 'shufflenetg2': ShuffleNetG2,
    # 'shufflenetg3': ShuffleNetG3,
    'shufflenetv2_0.5': functools.partial(ShuffleNetV2_05, num_classes=num_classes),
    'shufflenetv2_1.0': functools.partial(ShuffleNetV2_10, num_classes=num_classes),
    'shufflenetv2_1.5': functools.partial(ShuffleNetV2_15, num_classes=num_classes),
    'shufflenetv2_2.0': functools.partial(ShuffleNetV2_20, num_classes=num_classes),
    'vgg11': functools.partial(VGG_11, num_classes=num_classes),
    'vgg13': functools.partial(VGG_13, num_classes=num_classes),
    'vgg16': functools.partial(VGG_16, num_classes=num_classes),
    'vgg19': functools.partial(VGG_19, num_classes=num_classes)
}

losses = {
    "cross-entropy": torch.nn.CrossEntropyLoss
}


def get_pytorch_params(model_name: str = "", loss_name: str = "") -> tuple:
    return models.get(model_name, None), losses.get(loss_name, None)


def train(model, loader: DataLoader, criterion, optimizer, device: str = "cpu"):
    running_loss = 0.0
    for data, labels in loader:
        # get the inputs; data is a list of [inputs, labels]
        data = data.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss


def test(model, loader: DataLoader, device: str = "cpu"):
    correct = 0
    total= 0
    
    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    return acc

def train_model(model, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 10, device: str = "cpu", batch_size: int = 1):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model = model.to(device)
    losses, accs = [], []
    for epoch in range(epochs):
        loss = train(model=model, loader=train_loader, criterion=criterion, optimizer=optimizer, device=device)
        acc = test(model=model, loader=val_loader, device=device)
        print(f"[{epoch+1}/{epochs:5d}] loss: {loss/batch_size:.4f}; val_acc: {acc:.4f}")
        losses.append(loss)
        accs.append(acc)
    return losses, accs
