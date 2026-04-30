from torchvision.models import resnet101, resnet50


def res2net50_v1b_26w_4s(pretrained=True):
    return resnet50(weights=None)


def res2net50_v1b_14w_8s(pretrained=True):
    return resnet50(weights=None)


def res2net101_v1b_26w_4s(pretrained=True):
    return resnet101(weights=None)
