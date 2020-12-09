from .vgg import ConvNet4, ConvNet6, VGG9
from .resnet import resnet20, resnet32, resnet44, resnet56, resnet18, resnet34


def get_model(model_name, num_classes, img_size, do_perturb):
    kwargs = {"num_classes": num_classes, "img_size": img_size, "do_perturb": do_perturb}
    conv_channels = -1
    if "conv" in model_name:
        conv_channels = int(model_name.split("_")[-1])
        model_name = "_".join(model_name.split("_")[:-1])

    if model_name == "lenet":
        raise DeprecationWarning
    elif model_name == "conv4":
        return ConvNet4(conv_channels=conv_channels, **kwargs)
    elif model_name == "conv6":
        return ConvNet6(conv_channels=conv_channels, **kwargs)
    elif model_name == "vgg9":
        return VGG9(**kwargs)
    elif model_name == "resnet20":
        return resnet20(**kwargs)
    elif model_name == "resnet32":
        return resnet32(**kwargs)
    elif model_name == "resnet44":
        return resnet44(**kwargs)
    elif model_name == "resnet56":
        return resnet56(**kwargs)
    elif model_name == "resnet18":
        return resnet18(**kwargs)
    elif model_name == "resnet34":
        return resnet34(**kwargs)
    else:
        raise NotImplementedError
