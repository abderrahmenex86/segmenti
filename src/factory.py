import torch
import torch.nn as nn

from src.models import DeepLabV3, FocalDiceLoss, LinkNet, SegFormer, UNet


def build_model(model_type, num_classes, **kwargs):
    model_mapping = {
        "unet": lambda: UNet(in_channels=3, num_classes=num_classes, base_channels=kwargs.get("base_channels", 64)),
        "deeplabv3": lambda: DeepLabV3(num_classes=num_classes),
        "linknet": lambda: LinkNet(num_classes=num_classes),
        "segformer": lambda: SegFormer(num_classes=num_classes),
    }

    if model_type not in model_mapping:
        raise ValueError(f"Unknown model type: {model_type}. Select from {list(model_mapping.keys())}")

    return model_mapping[model_type]()


def build_optimizer(model_parameters, optimizer_type, learning_rate, weight_decay, **kwargs):
    optimizer_class = getattr(torch.optim, optimizer_type, None)
    if optimizer_class is None:
        raise ValueError(f"Unknown optimizer: {optimizer_type}. Check torch.optim spelling.")

    return optimizer_class(model_parameters, lr=learning_rate, weight_decay=weight_decay)


def build_scheduler(optimizer, scheduler_type, **kwargs):
    if scheduler_type == "none" or scheduler_type is None:
        return None

    scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_type, None)
    if scheduler_class is None:
        raise ValueError(f"Unknown scheduler: {scheduler_type}. Check torch.optim.lr_scheduler spelling.")

    if scheduler_type == "ReduceLROnPlateau":
        return scheduler_class(
            optimizer, mode="max", patience=kwargs.get("patience", 5) // 2, factor=kwargs.get("scheduler_factor", 0.5)
        )
    elif scheduler_type == "StepLR":
        return scheduler_class(
            optimizer, step_size=kwargs.get("scheduler_step_size", 10), gamma=kwargs.get("scheduler_gamma", 0.1)
        )
    elif scheduler_type == "CosineAnnealingLR":
        return scheduler_class(optimizer, T_max=kwargs.get("scheduler_t_max", kwargs.get("epochs", 50)))

    return scheduler_class(optimizer)


def build_pipeline(model_type, num_classes, optimizer_type, learning_rate, weight_decay, scheduler_type, **kwargs):
    model_instance = build_model(model_type, num_classes, **kwargs)
    loss_criterion = FocalDiceLoss(
        num_classes=num_classes, alpha=kwargs.get("focal_alpha", 0.25), gamma=kwargs.get("focal_gamma", 2.0)
    )
    optimizer_instance = build_optimizer(
        model_instance.parameters(), optimizer_type, learning_rate, weight_decay, **kwargs
    )
    scheduler_instance = build_scheduler(optimizer_instance, scheduler_type, **kwargs)

    return model_instance, loss_criterion, optimizer_instance, scheduler_instance
