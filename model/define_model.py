import timm


def load_pretrained_encoder(model_name="resnet18", num_classes=2):
    return timm.create_model(model_name, pretrained=True, num_classes=2)
