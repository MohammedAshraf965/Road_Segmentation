import torch.nn as nn

from torchvision.models.segmentation import deeplabv3_resnet101, lraspp_mobilenet_v3_large

def model_init(num_classes=2):
    # Load the best pretrained weights from the API using DEFAULT for the weights
    # model = deeplabv3_resnet101(weights='DEFAULT')
    # model.classifier[4] = nn.Conv2d(256, num_classes, 1)
    # model.aux_classifier[4] = nn.Conv2d(256, num_classes, 1)

    model = lraspp_mobilenet_v3_large(weights='DEFAULT')
    model.classifier.low_classifier = nn.Conv2d(40, num_classes, kernel_size=(1,1), stride=(1,1))
    model.classifier.high_classifier = nn.Conv2d(128, num_classes, kernel_size=(1,1), stride=(1,1))
    
    return model
