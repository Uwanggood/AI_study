import torch.nn as nn
import torch.utils.model_zoo as model_zoo

#https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py 참고
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # shape is 55 x 55 x 64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # shape is 27 x 27 x 64
            nn.Conv2d(64, 192, kernel_size=5, padding=2),  # shape is 27 x 27 x 192
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # shape 13 * 13 * 192
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # shape 13 * 13 * 384
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # shape 13 * 13 * 256
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # shape 13 * 13 * 256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # shape 6 * 6 * 256
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),  # shape 4096
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),  # shape 4096
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, **kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'))
    return model
