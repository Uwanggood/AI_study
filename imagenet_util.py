import torchvision.transforms.functional as F
import numpy as np


def preprocess_image_with_rgb(datas):
    x_train = []
    for i in range(len(datas)):
        image = datas[i].unsqueeze(0).unsqueeze(0).float()
        image = F.resize(image, [224, 224], antialias=True)
        image = image.squeeze(0).repeat(3, 1, 1)
        image = image.permute(1, 2, 0).numpy()
        x_train.append(image)
    return np.array(x_train)
