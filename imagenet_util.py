import tensorflow as tf
from skimage.color import gray2rgb
import numpy as np


def preprocess_image_with_rgb(datas):
    x_train = []
    for i in range(len(datas)):
        image = tf.constant(datas[i])
        image = image[tf.newaxis, ..., tf.newaxis]
        image = tf.image.resize_with_pad(image, target_width=224, target_height=224)[0, ..., 0].numpy()
        grey_image = gray2rgb(image)
        x_train.append(grey_image)
    return np.array(x_train)
