import torch

def simple_iou(box1, box2):
    """
    :param box1, box2: [x_center, y_center, width, height]
    :return: IoU value
    """

    b1_x1 = box1[0] - box1[2] / 2
    b1_x2 = box1[0] + box1[2] / 2
    b1_y1 = box1[1] - box1[3] / 2