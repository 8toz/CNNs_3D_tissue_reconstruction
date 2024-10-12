import numpy as np


def iou_numpy(outputs: np.array, labels: np.array, smooth=1e-6):

    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + smooth) / (union + smooth)

    thresholded = np.ceil(np.clip(20 * (iou - 0.7), 0, 10)) / 10

    return thresholded.mean()

def unstack_labels(predictions, n_classes=5):
    aux = []
    for label in range(n_classes):
        mask = (predictions == label).astype(np.uint8)
        aux.append(mask)
    
    result = np.stack(aux, axis=-1)
    return result