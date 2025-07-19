from skimage.io import imread
import numpy as np

def drawBoundaries(imgname,labels):
    """
    Draw boundaries on the input image based on the input labels.
    Args:
        img: Input image.
        labels: Label map.
        numlabels: Number of unique labels.
    Returns:
        Image with boundaries drawn.
    """
    img = imread(imgname)
    img = np.array(img)
    
    ht, wd = labels.shape

    for y in range(1, ht-1):
        for x in range(1, wd-1):
            if labels[y, x-1] != labels[y, x+1] or labels[y-1, x] != labels[y+1, x]:
                img[y, x, :] = 0

    return img

