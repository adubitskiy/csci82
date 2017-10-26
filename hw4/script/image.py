import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray
import matplotlib.image as mpimg

def load_jpg(folder, file_name, w, h, grey = True):
    img = mpimg.imread(folder + '/' + file_name + '.jpg')
    img_resized = resize(img, (w, h), mode='reflect')
    if grey:
        img_gray = rgb2gray(img_resized)
        return img_gray.reshape(w * h).astype(np.float32)
    return img_resized.reshape(w * h * 3).astype(np.float32)

def load_train_gray_jpg_120(file_name):
    return load_jpg('train', file_name, 120, 120)

def load_train_color_224(file_name):
    return load_jpg('train', file_name, 224, 224, False)



