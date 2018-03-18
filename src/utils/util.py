import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from PIL import Image
from chainer import datasets

width, height = 160, 160
IMG_MEAN = 'build/imageMean/image_mean.npy'

"""平均画像が未計算なら変換をかまさないバージョンの学習用データセットで平均を計算
Returns:
    [type] -- [description]
"""
def mean(d) :
    print("utils.mean called.")
    if not os.path.exists(IMG_MEAN):
        t, _ = datasets.split_dataset_random(d, int(len(d) * 0.8), seed=0)
        mean = np.zeros((3, height, width))
        for img, _ in tqdm_notebook(t, desc='Calc mean'):
            img = resize(img[:3].astype(np.uint8))
            mean += img
        mean = mean / float(len(d))
        np.save(IMG_MEAN, mean)
    else:
        mean = np.load(IMG_MEAN)
    # 平均画像の表示
    # plt.imshow(mean.transpose(1, 2, 0) / 255)
    # plt.show()
    mean = mean.mean(axis=(1, 2))
    print(f'utils.mean finished. {len(mean)}')
    return mean

""" 画像のresize関数
Returns:
    [type] -- [description]
"""
def resize(img):
    print("utils.resize called.")
    img = Image.fromarray(img.transpose(1, 2, 0))
    img = img.resize((width, height), Image.BICUBIC)
    r = np.asarray(img).transpose(2, 0, 1)
    print(f'utils.resize finished. {r}')
    return r 

""" 各データに行う変換
Returns:
    [type] -- [description]
"""
def transform(inputs):
    print("utils.transform called.")
    img, label = inputs
    img = img[:3, ...]
    img = resize(img.astype(np.uint8))
    img = img - mean[:, None, None]
    img = img.astype(np.float32)
    # ランダムに左右反転
    if np.random.rand() > 0.5:
        img = img[..., ::-1]
    print(f'utils.transform finished. img = {img}, label = {label}')
    return img, label
