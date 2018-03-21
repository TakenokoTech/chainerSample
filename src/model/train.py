import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from PIL import Image
from chainer import datasets
from common.log import log
from const import CONST

width, height = 160, 160
mean = None

"""平均画像が未計算なら変換をかまさないバージョンの学習用データセットで平均を計算
Returns:
    [type] -- [description]
"""
def calcMean(d) :
    log.info("train.mean called.")
    global mean
    if not os.path.exists(CONST.IMG_MEAN):
        t, _ = datasets.split_dataset_random(d, int(len(d) * 0.8), seed=0)
        mean = np.zeros((3, height, width))
        for img, _ in tqdm_notebook(t, desc='Calc mean'):
            img = resize(img[:3].astype(np.uint8))
            mean += img
        mean = mean / float(len(d))
        np.save(CONST.IMG_MEAN, mean)
    else:
        mean = np.load(CONST.IMG_MEAN)
    # 平均画像の表示
    plt.imshow(mean.transpose(1, 2, 0) / 255)
    # plt.show()
    plt.savefig(CONST.IMG_MEAN_PNG)
    mean = mean.mean(axis=(1, 2))
    log.info(f'train.mean finished. {mean}')

""" 画像のresize関数
Returns:
    [type] -- [description]
"""
def resize(img):
    # log.info("train.resize called.")
    img = Image.fromarray(img.transpose(1, 2, 0))
    img = img.resize((width, height), Image.BICUBIC)
    r = np.asarray(img).transpose(2, 0, 1)
    # log.info(f'train.resize finished. {len(r)}')
    return r 

""" 各データに行う変換
Returns:
    [type] -- [description]
"""
def transform(inputs):
    global mean
    # log.info("train.transform called.")
    img, label = inputs
    img = img[:3, ...]
    img = resize(img.astype(np.uint8))
    # img = img - mean[:, None, None]
    img = img.astype(np.float32)
    if np.random.rand() > 0.5: img = img[..., ::-1] # ランダムに左右反転
    # log.info(f'train.transform finished. img = {len(img)}, label = {label}')
    return img, label

""" 指定したエポックごとに学習率を10分の1にする
Returns:
    [type] -- [description]
"""
def lr_drop(trainer):
    # log.info("train.lr_drop called.")
    trainer.updater.get_optimizer('main').lr *= CONST.LR_DROP_RATIO
