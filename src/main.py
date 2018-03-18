import os
import glob
from itertools import chain
from chainer.datasets import LabeledImageDataset
from chainer.datasets import TransformDataset
from chainer.dataset import concat_examples
from chainer.training import extensions
from chainer.training import triggers
from chainer import datasets
from chainer import iterators
from chainer import training
from chainer import optimizers
from chainer import optimizer
import chainer.functions as F
import chainer.links as L
from IPython.display import Image

from utils.util import calcMean
from utils.util import transform
from utils.log import log
from model.Illust2Vec import Illust2Vec

if __name__ == '__main__':

    # buildフォルダ作成
    os.makedirs("build/result", exist_ok=True)
    os.makedirs("build/imageMean", exist_ok=True)
    os.makedirs("build/Illustration2Vec", exist_ok=True)

    # 画像ディレクトリ
    IMG_DIR = 'dataset/animeface-character-dataset/thumb'

    # 各キャラクター
    dnames = glob.glob('{}/*'.format(IMG_DIR))

    # 画像ファイルパス一覧
    fnames = [glob.glob('{}/*.png'.format(d)) for d in dnames if not os.path.exists('{}/ignore'.format(d))]
    fnames = list(chain.from_iterable(fnames))
    # for f in fnames: log.debug(f)

    # それぞれにフォルダ名から一意なIDを付与
    labels = [os.path.basename(os.path.dirname(fn)) for fn in fnames]
    dnames = [os.path.basename(d) for d in dnames if not os.path.exists('{}/ignore'.format(d))]
    labels = [dnames.index(l) for l in labels]
    # for l in labels: log.debug(l)

    # データセット作成
    dataset = LabeledImageDataset(list(zip(fnames, labels)))
    # for d in dataset: log.debug(d)
    calcMean(dataset)

    # 変換付きデータセットにする
    transData = TransformDataset(dataset, transform)
    # for d in transData: log.debug(d)

    # データセットを学習用と検証用に分ける
    train, valid = datasets.split_dataset_random(transData, int(len(dataset) * 0.8), seed=0)

    # モデルインポート
    n_classes = len(dnames)
    model = Illust2Vec(n_classes)
    model = L.Classifier(model)

    # 
    batchsize = 64
    gpu_id = 0
    initial_lr = 0.01
    lr_drop_epoch = [10]
    lr_drop_ratio = 0.1
    train_epoch = 10

    #
    train_iter = iterators.MultiprocessIterator(train, batchsize)
    valid_iter = iterators.MultiprocessIterator(valid, batchsize, repeat=False, shuffle=False)
    opt = optimizers.MomentumSGD(lr=initial_lr)
    opt.setup(model)
    opt.add_hook(optimizer.WeightDecay(0.0001))
    updater = training.StandardUpdater(train_iter, opt, device=gpu_id)
    trainer = training.Trainer(updater, (train_epoch, 'epoch'), out='build/result')
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.observe_lr())

    # 標準出力に書き出したい値
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time', 'lr']))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png'))

    # モデルのtrainプロパティをFalseに設定してvalidationするextension
    trainer.extend(extensions.Evaluator(valid_iter, model, device=gpu_id))

    # 指定したエポックごとに学習率を10分の1にする
    def lr_drop(trainer):
        log.info("main.lr_drop called.")
        trainer.updater.get_optimizer('main').lr *= lr_drop_ratio

    trainer.extend(lr_drop, trigger=triggers.ManualScheduleTrigger(lr_drop_epoch, 'epoch'))
    trainer.run()

    Image(filename='build/result/loss.png')
    Image(filename='build/result/accuracy.png')