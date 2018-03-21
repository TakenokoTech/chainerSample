class CONST:

    # 画像ディレクトリ
    IMG_DIR = 'dataset/animeface-character-dataset/thumb'
    NEW_LOG = 'build/result/log'
    RESULT_MODEL = 'build/result/trainer.model'
    RESULT_LOG = 'build/result/result.log'

    BATCH_SIZE = 64
    GPU_ID = 0
    INITIAL_LR = 0.01
    LR_DROP_EPOCH = [10]
    LR_DROP_RATIO = 0.1
    TRAIN_EPOCH = 10
    CHECK_COUNT = 100

    CAFFEMODEL_FN = './dataset/Illustration2Vec/illust2vec_ver200.caffemodel'
    PKL_FN = './build/Illustration2Vec/illust2vec_ver200.pkl'

    IMG_MEAN = 'build/imageMean/image_mean.npy'
    IMG_MEAN_PNG = 'build/imageMean/image_mean.png'