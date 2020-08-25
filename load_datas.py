import numpy as np
import os
from keras import backend as K
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

#現在のプロジェクトのルートディレクトリ、つまり現在のスクリプトディレクトリの上位ディレクトリを取得します
os_path = r'C:\Users\wlx\Documents\py_study\deeplearning\cnn_face_recogntion'
IMAGE_SIZE = 64

class Dataset:
    def __init__(self,train_npz_path,test_npz_path,train_img_dir):

        #トレーニングデータセットの読み込みパス
        self.train_npz_path = train_npz_path
        #テストデータセットの読み込みパス
        self.test_npz_path = test_npz_path
        #画像タイプ
        self.user_num = len(os.listdir(train_img_dir))
        #現在のライブラリで採用されている次元順序
        self.input_shape = None

    #トレーニングデータセットを読み込み、クロス検証の原則に従ってデータセットを分割し、関連する前処理作業を実行します。
    def load_train_valid(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE,
             img_channels=1):
        #データ・タイプ
        nb_classes = self.user_num
        #データセットをメモリにロードする
        datas = np.load(self.train_npz_path)
        images,labels = datas['face_imgs'],datas['face_labels']
        #ラベルの寸法を変換する
        labels = labels.reshape(len(labels),1)

        train_images, valid_images, train_labels, valid_labels = train_test_split(images,labels, test_size=0.3,random_state=1)
        # 現在の次元の順序が 'th' == 'channels_first'の場合、画像データを入力するときの順序は次のとおりです：channels,rows,cols，それ以外の場合：rows,cols,channels
        # コードのこの部分は、kerasライブラリで必要な次元の順序に従ってトレーニングデータセットを再編成することです
        if K.image_data_format() == 'channels_first':
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)

        #トレーニングセット、検証セット、およびテストセットの数を出力します
        print(train_images.shape[0], 'train samples')
        print(valid_images.shape[0], 'valid samples')

        # 我们的模型使用categorical_crossentropy作为损失函数，因此需要根据类别数量nb_classes将
        # 类别标签进行one-hot编码使其向量化
        train_labels = np_utils.to_categorical(train_labels, nb_classes)
        valid_labels = np_utils.to_categorical(valid_labels, nb_classes)

        # ピクセルデータは正規化のためにフロートされます
        train_images = train_images.astype('float32')
        valid_images = valid_images.astype('float32')

        # それを正規化し、画像のピクセル値を0〜1の間隔に正規化します
        train_images /= 255
        valid_images /= 255

        return train_images,valid_images,train_labels,valid_labels

    #テストデータセットの読み込み
    def load_test(self,img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE,
             img_channels=1):
        # データ・タイプ
        nb_classes = self.user_num
        #テストデータセットをメモリに読み込む
        datas = np.load(self.test_npz_path)
        test_images, test_labels = datas['face_imgs'], datas['face_labels']
        #test_labelsの次元を変換する
        test_labels = test_labels.reshape(len(test_labels),1)
        # 現在の次元の順序が 'th' == 'channels_first'の場合、画像データを入力するときの順序は次のとおりです：channels,rows,cols，それ以外の場合：rows,cols,channels
        # コードのこの部分は、kerasライブラリで必要な次元の順序に従ってトレーニングデータセットを再編成することです
        if K.image_data_format() == 'channels_first':
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
        else:
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)

        #トレーニングセット、検証セット、およびテストセットの数を出力します
        print(test_images.shape[0], 'test samples')

        # 我们的模型使用categorical_crossentropy作为损失函数，因此需要根据类别数量nb_classes将
        # 类别标签进行one-hot编码使其向量化
        test_labels = np_utils.to_categorical(test_labels, nb_classes)
        #ピクセルデータは正規化のためにフロートされます
        test_images = test_images.astype('float32')
        #それを正規化し、画像のピクセル値を0〜1の間隔に正規化します
        test_images /= 255

        return test_images,test_labels


if __name__ == '__main__':
    train_npz_path = os_path + r'\data\npz\video_face_train.npz'
    test_npz_path = os_path + r'\data\npz\video_face_test.npz'
    train_img_dir = os_path + r'\data\video_face_train'
    #試してみる
    datasets = Dataset(train_npz_path,test_npz_path,train_img_dir)
    train_images,valid_images,train_labels,valid_labels = datasets.load_train_valid()
    test_images, test_labels = datasets.load_test()
    print(train_labels)
    print(test_labels)
