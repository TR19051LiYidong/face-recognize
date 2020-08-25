from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import load_model
from losshistory import LossHistory

#現在のプロジェクトのルートディレクトリ、つまり現在のスクリプトディレクトリの上位ディレクトリを取得します
os_path = r'C:\Users\wlx\Documents\py_study\deeplearning\cnn_face_recogntion'

#CNNネットワークモデルクラス
class Model:
    def __init__(self):
        self.model = None
        self.history = LossHistory()

        # モデリング
    def build_model(self, dataset,nb_classes=3):

        # 空のネットワークモデルを構築します。これは線形スタックモデルであり、各ニューラルネットワークレイヤーが順次追加されます。専門家の名前は順次モデルまたは線形スタックモデルです。
        self.model = Sequential()

        #次のコードは、CNNネットワークに必要なレイヤーを順次追加します。追加はネットワークレイヤーです
        self.model.add(Convolution2D(32, 3, 3, border_mode='same',
                                     input_shape=dataset.input_shape))  #レイヤー1：2次元たたみ込みレイヤー
        self.model.add(Activation('relu'))                              #レイヤー1：アクティベーション機能

        self.model.add(Convolution2D(32, 3, 3))                         #レイヤー2：2次元畳み込みレイヤー
        self.model.add(Activation('relu'))                              #レイヤー2：アクティベーション機能
        #プーリング層の役割：
        #1.invariance:translation,rotation,scale
        #2.主な特徴を維持しながら、次元の削減が行われ、過剰適合を防止し、モデルの汎化能力を改善します
        self.model.add(MaxPooling2D(pool_size=(2, 2)))                  #レイヤー3：プーリング層
        self.model.add(Dropout(0.25))                                   #レイヤー3：Dropout——このレイヤーの各ノードは、非アクティブ化の確率が25％です。

        self.model.add(Convolution2D(64, 3, 3, border_mode='same'))     #レイヤー4:2次元たたみ込みレイヤー
        self.model.add(Activation('relu'))                              #レイヤー4:アクティベーション機能

        self.model.add(Convolution2D(64, 3, 3))                         #レイヤー5:2维卷积层
        self.model.add(Activation('relu'))                              #レイヤー5:アクティベーション機能

        self.model.add(MaxPooling2D(pool_size=(2, 2)))                  #レイヤー6：プーリング層
        self.model.add(Dropout(0.25))                                   #レイヤー6:Dropout

        self.model.add(Flatten())                                       #レイヤー7:Flatten()——多次元入力1次元になる
        self.model.add(Dense(512))                                      #レイヤー7:Dense層,完全に接続されたレイヤー
        self.model.add(Activation('relu'))                              #レイヤー7:アクティベーション機能
        self.model.add(Dropout(0.5))                                    #レイヤー7:Dropout

        self.model.add(Dense(nb_classes))                               #レイヤー８:Dense層
        self.model.add(Activation('softmax'))                           #レイヤー８:分類層、最終結果を出力

        #出力モデルの概要
        self.model.summary()

    # トレーニングモデル
    def train(self, train_images,train_labels,valid_images,valid_labels, batch_size=20, nb_epoch=10, data_augmentation=False):
        sgd = SGD(lr=0.01, decay=1e-6,
                  momentum=0.9, nesterov=True)  # SGD + momentumのオプティマイザをトレーニングに使用し、最初にオプティマイザオブジェクトを生成します
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])  # 実際のモデル構成作業を完成する

        # データ拡張を使用しない場合、いわゆる拡張とは、回転、反転、ノイズの追加などによって提供するトレーニングデータから新しいトレーニングデータを作成し、トレーニングデータのサイズを意識的に増やし、モデルトレーニングの量を増やすことです。
        if not data_augmentation:
            self.model.fit(train_images,
                           train_labels,
                           batch_size=batch_size,
                           nb_epoch=nb_epoch,
                           validation_data=(valid_images,valid_labels),
                           shuffle=True,
                           callbacks=[self.history])
        #リアルタイムデータを使用して改善する
        else:
            # データプロモーション用のデータジェネレーターを定義します。ジェネレーターオブジェクトdatagenを返します。datagenが呼び出されるたびに一連のデータを生成し（順次生成）、メモリを節約します。これは実際にはpythonデータジェネレーターです
            datagen = ImageDataGenerator(
                featurewise_center=False,               # 入力データを分散化するかどうか（平均値は0）
                samplewise_center=False,                # 入力データの各サンプルを0にするかどうか
                featurewise_std_normalization=False,    # データが標準化されているかどうか（データセットの標準偏差で割った入力データ）
                samplewise_std_normalization=False,     # 各サンプルデータを独自の標準偏差で除算するかどうか
                zca_whitening=False,                    # 入力データにZCAホワイトニングを適用するかどうか
                rotation_range=20,                      # データが増加したときの画像のランダムな回転の角度（範囲0〜180）
                width_shift_range=0.2,                  # データがプロモートされたときの画像の水平オフセットの振幅（単位は画像の幅の割合、0〜1の浮動小数点数）
                height_shift_range=0.2,                 # 上記と同じですが、ここは垂直です
                horizontal_flip=True,                   # ランダムな水平反転を実行するかどうか
                vertical_flip=False)                    # ランダム垂直フリップを実行するかどうか

            # 固有値の正規化、ZCAホワイトニングなどのトレーニングサンプルセット全体の数を計算します。
            datagen.fit(train_images)

            # ジェネレーターを使用してモデルのトレーニングを開始します
            self.model.fit_generator(datagen.flow(train_images, train_labels,
                                                  batch_size=batch_size),
                                     samples_per_epoch=train_images.shape[0],
                                     nb_epoch=nb_epoch,
                                     validation_data=(valid_images, valid_labels),
                                     callbacks=[self.history])

    MODEL_PATH = os_path + '/data/model/aggregate.face.model.h5'

    def save_model(self, file_path=MODEL_PATH):
        self.model.save(file_path)

    def load_model(self, file_path=MODEL_PATH):
        self.model = load_model(file_path)

    def evaluate(self, test_images,test_labels):
        score = self.model.evaluate(test_images, test_labels, verbose=1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))



if __name__ == '__main__':
    from load_datas import Dataset

    #トレーニングデータパス
    train_npz_path = os_path + r'\data\npz\video_face_train.npz'
    #テストデータパス
    test_npz_path = os_path + r'\data\npz\video_face_test.npz'
    #トレーニング画像を保存するパス
    train_img_dir = os_path + r'\data\video_face_train'
    #データオブジェクトをインスタンス化する
    datasets = Dataset(train_npz_path, test_npz_path, train_img_dir)
    #トレーニングと検証データをインポートする
    train_images,valid_images,train_labels,valid_labels = datasets.load_train_valid()
    #テストデータをインポートする
    test_images, test_labels = datasets.load_test()

    #cnnモデルオブジェクトのインスタンス化
    cnn_model = Model()
    #モデルを作成する
    cnn_model.build_model(datasets,nb_classes=datasets.user_num)        #datasets.user_num种类数量
    #トレーニングモデル
    cnn_model.train(train_images,train_labels,valid_images,valid_labels,batch_size=20,nb_epoch=10)
    #評価モデル
    cnn_model.evaluate(test_images,test_labels)
    #損失曲線と精度曲線を描く
    cnn_model.history.loss_plot('epoch')
    #モデルを保存
    cnn_model.save_model()

    # #モデルをロードする
    # model_path = r'C:\Users\wlx\Documents\py_study\deeplearning\cnn_face_recogntion\data\model\aggregate.face.model.h5'
    # cnn_model = load_model(model_path)
    # score = cnn_model.model.evaluate(test_images, test_labels, verbose=1)
    # print("%s: %.2f%%" % (cnn_model.model.metrics_names[1], score[1] * 100))



