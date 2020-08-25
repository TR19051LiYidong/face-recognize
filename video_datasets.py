import numpy as np
import dlib
import cv2
import os

#############################
#カテゴリごとに50のトレーニング画像
#############################

#現在のプロジェクトのルートディレクトリ、つまり現在のスクリプトディレクトリの上位ディレクトリを取得します
os_path = r'C:\Users\wlx\Documents\py_study\deeplearning\cnn_face_recogntion'
#正面検出器のインポート（インスタンス化）
detector = dlib.get_frontal_face_detector()

#画像サイズ64 * 64
img_size = 64

#画像を64 * 64に圧縮します
def reszie_image(image,height=img_size,width=img_size):
    top,bottom,left,right = 0,0,0,0
    ##############如果图像不是正方形################
    #画像サイズを取得
    h,w = image.shape
    #異なる長さと幅については、最大の側の長さを取ります
    longest_edge = max(h,w)
    #短辺に追加する必要があるピクセル値を計算します
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass
    #############################################
    if top==0 and bottom==0 and left==0 and right==0:
        return cv2.resize(image,(height,width))
    else:
        #境界線の塗りつぶしの色を定義
        BLACK = [0, 0, 0]
        #画像に枠線を追加する
        constant_img = cv2.copyMakeBorder(image,top,bottom,left,right,cv2.BORDER_CONSTANT,value=BLACK)
        return cv2.resize(constant_img,(height,width))

#各タイプのデータに一意のラベル値を割り当てます
def label_id(label,class_list,class_num):
    for i in range(class_num):
        if label == class_list[i]:
            return i

#トレーニング写真の顔データを取得
def TrainFeature(root_train_path,npz_path):   #root_train_path:训练数据的根目录
    #训练数据集
    images_train = []
    #训练数据的标签集
    labels_train = []

    #カテゴリー数
    class_list = os.listdir(root_train_path)
    class_num = len(class_list)

    for img_dir in class_list:
        for img_name in os.listdir(root_train_path+'\\'+img_dir):
            image = cv2.imread(root_train_path+'\\'+img_dir+'\\'+img_name,cv2.IMREAD_GRAYSCALE)
            image = reszie_image(image)

            images_train.append(image)
            labels_train.append(img_dir)
    images_train = np.array(images_train)
    labels_train = np.array([label_id(label,class_list,class_num) for label in labels_train])

    #既存のデータベースを読み取り、そうでない場合は、データベースを作成します
    try:
        npz_dates = np.load(npz_path)
    except:
        np.savez(npz_path, face_imgs=images_train, face_labels=labels_train,face_names = class_list)
    else:
        # データを保存する
        np.savez(npz_path, face_imgs=images_train, face_labels=labels_train,face_names = class_list)
        npz_dates.close()

#テストデータを取得する
def TestFeature(root_test_path,npz_path):   #root_test_path:测试数据的根目录
    #トレーニングデータセット
    images_test = []
    #トレーニングデータのラベルセット
    labels_test = []
    #データの名前


    #カテゴリー数
    class_list = os.listdir(root_test_path)
    class_num = len(class_list)

    for img_dir in class_list:
        for img_name in os.listdir(root_test_path+'\\'+img_dir):
            image = cv2.imread(root_test_path+'\\'+img_dir+'\\'+img_name,cv2.IMREAD_GRAYSCALE)
            image = reszie_image(image)

            images_test.append(image)
            labels_test.append(img_dir)
    images_test = np.array(images_test)
    labels_test = np.array([label_id(label,class_list,class_num) for label in labels_test])

    # 既存のデータベースを読み取り、そうでない場合は、データベースを作成します
    try:
        npz_dates = np.load(npz_path)
    except:
        np.savez(npz_path, face_imgs=images_test, face_labels=labels_test)
    else:
        #データを保存する
        np.savez(npz_path, face_imgs=images_test, face_labels=labels_test)
        npz_dates.close()

if __name__ == '__main__':
    print('#################测试###################')
    #トレーニングデータのルートディレクトリ
    root_train_path = r'C:\Users\wlx\Documents\py_study\deeplearning\cnn_face_recogntion\data\video_face_train'
    npz_train_path = r'C:\Users\wlx\Documents\py_study\deeplearning\cnn_face_recogntion\data\npz\video_face_train.npz'
    #テストデータのルートディレクトリ
    root_test_path = r'C:\Users\wlx\Documents\py_study\deeplearning\cnn_face_recogntion\data\video_face_test'
    npz_test_path = r'C:\Users\wlx\Documents\py_study\deeplearning\cnn_face_recogntion\data\npz\video_face_test.npz'
    TrainFeature(root_train_path,npz_train_path)
    TestFeature(root_test_path,npz_test_path)

    data_train = np.load(npz_train_path)
    print(data_train['face_imgs'].shape)
    print(data_train['face_labels'].shape)
    print(data_train['face_names'])
    data_test = np.load(npz_test_path)
    print(data_test['face_imgs'].shape)
    print(data_test['face_labels'])
