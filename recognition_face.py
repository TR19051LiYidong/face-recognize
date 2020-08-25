from keras.models import load_model
from keras import backend as K
from video_datasets import reszie_image,img_size
import cv2

def face_predict(image,model,class_list,classes):
    ##############顔の特徴を抽出する###################
    # グレースケール処理
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #検出器を使用して画像内の顔を認識し、顔リストを作成します
    face_dets = detector(img_gray, 1)
    #顔がない場合は、画像を元のパスに戻します
    if len(face_dets) == 0:
        return image
    else:
        for det in face_dets:
            #顔領域を抽出
            face_top = det.top() if det.top() > 0 else 0
            face_bottom = det.bottom() if det.bottom() > 0 else 0
            face_left = det.left() if det.left() > 0 else 0
            face_right = det.right() if det.right() > 0 else 0

            face_temp = img_gray[face_top:face_bottom, face_left:face_right]         #灰度图
            face_img = None
            #圧縮画像は64 * 64
            little_face = reszie_image(face_temp)
            ####################################################
            # ディメンションの順序は、バックエンドシステムに従って決定されます
            if K.image_data_format() == 'channels_first':
                face_img = little_face.reshape((1,1,img_size, img_size))               #与模型训练不同，这次只是针对1张图片进行预测
            elif K.image_data_format() == 'channels_last':
                face_img = little_face.reshape((1, img_size, img_size, 1))
            #浮動小数点と正規化
            face_img = face_img.astype('float32')
            face_img /= 255

            #入力が各カテゴリに属する​​確率を与える
            result_probability = model.predict_proba(face_img)
            #print('result:', result_probability)

            # カテゴリ予測を与える（変更）
            if max(result_probability[0]) >= 0.9:
                result = model.predict_classes(face_img)
                #print('result:', result)
                #カテゴリ予測結果を返す
                faceID = result[0]
            else:
                faceID = -1
            #額縁
            cv2.rectangle(image, (face_left - 10, face_top - 10), (face_right + 10, face_bottom + 10), color,
                          thickness=2)
            #face_id判定
            if faceID in classes:
                # テキストプロンプトは誰ですか
                cv2.putText(image, class_list[faceID],
                            (face_left, face_top - 30),  # 座標
                            cv2.FONT_HERSHEY_SIMPLEX,  # フォント
                            1,  # フォントサイズ
                            (255, 0, 255),  # 色
                            2)  # ワードライン幅
            else:
                # テキストプロンプトは誰ですか
                cv2.putText(image, 'None ',
                            (face_left, face_top - 30),  # 座標
                            cv2.FONT_HERSHEY_SIMPLEX,  # フォント
                            1,  # フォントサイズ
                            (255, 0, 255),  # 色
                            2)  # ワードライン幅
    return image

if __name__ == '__main__':
    import dlib
    import numpy as np
    #カテゴリー一覧
    npz_train_path = r'C:\Users\wlx\Documents\py_study\deeplearning\cnn_face_recogntion\data\npz\video_face_train.npz'
    data_train = np.load(npz_train_path)
    class_list = data_train['face_names']
    classes = [i for i in range(len(class_list))]
    #モデルをロードする
    model_path = r'C:\Users\wlx\Documents\py_study\deeplearning\cnn_face_recogntion\data\model\aggregate.face.model.h5'
    cnn_model = load_model(model_path)
    #顔を囲む長方形の枠の色
    color = (0, 255, 0)
    # 指定されたカメラのリアルタイムビデオストリームをキャプチャする
    cap = cv2.VideoCapture(0)
    # 正面検出器のインポート（インスタンス化）
    detector = dlib.get_frontal_face_detector()
    # 顔を認識するためのサイクル検出
    while True:
        ret, frame = cap.read()  # ビデオのフレームを読む
        if ret is False:
            continue
        else:
            frame = face_predict(frame,cnn_model,class_list,classes)
        cv2.imshow("login", frame)

        # 10ミリ秒待って、キー入力があるかどうかを確認します
        k = cv2.waitKey(10)
        # qを入力すると、ループを終了します
        if k & 0xFF == ord('q'):
            break

    # カメラを離してすべてのウィンドウを破壊する
    cap.release()
    cv2.destroyAllWindows()
