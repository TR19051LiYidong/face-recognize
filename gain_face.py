################
#カメラで写真を撮って人物の顔情報を取得する
################

import cv2
import os
import dlib

#ディレクトリを作成する
def CreateFolder(path):
    #先頭のスペースを削除します
    del_path_space = path.strip()
    #尾を削除します
    del_path_tail = del_path_space.rstrip('\\')
    #入力パスがすでに存在するかどうかを判別します
    isexists = os.path.exists(del_path_tail)
    if not isexists:
        os.makedirs(del_path_tail)
        return True
    else:
        return False

#顔を抽出して保存する
def CatchPICFromVideo(window_name,camera_idx,catch_pic_num,path_name):

    #入力パスが存在するかどうかを確認し、存在しない場合は作成してください
    CreateFolder(path_name)
    cv2.namedWindow(window_name)
    #
ビデオソース、保存されたビデオから、または直接USBカメラから取得できます
    cap = cv2.VideoCapture(camera_idx)

    #正面検出器のインポート（インスタンス化）
    detector = dlib.get_frontal_face_detector()

    #顔が認識された後に描画されるフレームの色、RGB形式
    color = (0, 255, 0)
    #顔の数
    num = 0
    while cap.isOpened():
        ok, frame = cap.read()  #データのフレームを読み取る
        #画像の取得に失敗しました、終了する
        if not ok:
            break
        #現在のフレーム画像をグレースケール画像に変換します
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #検出器を使用して画像内の顔を認識し、顔リストを作成します
        face_dets = detector(img_gray, 1)
        det = None
        #顔がない場合は、このループを終了します
        if len(face_dets) == 0:
            continue
        elif len(face_dets) > 1:
            #最大の顔だけいります
            temp_area = 0
            temp = 0
            for i, face_area in enumerate(face_dets):
                if (face_area.right() - face_area.left()) * (face_area.bottom() - face_area.top()) > temp_area:
                    temp_area = (face_area.right() - face_area.left()) * (face_area.bottom() - face_area.top())
                    temp = i
            det = face_dets[temp]
        else:
            det = face_dets[0]
        # 顔領域を抽出
        face_top = det.top() if det.top() > 0 else 0
        face_bottom = det.bottom() if det.bottom() > 0 else 0
        face_left = det.left() if det.left() > 0 else 0
        face_right = det.right() if det.right() > 0 else 0
        # 現在のフレームを画像として保存する
        img_name = '%s\%d.jpg' % (path_name, num)

        #face_img = frame[face_top:face_bottom, face_left:face_right]
        face_img = img_gray[face_top:face_bottom, face_left:face_right]          #保存灰度人脸图
        cv2.imwrite(img_name, face_img)
        #顔の数+ 1
        num += 1
        #長方形のフレームを描画すると、認識された顔よりも少し大きくなります
        cv2.rectangle(frame, (det.left() - 10,det.top() - 10), (det.right() + 10, det.bottom() + 10), color, 2)
        #現在キャプチャされている顔画像の数を表示します
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'num:%d' % (num), (det.left() + 30, det.top() + 30), font, 1, (255, 0, 255), 4)
        # 指定した最大保存数量を超えてプログラムを終了します
        if num > (catch_pic_num): break
        # 画像を表示
        cv2.imshow(window_name, frame)
        #キーボードの「Q」を押して、収集を中断します
        c = cv2.waitKey(25)
        if c & 0xFF == ord('q'):
            break
    # カメラを離してすべてのウィンドウを破壊する
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    while True:
        print("情報を入力しますか(Yes or No)?")
        if input() == 'Yes':
            #名前(英語を入力する、漢字はエラーが発生しやすい)
            new_user_name = input("名前を入力してください："")

            print("カメラを見てください！")

            #画像の枚数は自分で設定します。画像が多いほど認識精度は上がりますが、学習速度は遅くなります
            window_name = 'information'           #画像ウィンドウ
            camera_id = 0                        #カメラのID番号
            images_num = 50                      #収集した写真の数
            # 画像保存場所
            path = r'C:\Users\Administrator\Desktop\work\cnn_face_recogntion\data\video_face_train' + '/' + new_user_name

            CatchPICFromVideo(window_name, camera_id, images_num, path)
        else:
            break
