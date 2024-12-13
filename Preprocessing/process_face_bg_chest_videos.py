from SeetaFace6.seetaFace6Python.seetaface.api import *
import cv2
import numpy as np
import os
import mediapipe as mp
from sklearn.decomposition import PCA


def read_face_bg_videos(vid_dir,save_bg_dir,save_face_dir):
    '''  # input：
    vid_dir: path of original videos, Secondary file directory
    save_bg_dir: path to save the background spatiotemporal map
    save_face_dir： path to save the face imgaes
    '''
    #Face recognition parameter setting
    init_mask = FACE_DETECT | LANDMARKER5 | LANDMARKER68 | LANDMARKER_MASK
    seetaFace = SeetaFace(init_mask)
    seetaFace.SetProperty(DetectProperty.PROPERTY_MIN_FACE_SIZE, 80)
    seetaFace.SetProperty(DetectProperty.PROPERTY_THRESHOLD, 0.9)

    #Face and background extraction
    vidlist = os.listdir(vid_dir)
    for s in vidlist:
        path = os.path.join(vid_dir, s).replace('\\', '/')
        in_list = os.listdir(path)
        for m in in_list:
            vidcap = cv2.VideoCapture(os.path.join(path, m, "data.avi").replace('\\', '/'))
            n = 0
            imgList = []
            bgList = []
            while (vidcap.isOpened()):
                ret, frame = vidcap.read()
                fps = round(vidcap.get(5))
                if ret == True:
                    if n >= 0:
                        ind = ("%d" % n).zfill(4)
                        image = frame
                        fps = round(vidcap.get(5))
                        ih, iw, ic = image.shape
                        detect_result = seetaFace.Detect(image)
                        facePos = detect_result.data[0].pos
                        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        imgCrop = image_rgb[facePos.y:facePos.y + facePos.height, facePos.x:facePos.x + facePos.width,
                                  :]
                        imgCrop = cv2.resize(imgCrop, (128, 128), interpolation=cv2.INTER_CUBIC)  # weight,high
                        imgList.append(imgCrop)
                        bg1 = image_rgb[0:150, 0:150, :]
                        bg2 = image_rgb[150:300, 0:150, :]
                        bg3 = image_rgb[0:300, 500:600, :]
                        bgList.append([np.mean(np.mean(bg1,0),0),np.mean(np.mean(bg2,0),0),np.mean(np.mean(bg3,0),0)])
                        print(n)
                        n += 1
                else:
                    break
            print('Extraction complete', s, '-', m)
            imgR = np.stack(imgList, 0)
            imgR1 = np.uint8(imgR)
            np.save(os.path.join(save_face_dir, s + "-" + m + ".npy").replace('\\', '/'), imgR1)
            imgB = np.stack(bgList, 0)
            pca = PCA(n_components=1)  # Load PCA algorithm and set the number of principal components to 1 after dimensionality reduction
            imgBB = np.zeros((len(imgB),3))
            imgBB[:,0] = pca.fit_transform(imgB[:,0,:])[:,0]
            imgBB[:,1] = pca.fit_transform(imgB[:,1,:])[:,0]
            imgBB[:,2] = pca.fit_transform(imgB[:,2,:])[:,0]
            imgBB = imgB.repeat(12, axis=1)#spatiotemporal map of background
            np.save(os.path.join(save_bg_dir, s + "-" + m + ".npy").replace('\\', '/'), imgBB)
            vidcap.release()  # release videos
            cv2.destroyAllWindows()
def read_chest_videos(vid_dir,save_chest_dir):
    '''  # input：
      vid_dir: path of original videos, Secondary file directory
      save_chest_dir： path to save the chest imgaes
      '''

    # gesture recognition parameter setting
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # chest extraction
    vidlist = os.listdir(vid_dir)
    for s in vidlist:
        path = os.path.join(vid_dir, s).replace('\\', '/')
        in_list = os.listdir(path)
        for m in in_list:
            vidcap = cv2.VideoCapture(os.path.join(path, m, "data.avi").replace('\\', '/'))
            n = 0
            rgbList = []
            while (vidcap.isOpened()):
                ret, frame = vidcap.read()
                fps = round(vidcap.get(5))
                if ret == True:
                    if n >= 0:
                        ind = ("%d" % n).zfill(4)
                        image = frame
                        fps = round(vidcap.get(5))
                        ih, iw, ic = image.shape
                        # Convert the image to RGB
                        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        if n >= 0:
                            results = pose.process(imgRGB)
                            pose_landmarks = results.pose_landmarks.landmark
                        if len(imgRGB) - int(pose_landmarks[12].y * ih) > 150:
                            height = 150
                        else:
                            height = len(imgRGB) - int(pose_landmarks[12].y * ih)
                        rgbCrop = imgRGB[int(pose_landmarks[12].y*ih):int(pose_landmarks[12].y*ih+height),
                                              int(pose_landmarks[12].x*iw):int(pose_landmarks[11].x*iw),:]
                        rgbCrop = cv2.resize(rgbCrop, (256, 128))

                        print(n)
                        n += 1
                        rgbList.append(rgbCrop)
                else:
                    break
            imgR = np.stack(rgbList, 0)
            imgR1 = np.uint8(imgR)
            np.save(os.path.join(save_chest_dir,m[:-4]+".npy").replace('\\', '/'),imgR1)
            vidcap.release()  # Release videos
            cv2.destroyAllWindows()
