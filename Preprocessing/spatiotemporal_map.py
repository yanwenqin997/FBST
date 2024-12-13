import numpy as np
import os
from signal_processing_method import fft_snr

def Space_time_face(image, windows):
    '''  # input：
    # image: input face image
    # windows: window resolution
    # output：spatiotemporal sequence of face image
    '''
    row = np.size(image, axis=0)
    column = np.size(image, axis=1)
    r_win = int(row / windows)
    c_win = int(column / windows)
    RGB = np.zeros((16, 3))
    label = 0  # Label, not four corners
    num = 0  # rgb sequence
    for i in range(r_win):
        for j in range(c_win):
            if label == 0 or label == 3 or label == 12 or label == 15:
                label = label + 1
                continue
            else:
                R = image[i * windows:(i + 1) * windows, j * windows:(j + 1) * windows, 0]
                G = image[i * windows:(i + 1) * windows, j * windows:(j + 1) * windows, 1]
                B = image[i * windows:(i + 1) * windows, j * windows:(j + 1) * windows, 2]
                # Non-zero region mean
                RGB[num, 0] = R.flatten()[np.nonzero(R.flatten())].mean()
                RGB[num, 1] = G.flatten()[np.nonzero(G.flatten())].mean()
                RGB[num, 2] = B.flatten()[np.nonzero(B.flatten())].mean()
                label = label + 1
                num = num + 1
    RGB[np.where(np.isnan(RGB))] = 0
    RGB = RGB[0:12, :]
    return RGB
def Space_time_face_map(vid_dir,save_dir, windows):
    '''  input：
    vid_dir: path of face videos
    windows: window resolution
    save_dir：path to save the spatiotemporal map of face videos
    '''
    videoNameList = os.listdir(vid_dir)
    for i in videoNameList:
        vid_path = os.path.join(vid_dir, i).replace("\\", "/")
        vid = np.load(vid_path)
        rgb = np.zeros((len(vid), 12, 3))
        for j in range(len(vid)):
            image = vid[j]
            rgb[j] = Space_time_face(image, windows)
        np.save(os.path.join(save_dir, i).replace("\\", "/"), rgb)


def Space_time_chest_map(vid_dir,save_dir,window_length,window_height):
    '''  # input：
    vid_dir: path of chest videos
    window_length: length of window resolution
    window_height： height of window resolution
    save_dir：path to save the spatiotemporal map of chest videos
    '''
    vid_list = os.listdir(vid_dir)
    for s in vid_list:
        c = np.load(os.path.join(vid_dir, s).replace('\\', '/'))  #
        row_n = int((c.shape[2] - window_length) / window_length) + 1
        col_n = int((c.shape[1] - window_height) / window_height) + 1
        roi = np.zeros((len(c), row_n * col_n, 3))
        for column in range(0, col_n):
            for row in range(0, row_n):
                r = c[:, 32 * column:32 + 32 * column, 64 * row:64 + 64 * row, :]
                roi[:, column * row_n + row, :] = np.mean(np.mean(r, 1), 1)
        snr = np.zeros(row_n * col_n)
        for i in range(0, row_n * col_n):
            snr[i] = fft_snr(roi[:, i, 0], 30) # Computational snr
        a = np.argsort(-snr)
        roi_max = roi[:, a[:12], :]#The 12 ROIs with the largest snr were selected to construct the chest spatiotemporal map
        np.save(os.path.join(save_dir,s[:-4]+'.npy').replace('\\', '/'),roi_max)
if __name__ == '__main__':
    windows = 32 #you can resize the window to the size you need
    window_length =64
    window_height = 32
    vid_face_dir = 'your path'#path of face videos
    vid_chest_dir = 'your path'  # path of chest videos
    save_face_dir = 'your save path'#path to save the spatiotemporal map of face videos
    save_chest_dir = 'your save path'  # path to save the spatiotemporal map of chest videos
    Space_time_face_map(vid_face_dir, save_face_dir, windows)
    Space_time_chest_map(vid_chest_dir,save_chest_dir,window_length,window_height)