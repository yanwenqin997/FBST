import numpy as np
import scipy.interpolate as spi
import scipy
import os
from signal_processing_method import filt,fft_HR

def Data_Aug(rgb_dir,gt_dir,bg_pca_dir,fs):
    """
      Data augmentation based on heart rate
      rgb_dir: raw signals are derived from face viedos
      gt_dir：ground truth signals(BVP signals)
      bg_pca_dir：Raw signals after PCA of background signals
      """
    dir_list = os.listdir(gt_dir)
    for s in dir_list:
        bvp = np.load(os.path.join(gt_dir, s).replace('\\', '/'))
        rgb = np.load(os.path.join(rgb_dir, s).replace('\\', '/'))
        bg_pca = np.load(os.path.join(bg_pca_dir, s).replace('\\', '/'))
        bvp_filt = filt(bvp,fs,0.667,3)
        hr = fft_HR(bvp_filt, fs)#heart rate (40-240bpm）
        if hr > 90:  # 1.5 interpolated upsamples
            bvp_len = np.linspace(0, len(bvp) - 1, len(bvp))
            f_bvp = scipy.interpolate.interp1d(bvp_len, bvp, kind='cubic')
            bvp_len_new = np.linspace(0, len(bvp) - 1, int(1.5 * len(bvp)) - 1)
            bvp_re = f_bvp(bvp_len_new)

            f_rgb = scipy.interpolate.interp1d(bvp_len, rgb, kind='cubic', axis=0)
            bvp_len_new = np.linspace(0, len(bvp) - 1, int(1.5 * len(bvp)) - 1)
            rgb_re = f_rgb(bvp_len_new)

            f_bg_pca = scipy.interpolate.interp1d(bvp_len, bg_pca, kind='cubic', axis=0)
            bvp_len_new = np.linspace(0, len(bvp) - 1, int(1.5 * len(bvp)) - 1)
            bg_pca_re = f_bg_pca(bvp_len_new)
            np.save(os.path.join(gt_dir, s[:-4] + s[:-4]).replace('\\', '/'), bvp_re)
            np.save(os.path.join(rgb_dir, s[:-4] + s[:-4]).replace('\\', '/'), rgb_re)
            np.save(os.path.join(bg_pca_dir, s[:-4] + s[:-4]).replace('\\', '/'), bg_pca_re)

        elif hr < 90:  # Double downsampling
            bvp_len = np.linspace(0, len(bvp) - 1, int((len(bvp) - 1) / 2))
            bvp_len = bvp_len.astype(int)
            bvp_re = bvp[bvp_len]
            rgb_re = rgb[bvp_len, :, :]
            bg_pca_re = bg_pca[bvp_len, :, :]
            np.save(os.path.join(gt_dir, s[:-4] + s[:-4]).replace('\\', '/'), bvp_re)

            np.save(os.path.join(rgb_dir, s[:-4] + s[:-4]).replace('\\', '/'), rgb_re)
            np.save(os.path.join(bg_pca_dir, s[:-4] + s[:-4]).replace('\\', '/'), bg_pca_re)

