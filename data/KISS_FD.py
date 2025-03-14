import utils.utils as ut
import utils.utils_PM as ut_p
import numpy as np
import os
import os.path as path
import scipy.io as sio
import cv2
from skimage import io
from tqdm import tqdm
import utils.utils as ut
from data.SLP_RD import uni_mod
from utils.utils_ds import *
from os import path
from utils import vis

class KISS_FD:
    def __init__(self, ds, opts, phase='inference', id=None):
        self.opts = opts
        self.ds = ds
        self.phase = phase
        self.data_dir = path.join(opts.data_dir, 'KISS')
        self.image_dir = path.join(self.data_dir, 'images')
        self.id_folder = path.join(self.image_dir, id)
        self.label_dir = path.join(self.data_dir, 'labels')
        self.image_list = os.listdir(self.id_folder)
        self.image_list.sort()
        
    def jt_hm(self, idx):
        mods = self.opts.mod_src
        n_jt = 14
        sz_pch = self.opts.sz_pch
        out_shp = self.opts.out_shp[:2]
        mod0 = mods[0]
        li_img = []
        li_mean = [0.609, 0.609, 0.609]
        li_std = [0.144, 0.144, 0.144]
        # li_mean = self.ds.means[mod0]
        # li_std = self.ds.stds[mod0]
        nmIdx = f'frame_{idx:05d}.jpg'
        imgPth = path.join(self.id_folder, nmIdx)
        img = io.imread(imgPth)
        img = np.array(img)
        
        sz_ori = img.shape[:2]
        bb = [0, 0, sz_ori[0], sz_ori[1]]  # full image bb , make square bb
        bb = ut.adj_bb(bb, rt_xy=1) # get sqrt from ori size
        
        if not 'RGB' in mod0:
            img = img[..., None]
        li_img.append(img)

        img_cb = np.concatenate(li_img, axis=-1)
        scale, rot, do_flip, color_scale, do_occlusion = 1.0, 0.0, False, [1.0, 1.0, 1.0], False
        
        img_patch, trans = generate_patch_image(img_cb, bb, do_flip, scale, rot, do_occlusion, input_shape=self.opts.sz_pch[::-1])
        
        # # 이미지 형식 확인하여 조정
        # if img_patch.shape[0] == 3:  # 이미 (C, H, W) 형식인 경우
        #     # 채널 우선(channel-first) 형식은 그대로 유지하고 ToTensor()를 건너뛰기
        #     img_patch_float = img_patch.astype(np.float32) / 255.0
        #     # Normalize 직접 적용
        #     for c in range(len(li_mean[0])):
        #         img_patch_float[c] = (img_patch_float[c] - li_mean[0][c]) / li_std[0][c]
        #     pch_tch = torch.from_numpy(img_patch_float)
        # else:  # (H, W, C) 형식인 경우
        #     if img_patch.ndim < 3:
        #         img_channels = 1
        #         img_patch = img_patch[..., None]  # 채널 차원 추가
        #     else:
        #         img_channels = img_patch.shape[2]
            
        #     # 픽셀값 범위 조정
        #     for i in range(img_channels):
        #         img_patch[..., i] = np.clip(img_patch[..., i], 0, 255)
            
        
        if img_patch.ndim<3:
            img_channels = 1        # add one channel
            img_patch = img_patch[..., None]
        else:
            img_channels = img_patch.shape[2]   # the channels
            for i in range(img_channels):
                img_patch[:, :, i] = np.clip(img_patch[:, :, i] * color_scale[i], 0, 255)
        
        
        # 일반적인 변환 적용
        trans_tch = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=li_mean, std=li_std)
        ])
        pch_tch = trans_tch(img_patch)
        
        rst = {
            'pch': pch_tch,
            'bb': bb.astype(np.float32)
        }
        return rst
        
    def __getitem__(self, index):
        rst = self.jt_hm(index)
        return rst
    
    def __len__(self):
        return len(self.image_list)