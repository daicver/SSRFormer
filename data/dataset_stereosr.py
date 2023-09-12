import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util
import torch
import cv2
class DatasetSR(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for SISR.
    # If only "paths_H" is provided, sythesize bicubicly downsampled L on-the-fly.
    # -----------------------------------------
    # e.g., SRResNet
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(DatasetSR, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = opt['scale'] if opt['scale'] else 4
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 96
        self.L_size = self.patch_size // self.sf

        # ------------------------------------
        # get paths of L/H
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        self.paths_L = util.get_image_paths(opt['dataroot_L'])

        assert self.paths_H, 'Error: H path is empty.'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L), len(self.paths_H))

    def __getitem__(self, index):

        L_path = None
        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]

        img_H_L = util.imread_uint(H_path, self.n_channels)
        img_H_R = util.imread_uint(H_path[:-5] + 'R.png', self.n_channels)

        img_H_L = util.uint2single(img_H_L)
        img_H_R = util.uint2single(img_H_R)

        if self.paths_L:
            L_path = self.paths_L[index]
            img_L_L = util.imread_uint(L_path, self.n_channels)
            img_L_R = util.imread_uint(L_path[:-5] + 'R.png', self.n_channels)
            img_L_L = util.uint2single(img_L_L)
            img_L_R = util.uint2single(img_L_R)
        else:
            H, W = img_H_L.shape[:2]
            img_L_L = util.imresize_np(img_H_L, 1 / self.sf, True)
            img_L_R = util.imresize_np(img_H_R, 1 / self.sf, True)

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':

            H, W, C = img_L_L.shape

            # # --------------------------------
            # # randomly crop the L patch
            # # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.L_size))
            rnd_w = random.randint(0, max(0, W - self.L_size))
            img_L_L = img_L_L[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]
            img_L_R = img_L_R[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]

            # # --------------------------------
            # # crop corresponding H patch
            # # --------------------------------
            rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
            img_H_L = img_H_L[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]
            img_H_R = img_H_R[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = random.randint(0, 7)
            img_L_L, img_H_L = util.augment_img(img_L_L, mode=mode), util.augment_img(img_H_L, mode=mode)
            img_L_R, img_H_R = util.augment_img(img_L_R, mode=mode), util.augment_img(img_H_R, mode=mode)

        if self.opt['phase'] == 'test':

            H, W, C = img_L_L.shape

            # # --------------------------------
            # # randomly crop the L patch
            # # --------------------------------
            l_size = 96
            h_size = l_size*4
            rnd_h = 0
            rnd_w = 0
            img_L_L = img_L_L[rnd_h:rnd_h + l_size, rnd_w:rnd_w + l_size, :]
            img_L_R = img_L_R[rnd_h:rnd_h + l_size, rnd_w:rnd_w + l_size, :]

            # # --------------------------------
            # # crop corresponding H patch
            # # --------------------------------
            rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
            img_H_L = img_H_L[rnd_h_H:rnd_h_H + h_size, rnd_w_H:rnd_w_H + h_size, :]
            img_H_R = img_H_R[rnd_h_H:rnd_h_H + h_size, rnd_w_H:rnd_w_H + h_size, :]

        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        img_H_L, img_L_L = util.single2tensor3(img_H_L), util.single2tensor3(img_L_L)
        img_H_R, img_L_R = util.single2tensor3(img_H_R), util.single2tensor3(img_L_R)

        if L_path is None:
            L_path = H_path

        return {'LL': img_L_L, 'LR': img_L_R, 'HL': img_H_L, 'HR': img_H_R, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)
