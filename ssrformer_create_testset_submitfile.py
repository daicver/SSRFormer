import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import requests

from models.network_ssrformer import SSRFormer as net
from utils import utils_image as util


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='stereo_sr', help='classical_sr, lightweight_sr, real_sr, '
                                                                     'gray_dn, color_dn, jpeg_car')
    parser.add_argument('--scale', type=int, default=4, help='scale factor: 1, 2, 3, 4, 8') # 1 for dn and jpeg car
    parser.add_argument('--training_patch_size', type=int, default=48, help='patch size used in training SwinIR. '
                                       'Just used to differentiate two different settings in Table 2 of the paper. '
                                       'Images are NOT tested patch by patch.')
    parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
    parser.add_argument('--model_path', type=str,default='stereosuperresolution/ssrformer_patch48x4/models/424000_E.pth')
    parser.add_argument('--folder_lq', type=str, default="datasets/Test/LR_x4", help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # set up model
    if os.path.exists(args.model_path):
        print(f'loading model from {args.model_path}')
    else:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'.format(os.path.basename(args.model_path))
        r = requests.get(url, allow_redirects=True)
        print(f'downloading model {args.model_path}')
        open(args.model_path, 'wb').write(r.content)
        
    model = define_model(args)
    model.eval()
    model = model.to(device)

    # setup folder and path
    folder, save_dir, border, window_size = setup(args)
    os.makedirs(save_dir, exist_ok=True)
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    test_results['psnr_b'] = []
    psnr, ssim, psnr_y, ssim_y, psnr_b = 0, 0, 0, 0, 0

    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*L.png')))):
        print('{:d}/100'.format(idx))
        # read image
        imgname, img_lqL, img_lqR = get_image_pair(args, path)  # image to HWC-BGR, float32
        
        img_lqL = np.transpose(img_lqL if img_lqL.shape[2] == 1 else img_lqL[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lqL = torch.from_numpy(img_lqL).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

        img_lqR = np.transpose(img_lqR if img_lqR.shape[2] == 1 else img_lqR[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lqR = torch.from_numpy(img_lqR).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lqL.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lqL = torch.cat([img_lqL, torch.flip(img_lqL, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lqL = torch.cat([img_lqL, torch.flip(img_lqL, [3])], 3)[:, :, :, :w_old + w_pad]
            
            img_lqR = torch.cat([img_lqR, torch.flip(img_lqR, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lqR = torch.cat([img_lqR, torch.flip(img_lqR, [3])], 3)[:, :, :, :w_old + w_pad]
            
            outputL, outputR = test(img_lqL, img_lqR, model, args, window_size)
            
            outputL = outputL[..., :h_old * args.scale, :w_old * args.scale]
            outputR = outputR[..., :h_old * args.scale, :w_old * args.scale]

        # save image
        outputL = outputL.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        outputR = outputR.data.squeeze().float().cpu().clamp_(0, 1).numpy()

        if outputL.ndim == 3:
            outputL = np.transpose(outputL[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
            outputR = np.transpose(outputR[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        
        outputL = (outputL * 255.0).round().astype(np.uint8)  # float32 to uint8
        outputR = (outputR * 255.0).round().astype(np.uint8)  # float32 to uint8

        imgRname = imgname[:-1] + 'R'
        
        cv2.imwrite(f'{save_dir}/{imgname}.png', outputL)
        cv2.imwrite(f'{save_dir}/{imgRname}.png', outputR)


def define_model(args):
    model = net(upscale=args.scale, in_chans=3, img_size=args.training_patch_size, window_size=8,
                img_range=1., depths=[6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6],
                mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
    param_key_g = 'params'
    
    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
        
    return model


def setup(args):

    save_dir = f'results/ssrformer_test_x{args.scale}'
    folder = args.folder_lq
    border = args.scale
    window_size = 8

    return folder, save_dir, border, window_size

def get_image_pair(args, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))
    img_lqL = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img_lqR = cv2.imread(path[:-5]+'R.png', cv2.IMREAD_COLOR).astype(np.float32) / 255.
    return imgname, img_lqL, img_lqR


def test(img_lqL, img_lqR,  model, args, window_size):
    if args.tile is None:
        # test the image as a whole
        outputL, outputR = model(img_lqL, img_lqR)
    else:
        # test the image tile by tile
        b, c, h, w = img_lqL.size()

        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        EL = torch.zeros(b, c, h*sf, w*sf).type_as(img_lqL)
        ER = torch.zeros(b, c, h*sf, w*sf).type_as(img_lqR)
        WL = torch.zeros_like(EL)
        WR = torch.zeros_like(ER)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:

                in_patchL = img_lqL[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                in_patchR = img_lqR[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                
                out_patchL, out_patchR = model(in_patchL, in_patchR)
                
                out_patchL_mask = torch.ones_like(out_patchL)
                out_patchR_mask = torch.ones_like(out_patchR)

                EL[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patchL)
                WL[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patchL_mask)

                ER[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patchR)
                WR[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patchR_mask)

                del in_patchL, in_patchR, out_patchL, out_patchR, out_patchL_mask, out_patchR_mask

        outputL = EL.div_(WL)
        outputR = ER.div_(WR)

    return outputL, outputR

if __name__ == '__main__':
    main()
