"""
Evaluation script for Real ISR dataset using multiple image quality metrics:
 - PSNR, SSIM (scikit-image)
 - LPIPS (lpips library)
 - DISTS, NIQE (piq library)
 - FID (pytorch-fid)

Usage:
    python Evaluation.py \
        --gt_dir /path/to/gt_images \
        --pred_dir /path/to/generated_images \
        --output_csv metrics_results.csv
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, \
                             structural_similarity as compare_ssim
import torch
import lpips
import piq
from pytorch_fid import fid_score
import tempfile
import shutil

def load_and_align(gt_path: str, pred_path: str):
    gt = Image.open(gt_path).convert('RGB')
    pred = Image.open(pred_path).convert('RGB')

    gt_w, gt_h = gt.size
    pred = pred.resize((gt_w, gt_h), resample=Image.BICUBIC)

    gt_np   = np.array(gt)   / 255.0
    pred_np = np.array(pred) / 255.0
    return gt_np, pred_np

def resize_images_to_dir(src_dir, target_size=(299, 299)):
    # resize all the images in src_dir to the specified size, save them to the temporary directory, and return the path
    tmp_dir = tempfile.mkdtemp()
    for img_name in os.listdir(src_dir):
        img_path = os.path.join(src_dir, img_name)
        try:
            img = Image.open(img_path).convert('RGB').resize(target_size, Image.BICUBIC)
            img.save(os.path.join(tmp_dir, img_name))
        except Exception as e:
            print(f"[Warning] Skipping {img_name}: {e}")
    return tmp_dir

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare the LPIPS model
    lpips_model = lpips.LPIPS(net='alex').to(device)

    # Resize the image to ensure that the FID can be stacked and calculated normally
    gt_resized_dir = resize_images_to_dir(args.gt_dir)
    pred_resized_dir = resize_images_to_dir(args.pred_dir)

    # Calculate FID
    fid_value = fid_score.calculate_fid_given_paths(
        [gt_resized_dir, pred_resized_dir],
        batch_size=32,
        device=str(device),
        dims=2048,
        num_workers=0
    )

    # Clean up the temporary directory
    shutil.rmtree(gt_resized_dir)
    shutil.rmtree(pred_resized_dir)

    # Traverse all the images to calculate other indicators
    results = []
    gt_paths   = sorted(glob.glob(os.path.join(args.gt_dir,  '*')))
    pred_paths = sorted(glob.glob(os.path.join(args.pred_dir, '*')))

    # Initialization index
    dists_metric = piq.DISTS().to(device)
    # niqe_metric = piq.NIQE()

    for gt_path, pred_path in zip(gt_paths, pred_paths):
        img_name = os.path.basename(gt_path)
        gt, pred = load_and_align(gt_path, pred_path)

        # PSNR
        psnr = compare_psnr(gt, pred, data_range=1.0)

        # SSIM
        h, w, _ = gt.shape
        min_dim = min(h, w)
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        ssim = compare_ssim(gt, pred, data_range=1.0, channel_axis=-1, win_size=win_size)

        # LPIPS
        t0 = (torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device).float() * 2 - 1)
        t1 = (torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(device).float() * 2 - 1)
        lpips_val = lpips_model(t0, t1).item()

        # DISTS
        gt_t = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device).float()
        pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(device).float()
        dists_val = dists_metric(pred_t, gt_t).item()

        # NIQE
        # niqe_val = niqe_metric(pred_t).item()

        results.append({
            'image': img_name,
            'PSNR': psnr,
            'SSIM': ssim,
            'LPIPS': lpips_val,
            'DISTS': dists_val,
            # 'NIQE': niqe_val,
        })

    # Add FID
    df = pd.DataFrame(results)
    fid_row = {k: None for k in df.columns}
    fid_row['image'] = 'all'
    fid_row['FID']   = fid_value
    df = pd.concat([df, pd.DataFrame([fid_row])], ignore_index=True)

    df.to_csv(args.output_csv, index=False)
    print(f"[Done] Metrics saved to {args.output_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', required=False,
        default="E:/Sydney_study/5329/A2/RealSR (ICCV2019)/HR",
        help='Directory of ground-truth images')
    parser.add_argument('--pred_dir', required=False,
        default="C:/Users/C/SeeSR/preset/datasets/output/raw_seesr/Predict-original/sample00",# C:/Users/C/SeeSR/preset/datasets/output/sample00
        help='Directory of predicted images')
    parser.add_argument('--output_csv', default='C:/Users/C/SeeSR/preset/datasets/output/metrics_results_rawseesr.csv',
        help='CSV file to save metrics')
    args = parser.parse_args()
    main(args)
