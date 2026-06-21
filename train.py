"""
dataset_loader.py  → 读取图片、pose、focal
rays.py            → 采样 rays_o, rays_d, target_rgb
model.py           → NeRF MLP
render.py          → volume rendering 得到 pred_rgb
train.py           → loss + backprop + optimizer
"""


import os
import argparse

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dataset.dataset_loader import load_blender_data
from ray import sample_random_rays_from_one_image,sample_foreground_rays_from_one_image,sample_mixed_rays_from_one_image
from model import NeRFMLP
from render import render_rays


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Tiny NeRF model")

    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to Blender scene folder, e.g. data/lego"
    )

    parser.add_argument(
        "--target_size",
        type=int,
        default=100,
        help="Resize images to target_size x target_size"
    )
    
    parser.add_argument(
        "--foreground_only",
        action="store_false",
        help="Only sample rays from foreground pixels"
    )

    parser.add_argument(
        "--mixed_sampling",
        action="store_true",
        help="Sample rays from both foreground and background pixels"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Threshold for foreground pixel detection"
    )

    parser.add_argument(
        "--max_images",
        type=int,
        default=70,
        help="Number of training images to load. Use -1 for all images."
    )

    parser.add_argument(
        "--num_steps",
        type=int,
        default=3000,
        help="Number of training steps"
    )

    parser.add_argument(
        "--num_rays",
        type=int,
        default=1024,
        help="Number of random rays per training step"
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=64,
        help="Number of sampled points along each ray"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate"
    )

    parser.add_argument(
        "--near",
        type=float,
        default=2.0,
        help="Near sampling bound"
    )

    parser.add_argument(
        "--far",
        type=float,
        default=6.0,
        help="Far sampling bound"
    )

    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension of NeRF MLP"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="outputs",
        help="Directory to save outputs"
    )

    return parser.parse_args()


def psnr_from_mse(mse):
    """
    Peak Signal-to-Noise Ratio 峰值信噪比
    PSNR = -10 * log10(MSE)
    Higher PSNR means better reconstruction.
    """
    return -10.0 * torch.log10(mse)


def main(args):
    

    os.makedirs(args.save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    max_images = None if args.max_images == -1 else args.max_images

    # 1. Load dataset
    images, transform_matrixs, focal, H, W = load_blender_data(
        data_dir=args.data_dir,
        split="train",
        target_size=args.target_size,
        white_background=True,
        max_images=max_images,
    )

    images = images.to(device)
    transform_matrixs = transform_matrixs.to(device)

    print("Loaded data")
    print("images:", images.shape)
    print("poses:", transform_matrixs.shape)
    print("focal:", focal)
    print("H, W:", H, W)

    # 2. Build model
    model = NeRFMLP(
        pos_freqs=10,
        dir_freqs=4,
        hidden_dim=256,
        num_layers=8,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_history = []
    psnr_history = []

    # 3. Training loop
    for step in range(args.num_steps):
        model.train()

        # Sample random rays and target RGB values
        if args.foreground_only:
            batch_rays_o, batch_rays_d, target_rgb = sample_foreground_rays_from_one_image(
                images=images,
                transform_matrixs=transform_matrixs,
                focal=focal,
                H=H,
                W=W,
                num_rays=args.num_rays,
                threshold=args.threshold,
            )
        elif args.mixed_sampling:
            batch_rays_o, batch_rays_d, target_rgb = sample_mixed_rays_from_one_image(
                images=images,
                transform_matrixs=transform_matrixs,
                focal=focal,
                H=H,
                W=W,
                num_rays=args.num_rays,
                threshold=args.threshold,
            )
        else:
            batch_rays_o, batch_rays_d, target_rgb = sample_random_rays_from_one_image(
                images=images,
                transform_matrixs=transform_matrixs,
                focal=focal,
                H=H,
                W=W,
            num_rays=args.num_rays,
        )

        # Render predicted RGB
        result = render_rays(
            model=model,
            rays_o=batch_rays_o,
            rays_d=batch_rays_d,
            near=args.near,
            far=args.far,
            num_samples=args.num_samples,
            randomized=True,
            white_background=True,
        )

        pred_rgb = result["rgb"]

        # Mean Squared Error image reconstruction loss
        loss = F.mse_loss(pred_rgb, target_rgb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            psnr = psnr_from_mse(loss)

        loss_history.append(loss.item())
        psnr_history.append(psnr.item())

        if step % 100 == 0:
            print(
                f"Step {step:05d} | "
                f"Loss: {loss.item():.6f} | "
                f"PSNR: {psnr.item():.2f} "
                f"acc mean: {result['acc'].mean().item():.6f} | "
                f"acc max: {result['acc'].max().item():.6f}"
            )

    # 4. Save model
    ckpt_path = os.path.join(args.save_dir, "tiny_nerf.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "H": H,
            "W": W,
            "focal": focal,
            "args": vars(args),
        },
        ckpt_path,
    )
    print("Saved checkpoint to:", ckpt_path)

    # 5. Save loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history)
    plt.xlabel("Training step")
    plt.ylabel("MSE loss")
    plt.title("Tiny NeRF Training Loss")
    plt.grid(True)

    loss_path = os.path.join(args.save_dir, "loss_curve.png")
    plt.savefig(loss_path)
    plt.close()
    print("Saved loss curve to:", loss_path)

    # 6. Save PSNR curve
    plt.figure(figsize=(8, 4))
    plt.plot(psnr_history)
    plt.xlabel("Training step")
    plt.ylabel("PSNR")
    plt.title("Tiny NeRF Training PSNR")
    plt.grid(True)

    psnr_path = os.path.join(args.save_dir, "psnr_curve.png")
    plt.savefig(psnr_path)
    plt.close()
    print("Saved PSNR curve to:", psnr_path)


if __name__ == "__main__":
    args = parse_args()
    #default setup for testing
    args.data_dir = "D:/project/nerf/nerf-synthetic-dataset/nerf_synthetic/chair"
    args.split = "train"
    args.target_size = 200
    args.foreground_only = False
    args.mixed_sampling = True
    args.threshold = 0.05
    args.max_images = 50
    args.white_background = True
    args.black_background = False
    args.num_steps = 8000
    args.num_rays = 2048
    args.num_samples = 64
    args.save_dir = "D:/project/nerf/small_nerf/outputs/chair_tiny_nerf"
    #####

    main(args)