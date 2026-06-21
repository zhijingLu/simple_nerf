import os
import argparse
import torch
import numpy as np
from PIL import Image

from dataset.dataset_loader import load_blender_data
from ray import get_rays
from model import NeRFMLP
from render import render_rays


def parse_args():
    parser = argparse.ArgumentParser(description="Render an image using trained Tiny NeRF")

    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to Blender scene folder, e.g. data/lego"
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="outputs/tiny_nerf.pth",
        help="Path to trained checkpoint"
    )

    parser.add_argument(
        "--target_size",
        type=int,
        default=100,
        help="Image size used during training"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to render"
    )

    parser.add_argument(
        "--image_index",
        type=int,
        default=0,
        help="Which image pose to render"
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=64,
        help="Number of samples along each ray"
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=4096,
        help="Number of rays rendered at once"
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
        "--save_path",
        type=str,
        default="outputs/rendered.png",
        help="Path to save rendered image"
    )

    return parser.parse_args()


def save_image(rgb, save_path):
    """
    rgb: [H, W, 3], values in [0, 1]
    """
    rgb = rgb.detach().cpu().numpy()
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = (rgb * 255).astype(np.uint8)

    img = Image.fromarray(rgb)
    img.save(save_path)


def main(args):
    

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 1. Load checkpoint
    checkpoint = torch.load(args.ckpt_path, map_location=device)

    H = checkpoint["H"]
    W = checkpoint["W"]
    focal = checkpoint["focal"]

    print("Loaded checkpoint")
    print("H, W:", H, W)
    print("focal:", focal)

    # 2. Load data to get camera pose and ground truth image
    images, transform_matrixs, _, _, _ = load_blender_data(
        data_dir=args.data_dir,
        split=args.split,
        target_size=args.target_size,
        white_background=True,
        max_images=1,
    )

    images = images.to(device)
    transform_matrixs = transform_matrixs.to(device)
  
    target_img = images[args.image_index]
    c2w = transform_matrixs[args.image_index]

    # 3. Build model
    model = NeRFMLP(
        pos_freqs=10,
        dir_freqs=4,
        hidden_dim=256,
        num_layers=8,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 4. Generate rays for the full image
    rays_o, rays_d = get_rays(H, W, focal, c2w)

    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)

    print("Total rays:", rays_o.shape[0])

    # 5. Render rays in chunks
    rendered_rgb_list = []

    with torch.no_grad():
        for i in range(0, rays_o.shape[0], args.chunk_size):
            rays_o_chunk = rays_o[i:i + args.chunk_size]
            rays_d_chunk = rays_d[i:i + args.chunk_size]

            result = render_rays(
                model=model,
                rays_o=rays_o_chunk,
                rays_d=rays_d_chunk,
                near=args.near,
                far=args.far,
                num_samples=args.num_samples,
                randomized=False,
                white_background=True,
            )

            rendered_rgb_list.append(result["rgb"])

            print(f"Rendered rays {i} to {min(i + args.chunk_size, rays_o.shape[0])}")

    rendered_rgb = torch.cat(rendered_rgb_list, dim=0)
    rendered_img = rendered_rgb.reshape(H, W, 3)

    # 6. Save rendered image
    save_image(rendered_img, args.save_path)
    print("Saved rendered image to:", args.save_path)

    # 7. Save target image for comparison
    target_save_path = args.save_path.replace(".png", "_target.png")
    save_image(target_img, target_save_path)
    print("Saved target image to:", target_save_path)


if __name__ == "__main__":
    args = parse_args() 
    args.data_dir = "D:/project/nerf/nerf-synthetic-dataset/nerf_synthetic/chair"
    args.split = "test"
    args.ckpt_path = "D:/project/nerf/small_nerf/outputs/chair_tiny_nerf/tiny_nerf.pth"
    args.target_size = 200
    args.image_index = 0
    args.save_path= "D:/project/nerf/small_nerf/outputs/chair_tiny_nerf/rendered_0.png"

    main(args)