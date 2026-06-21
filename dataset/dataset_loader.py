import os
import json
import argparse
import numpy as np
from PIL import Image
import torch


def load_blender_data(
    data_dir,
    split="train",
    target_size=None,
    white_background=True,
    max_images=None,
):
    """
    Load Blender NeRF synthetic dataset.

    Returns:
        images: [N, H, W, 3]
        transform_matrix:  [N, 4, 4]
        focal:  float
        H, W:   image height and width
    """

    json_path = os.path.join(data_dir, f"transforms_{split}.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Cannot find JSON file: {json_path}")

    with open(json_path, "r") as f:
        meta = json.load(f)

    camera_angle_x = meta["camera_angle_x"]
    frames = meta["frames"]

    if max_images is not None:
        frames = frames[:max_images]

    images = []
    transform_matrixs = []

    for frame in frames:
        file_path = frame["file_path"]

        # file_path is usually like "./train/r_0"
        img_path = os.path.join(data_dir, file_path + ".png")

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Cannot find image file: {img_path}")
        print("Loading image:", img_path)
        img = Image.open(img_path)
        img = np.array(img).astype(np.float32) / 255.0

        # Blender synthetic images are usually RGBA
        if img.shape[-1] == 4:
            rgb = img[..., :3]
            alpha = img[..., 3:4]

            if white_background:
                white = np.ones_like(rgb)
                img = rgb * alpha + (1.0 - alpha) * white
            else:
                img = rgb * alpha
        else:
            img = img[..., :3]

        if target_size is not None:
            img_pil = Image.fromarray((img * 255.0).astype(np.uint8))
            img_pil = img_pil.resize((target_size, target_size), Image.BILINEAR)
            img = np.array(img_pil).astype(np.float32) / 255.0

        transform_matrix = np.array(frame["transform_matrix"], dtype=np.float32)

        images.append(img)
        transform_matrixs.append(transform_matrix)

    images = np.stack(images, axis=0)
    transform_matrixs = np.stack(transform_matrixs, axis=0)

    H, W = images.shape[1], images.shape[2]

    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    images = torch.from_numpy(images).float()
    transform_matrixs = torch.from_numpy(transform_matrixs).float()

    return images, transform_matrixs, float(focal), H, W


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load Blender Synthetic NeRF dataset"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        #required=True,
        help="Path to Blender scene folder, e.g. data/lego"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to load"
    )

    parser.add_argument(
        "--target_size",
        type=int,
        default=100,  #original 800
        help="Resize images to target_size x target_size. Use 64 or 100 for TinyNeRF."
    )

    parser.add_argument(
        "--max_images",
        type=int,
        default=20,
        help="Maximum number of images to load. Use -1 to load all images."
    )

    parser.add_argument(
        "--white_background",
        action="store_true",
        help="Composite RGBA images onto white background"
    )

    parser.add_argument(
        "--black_background",
        action="store_true",
        help="Composite RGBA images onto black background"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    #default setup for testing
    args.data_dir = "D:/project/nerf/nerf-synthetic-dataset/nerf_synthetic/chair"
    args.split = "train"
    args.target_size = 100
    args.max_images = 20
    args.white_background = True
    args.black_background = False
    #####


    if args.max_images == -1:
        max_images = None
    else:
        max_images = args.max_images

    if args.black_background:
        white_background = False
    else:
        white_background = True

    images, transform_matrixs, focal, H, W = load_blender_data(
        data_dir=args.data_dir,
        split=args.split,
        target_size=args.target_size,
        white_background=white_background,
        max_images=max_images,
    )

    print("Loaded Blender data successfully.")
    print("data_dir:", args.data_dir)
    print("split:", args.split)
    print("images shape:", images.shape)
    print("transform_matrixs shape:", transform_matrixs.shape)
    print("focal:", focal)
    print("H:", H)
    print("W:", W)
    print("first transform matrix:")
    print(transform_matrixs[0])