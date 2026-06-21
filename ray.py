import torch

def get_rays(H, W, focal, c2w):
    """
    Generate camera rays for all pixels in one image.

    Args:
        H: int
            Image height.

        W: int
            Image width.

        focal: float
            Focal length in pixels.

        c2w: torch.Tensor
            Camera-to-world matrix, shape [4, 4].

    Returns:
        rays_o: torch.Tensor
            Ray origins, shape [H, W, 3].

        rays_d: torch.Tensor
            Ray directions, shape [H, W, 3].
    """

    device = c2w.device
    #get pixel coordinates  
    j,i = torch.meshgrid(
        torch.arange(H, dtype=torch.float32,device=device),
        torch.arange(W, dtype=torch.float32,device=device),
        indexing="ij"
    )
    #pixel coordinates to camera coordinates
    # (y is downward in pixel coordinates but upward in camera coordinates)
    #(z is forward in camera coordinates but backward in pixel coordinates)
    dirs = torch.stack(
        [
            (i - W * 0.5) / focal,          # x direction, shape [H, W]
            -(j - H * 0.5) / focal,         # y direction, shape [H, W]
            -torch.ones_like(i),            # z direction, shape [H, W]
        ],
        dim=-1,
    )  # [H, W, 3]
    rays_d = dirs @ c2w[:3, :3].T
    rays_o = c2w[:3, 3].expand(rays_d.shape)

    return rays_o, rays_d


def sample_random_rays(images, transform_matrixs, focal, H, W, num_rays):
    """
    Randomly sample rays and target RGB values from the training images.

    Args:
        images: torch.Tensor
            Shape [N, H, W, 3].

        transform_matrixs: torch.Tensor
            Shape [N, 4, 4].

        focal: float
            Focal length in pixels.

        H: int
            Image height.

        W: int
            Image width.

        num_rays: int
            Number of rays to sample.

    Returns:
        batch_rays_o: torch.Tensor
            Shape [num_rays, 3].

        batch_rays_d: torch.Tensor
            Shape [num_rays, 3].

        target_rgb: torch.Tensor
            Shape [num_rays, 3].
    """

    device = images.device
    num_images = images.shape[0]

    # the key step: randomly choose an image, then randomly choose a pixel in that image
    image_indices = torch.randint(
        low=0,
        high=num_images,
        size=(num_rays,),
        device=device
    )

    # Randomly choose pixel coordinates.
    ys = torch.randint(
        low=0,
        high=H,
        size=(num_rays,),
        device=device
    )

    xs = torch.randint(
        low=0,
        high=W,
        size=(num_rays,),
        device=device
    )

    batch_rays_o = []
    batch_rays_d = []
    target_rgb = []

    # Simple version: loop over sampled rays.
    # This is easy to understand and fine for a small experiment.
    for img_idx, y, x in zip(image_indices, ys, xs):
        c2w = transform_matrixs[img_idx]

        rays_o, rays_d = get_rays(H, W, focal, c2w)

        batch_rays_o.append(rays_o[y, x])
        batch_rays_d.append(rays_d[y, x])
        target_rgb.append(images[img_idx, y, x])

    batch_rays_o = torch.stack(batch_rays_o, dim=0)
    batch_rays_d = torch.stack(batch_rays_d, dim=0)
    target_rgb = torch.stack(target_rgb, dim=0)

    return batch_rays_o, batch_rays_d, target_rgb


def sample_random_rays_from_one_image(images, transform_matrixs, focal, H, W, num_rays):
    """
    Randomly choose one image, then randomly sample rays from this image.
    This is more efficient than sampling across all images one by one.
    """

    device = images.device
    num_images = images.shape[0]

    img_idx = torch.randint(
        low=0,
        high=num_images,
        size=(1,),
        device=device
    ).item()

    target_img = images[img_idx]
    c2w = transform_matrixs[img_idx]

    rays_o, rays_d = get_rays(H, W, focal, c2w)

    coords = torch.stack(
        torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        ),
        dim=-1
    )  # [H, W, 2]

    coords = coords.reshape(-1, 2)  # [H*W, 2]

    select_inds = torch.randperm(coords.shape[0], device=device)[:num_rays]
    select_coords = coords[select_inds]

    ys = select_coords[:, 0]
    xs = select_coords[:, 1]

    batch_rays_o = rays_o[ys, xs]
    batch_rays_d = rays_d[ys, xs]
    target_rgb = target_img[ys, xs]

    return batch_rays_o, batch_rays_d, target_rgb


def sample_foreground_rays_from_one_image(
    images,
    transform_matrixs,
    focal,
    H,
    W,
    num_rays,
    threshold=0.05,
):
    """
    Only sample rays from foreground pixels of one image.

    images: [N, H, W, 3]
    transform_matrixs: [N, 4, 4]

    threshold:
        how far a pixel should be from white [1,1,1]
        to be considered foreground
    """
    device = images.device
    num_images = images.shape[0]    
    img_idx = torch.randint(
        low=0,
        high=num_images,
        size=(1,),
        device=device
    ).item()
    target_img = images[img_idx]          # [H, W, 3]
    c2w = transform_matrixs[img_idx]      # [4, 4]

    rays_o, rays_d = get_rays(H, W, focal, c2w)   # [H, W, 3]

    # foreground mask: pixels not close to white
    diff_from_white = torch.mean(torch.abs(target_img - 1.0), dim=-1)  # [H, W]
    fg_mask = diff_from_white > threshold

    fg_coords = torch.nonzero(fg_mask, as_tuple=False)  # [num_fg, 2], each row = [y, x],return the coordinates of foreground pixels

    if fg_coords.shape[0] == 0:
        raise RuntimeError("No foreground pixels found. Try lowering threshold.")

    # random sample from foreground coordinates
    select_inds = torch.randint(
        low=0,
        high=fg_coords.shape[0],
        size=(num_rays,),
        device=device,
    )

    selected_coords = fg_coords[select_inds]

    ys = selected_coords[:, 0]
    xs = selected_coords[:, 1]

    batch_rays_o = rays_o[ys, xs]
    batch_rays_d = rays_d[ys, xs]
    target_rgb = target_img[ys, xs]

    return batch_rays_o, batch_rays_d, target_rgb


def sample_mixed_rays_from_one_image(
    images,
    transform_matrixs,
    focal,
    H,
    W,
    num_rays,
   
    fg_ratio=0.7,
    threshold=0.05,
):
    device = images.device
    num_images = images.shape[0]    
    img_idx = torch.randint(
        low=0,
        high=num_images,
        size=(1,),
        device=device
    ).item()
    target_img = images[img_idx]
    c2w = transform_matrixs[img_idx]

    rays_o, rays_d = get_rays(H, W, focal, c2w)

    diff_from_white = torch.max(
        torch.abs(target_img - 1.0),
        dim=-1
    ).values

    fg_mask = diff_from_white > threshold
    bg_mask = ~fg_mask

    fg_coords = torch.nonzero(fg_mask, as_tuple=False)
    bg_coords = torch.nonzero(bg_mask, as_tuple=False)

    if fg_coords.shape[0] == 0:
        raise RuntimeError("No foreground pixels found.")

    num_fg = int(num_rays * fg_ratio)
    num_bg = num_rays - num_fg

    fg_inds = torch.randint(
        0,
        fg_coords.shape[0],
        size=(num_fg,),
        device=device,
    )
    selected_fg = fg_coords[fg_inds]

    if bg_coords.shape[0] > 0 and num_bg > 0:
        bg_inds = torch.randint(
            0,
            bg_coords.shape[0],
            size=(num_bg,),
            device=device,
        )
        selected_bg = bg_coords[bg_inds]
        selected_coords = torch.cat([selected_fg, selected_bg], dim=0)
    else:
        selected_coords = selected_fg

    # shuffle, avoid foreground always first. randpermute ,no replacement,randint with replacement
    perm = torch.randperm(selected_coords.shape[0], device=device)
    selected_coords = selected_coords[perm]

    ys = selected_coords[:, 0]
    xs = selected_coords[:, 1]

    batch_rays_o = rays_o[ys, xs]
    batch_rays_d = rays_d[ys, xs]
    target_rgb = target_img[ys, xs]

    return batch_rays_o, batch_rays_d, target_rgb