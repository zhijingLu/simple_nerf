import torch


def sample_points_along_rays(
    rays_o,
    rays_d,
    near=2.0,
    far=6.0,
    num_samples=64,
    randomized=True,
):
    """
    Sample 3D points along each ray.

    Args:
        rays_o: torch.Tensor
            Ray origins, shape [num_rays, 3].

        rays_d: torch.Tensor
            Ray directions, shape [num_rays, 3].

        near: float
            Near bound of sampling.

        far: float
            Far bound of sampling.

        num_samples: int
            Number of sampled points per ray.

        randomized: bool
            If True, use stratified random sampling.
            If False, use deterministic uniform sampling.

    Returns:
        points: torch.Tensor
            Sampled 3D points, shape [num_rays, num_samples, 3].

        z_vals: torch.Tensor
            Depth values along each ray, shape [num_rays, num_samples].
    """

    device = rays_o.device
    num_rays = rays_o.shape[0]

    # Uniform depth values between near and far.
    z_vals = torch.linspace(
        near,
        far,
        num_samples,
        device=device,
    )  # [num_samples]

    # Expand to all rays.
    z_vals = z_vals.expand(num_rays, num_samples)  # [num_rays, num_samples]

    if randomized:
        # Stratified sampling.
        # Instead of always sampling fixed points, sample a random point inside each interval.
        """
        for example: z_vals = [2, 3, 4, 5, 6]
        mids= [2.5, 3.5, 4.5, 5.5]
        upper = [2.5, 3.5, 4.5, 5.5, 6]
        lower = [2, 2.5, 3.5, 4.5, 5.5]
        """
        mids = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # [num_rays, num_samples - 1]

        upper = torch.cat([mids, z_vals[:, -1:]], dim=-1) 
        lower = torch.cat([z_vals[:, :1], mids], dim=-1)

        t_rand = torch.rand(z_vals.shape, device=device)
        z_vals = lower + (upper - lower) * t_rand #each z_val is randomly sampled in the interval [lower, upper]

    # Ray equation:
    # point = ray_origin + depth * ray_direction
    # None means add a new dimension for broadcasting, so that we can compute points for all rays and all sampled depths at once.
    # rays_o: [num_rays, 3] -> [num_rays, 1, 3]
    # rays_d: [num_rays, 3] -> [num_rays, 1, 3]
    # z_vals: [num_rays, num_samples] -> [num_rays, num_samples, 1]
    #points: [num_rays, num_samples, 3]
    points = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[:, :, None]

    return points, z_vals


def volume_render(rgb, sigma, z_vals, rays_d, white_background=True):
    """
    Composite RGB and density values along each ray using volume rendering.

    Args:
        rgb: torch.Tensor
            RGB values predicted by NeRF model.
            Shape [num_rays, num_samples, 3].

        sigma: torch.Tensor
            Density values predicted by NeRF model.
            Shape [num_rays, num_samples, 1] or [num_rays, num_samples].

        z_vals: torch.Tensor
            Depth values along rays.
            Shape [num_rays, num_samples].

        rays_d: torch.Tensor
            Ray directions.
            Shape [num_rays, 3].

        white_background: bool
            If True, composite remaining transparency onto white background.

    Returns:
        rgb_map: torch.Tensor
            Final rendered RGB color for each ray.
            Shape [num_rays, 3].

        depth_map: torch.Tensor
            Expected depth for each ray.
            Shape [num_rays].

        acc_map: torch.Tensor
            Accumulated opacity for each ray.
            Shape [num_rays].

        weights: torch.Tensor
            Contribution weight of each sampled point.
            Shape [num_rays, num_samples].
    """

    if sigma.dim() == 3:
        sigma = sigma[..., 0]  # [num_rays, num_samples]

    # Distance between adjacent samples.
    dists = z_vals[:, 1:] - z_vals[:, :-1]  # [num_rays, num_samples - 1]

    # Add a very large distance for the last sample.
    # This means the last sample represents everything behind it.
    last_dist = 1e10 * torch.ones_like(dists[:, :1])
    dists = torch.cat([dists, last_dist], dim=-1)  # [num_rays, num_samples]

    # If ray directions are not normalized, scale distances by ray length.
    # This is important for correct alpha calculation when rays are not unit length.
    # It is for real world.
    ray_norms = torch.norm(rays_d, dim=-1, keepdim=True)  # [num_rays, 1]
    dists = dists * ray_norms

    # Convert density to alpha in a distance interval.
    # That means how long the ray travels in the current point, and how much it will be occluded.
    # alpha_i = 1 - exp(-sigma_i * delta_i)

    alpha = 1.0 - torch.exp(-sigma * dists)  # [num_rays, num_samples]

    # Compute transmittance T_i. 透射率
    # The proportion of the light that remains unblocked 
    #   by preceding points before reaching the i-th sampling point.
    # T_i = product_{j < i} (1 - alpha_j)
    
    
    eps = 1e-10 # to prevent numerical issues when alpha is close to 1
    """
    for example: alpha = [0.8, 0.5, 0.9, 0.2]
        alpha_all = [1.0, 0.8, 0.5, 0.9, 0.2]
        cumprod = [1.0, 0.8, 0.4, 0.36, 0.072]

        1 position: 1
        2 position: 1 * 0.8 = 0.8
        3 position: 1 * 0.8 * 0.5 = 0.4
        4 position: 1 * 0.8 * 0.5 * 0.9 = 0.36
        5 position: 1 * 0.8 * 0.5 * 0.9 * 0.2 = 0.072
    """
    transmittance = torch.cumprod( #cumulative product
        torch.cat(
            [
                torch.ones((alpha.shape[0], 1), device=alpha.device),
                1.0 - alpha + eps,
            ],
            dim=-1,
        ),
        dim=-1,
    )[:, :-1]

    # Weight of each sampled point.
    weights = transmittance * alpha  # [num_rays, num_samples]

    # Final pixel color.
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # [num_rays, 3]

    # Expected depth.
    depth_map = torch.sum(weights * z_vals, dim=-1)  # [num_rays]

    # Accumulated opacity.
    acc_map = torch.sum(weights, dim=-1)  # [num_rays]

    if white_background:
        # If accumulated opacity is less than 1, remaining part is white background.
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, depth_map, acc_map, weights


def render_rays(
    model,
    rays_o,
    rays_d,
    near=2.0,
    far=6.0,
    num_samples=64,
    randomized=True,
    white_background=True,
):
    """
    Render RGB values for a batch of rays.

    Args:
        model:
            NeRF MLP model.
            It takes points and directions as input and returns rgb and sigma.

        rays_o: torch.Tensor
            Ray origins, shape [num_rays, 3].

        rays_d: torch.Tensor
            Ray directions, shape [num_rays, 3].

        near: float
            Near sampling bound.

        far: float
            Far sampling bound.

        num_samples: int
            Number of sampled points per ray.

        randomized: bool
            Whether to use stratified sampling.

        white_background: bool
            Whether to composite onto white background.

    Returns:
        result: dict
            {
                "rgb": rendered RGB, shape [num_rays, 3],
                "depth": rendered depth, shape [num_rays],
                "acc": accumulated opacity, shape [num_rays],
                "weights": volume rendering weights, shape [num_rays, num_samples],
                "z_vals": sampled depth values, shape [num_rays, num_samples],
            }
    """

    num_rays = rays_o.shape[0]

    # 1. Sample points along rays.
    points, z_vals = sample_points_along_rays(
        rays_o=rays_o,
        rays_d=rays_d,
        near=near,
        far=far,
        num_samples=num_samples,
        randomized=randomized,
    )
    # points: [num_rays, num_samples, 3]

    # 2. Flatten points for MLP.
    points_flat = points.reshape(-1, 3)  # [num_rays * num_samples, 3]

    # Each point along the same ray uses the same viewing direction.
    dirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    dirs_expanded = dirs[:, None, :].expand(points.shape)
    dirs_flat = dirs_expanded.reshape(-1, 3)

    # 3. Predict RGB and density.
    rgb_flat, sigma_flat = model(points_flat, dirs_flat)

    # 4. Reshape back to ray structure.
    rgb = rgb_flat.reshape(num_rays, num_samples, 3)
    sigma = sigma_flat.reshape(num_rays, num_samples, 1)

    # 5. Volume rendering.
    rgb_map, depth_map, acc_map, weights = volume_render(
        rgb=rgb,
        sigma=sigma,
        z_vals=z_vals,
        rays_d=rays_d,
        white_background=white_background,
    )

    return {
        "rgb": rgb_map,
        "depth": depth_map,
        "acc": acc_map,
        "weights": weights,
        "z_vals": z_vals,
    }


if __name__ == "__main__":
    from model import NeRFMLP

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = NeRFMLP(
        pos_freqs=10,
        dir_freqs=4,
        hidden_dim=128,
    ).to(device)

    num_rays = 1024

    rays_o = torch.zeros(num_rays, 3, device=device)
    rays_d = torch.randn(num_rays, 3, device=device)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    result = render_rays(
        model=model,
        rays_o=rays_o,
        rays_d=rays_d,
        near=2.0,
        far=6.0,
        num_samples=64,
        randomized=True,
        white_background=True,
    )

    print("rgb:", result["rgb"].shape)
    print("depth:", result["depth"].shape)
    print("acc:", result["acc"].shape)
    print("weights:", result["weights"].shape)
    print("z_vals:", result["z_vals"].shape)