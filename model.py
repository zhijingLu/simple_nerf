import torch
import torch.nn as nn
import torch.nn.functional as F


def positional_encoding(x, num_freqs):
    """
    Apply positional encoding to input coordinates.

    Args:
        x: torch.Tensor
            Shape [..., input_dim].
            For 3D points, input_dim = 3.
            For ray directions, input_dim = 3.

        num_freqs: int
            Number of frequency bands.

    Returns:
        encoded: torch.Tensor
            Shape [..., input_dim * (1 + 2 * num_freqs)].
    """

    encoded = [x]

    for i in range(num_freqs):
        freq = 2.0 ** i
        encoded.append(torch.sin(freq * x))
        encoded.append(torch.cos(freq * x))

    encoded = torch.cat(encoded, dim=-1)

    return encoded


class NeRFMLP(nn.Module):
    """
    small NeRF MLP.

    Input:
        3D point position x
        ray direction d

    Output:
        rgb:   [N, 3]
        sigma: [N, 1]
    """

    def __init__(
        self,
        pos_freqs=10,
        dir_freqs=4,
        hidden_dim=256,
        num_layers=8,
    ):
        super().__init__()

        self.pos_freqs = pos_freqs #10 group 2^0, 2^1, 2^2, ..., 2^9
        self.dir_freqs = dir_freqs #viewing direction 4 group 2^0, 2^1, 2^2, 2^3

        # Positional encoding dimension:
        # original 3 dims + sin/cos for each frequency
        self.pos_dim = 3 * (1 + 2 * pos_freqs) #sin(freq * x), sin(freq * y), sin(freq * z)
                                               #cos(freq * x), cos(freq * y), cos(freq * z)
                                               #+ original x, y, z
                                               #=3 + 2 * 3 * pos_freqs
        self.dir_dim = 3 * (1 + 2 * dir_freqs)

        # Point-processing MLP
        # self.fc1 = nn.Linear(self.pos_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.pts_linears = nn.ModuleList()
         # first layer
        self.pts_linears.append(nn.Linear(self.pos_dim, hidden_dim))
        for i in range(1, num_layers):
            if i == 4:
                # skip connection: concatenate encoded position again
                self.pts_linears.append(nn.Linear(hidden_dim + self.pos_dim, hidden_dim))
            else:
                self.pts_linears.append(nn.Linear(hidden_dim, hidden_dim))

        # Density head
        self.sigma_head = nn.Linear(hidden_dim, 1)

        # Feature layer before color prediction
        self.feature_head = nn.Linear(hidden_dim, hidden_dim)

        # Color head uses both point feature and viewing direction
        self.rgb_fc1 = nn.Linear(hidden_dim + self.dir_dim, hidden_dim // 2) # hidden_dim // 2 for smaller model
        self.rgb_fc2 = nn.Linear(hidden_dim // 2, 3)

    def forward(self, points, directions):
        """
        Args:
            points: torch.Tensor
                Shape [N, 3], sampled 3D points.

            directions: torch.Tensor
                Shape [N, 3], ray directions.
                Usually one direction is repeated for all sampled points on the same ray.

        Returns:
            rgb: torch.Tensor
                Shape [N, 3], values in [0, 1].

            sigma: torch.Tensor
                Shape [N, 1], density values >= 0.
        """

        # Normalize directions for stability
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)

        # Positional encoding
        encoded_points = positional_encoding(points, self.pos_freqs)
        encoded_dirs = positional_encoding(directions, self.dir_freqs)
        h = encoded_points
        for i, layer in enumerate(self.pts_linears):
            if i == 4:
                h = torch.cat([h, encoded_points], dim=-1)

            h = F.relu(layer(h))

        # # MLP for density and feature
        # h = F.relu(self.fc1(encoded_points)) #relu for non-linearity ,  
        #                                      # or it will be linear model and cannot learn complex function, hidden_dim is the number of neurons in each hidden layer
        # h = F.relu(self.fc2(h))
        # h = F.relu(self.fc3(h))
        # h = F.relu(self.fc4(h))

        # sigma should be non-negative
        sigma = F.softplus(self.sigma_head(h)) # density values >= 0

        # feature for RGB prediction
        features = self.feature_head(h)

        # RGB depends on both 3D position and viewing direction
        h_rgb = torch.cat([features, encoded_dirs], dim=-1)
        h_rgb = F.relu(self.rgb_fc1(h_rgb)) 
        rgb = torch.sigmoid(self.rgb_fc2(h_rgb)) # RGB values in [0, 1]

        return rgb, sigma


if __name__ == "__main__":
    # Simple test
    model = NeRFMLP(
        pos_freqs=10,
        dir_freqs=4,
        hidden_dim=128,
    )

    points = torch.randn(1024, 3)
    directions = torch.randn(1024, 3)

    rgb, sigma = model(points, directions)

    print("rgb shape:", rgb.shape)
    print("sigma shape:", sigma.shape)
    print("rgb min/max:", rgb.min().item(), rgb.max().item())
    print("sigma min/max:", sigma.min().item(), sigma.max().item())