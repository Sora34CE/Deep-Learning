from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]

class MSELoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        return nn.functional.mse_loss(logits, target)

class RegressionLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_fn = nn.L1Loss()
        return loss_fn(logits, target)


class MLPPlanner(nn.Module):
    # python3 -m homework.mlp_planner --model_name mlp_planner --lr 0.0001 --num_epoch 1 
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        embed_dim = 128,
        n_heads = 4,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        self.mha = nn.MultiheadAttention(embed_dim, n_heads)
        self.linear1 = nn.Linear(embed_dim, n_waypoints)
        self.linear2 = nn.Linear(n_track, 2)
        self.linear3 = nn.Linear(4, embed_dim)
        self.linear4 = nn.Linear(2, embed_dim)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # raise NotImplementedError
        track = torch.cat([track_left, track_right], dim=1)
        track = track.view(track.size(0), self.n_track, -1)
        track = self.linear3(track)
        track_left = self.linear4(track_left)
        track_right = self.linear4(track_right)
        combined_features, _ = self.mha(track, track_left, track_right)
        mlp_out = self.linear1(combined_features)
        mlp_out = mlp_out.view(mlp_out.size(0), mlp_out.size(2), -1)
        mlp_out = self.linear2(mlp_out)
        return mlp_out


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        self.query_embed = nn.Embedding(n_waypoints, d_model)
        self.weights = self.query_embed.weight

        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True
            ), num_layers=n_layers)
        
        self.lin1 = nn.Linear(d_model, n_waypoints)
        self.lin2 = nn.Linear(40, 2)
        self.lin3 = nn.Linear(2, d_model)
        self.lin4 = nn.Linear(256, 64)
        self.lin5 = nn.Linear(10, 2)

        self.lin6 = nn.Linear(20, 3)
        self.lin7 = nn.Linear(2, d_model)
        self.lin8 = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # raise NotImplementedError
        track = torch.cat([track_left, track_right], dim=1)
        track_weight = self.weights.unsqueeze(0).repeat(track.size(0), 1, 1)
        track = self.lin6(track.view(track.size(0), track.size(2), -1))
        track = self.lin7(track.view(track.size(0), track.size(2), -1))

        waypoints = self.decoder(track_weight, track)
        waypoints = self.lin8(waypoints)

        return waypoints


class CNNPlanner(torch.nn.Module):
    # python3 -m homework.train_planner --model_name cnn_planner --lr 0.001 --num_epoch 20
    class Block(torch.nn.Module):
        def __init__(self, in_channels, out_channels, stride):
            super().__init__()
            kernel_size = 3
            padding = (kernel_size - 1) // 2

            self.model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
            self.dropout = nn.Dropout2d(0.1)
            self.relu = nn.ReLU()
            if (stride != 1 or in_channels != out_channels):
                self.skip = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.skip = nn.Identity()

        def forward(self, x):
            # return self.model(x) + self.skip(x)
            out = self.model(x)
            out = self.dropout(out)
            out += self.skip(x)
            out = self.relu(out)
            return out
    
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        kernel_size = 11
        padding = (kernel_size - 1) // 2

        cnn_layers = [
            nn.Conv2d(3, n_waypoints, kernel_size=kernel_size, stride=2, padding=padding),
            nn.BatchNorm2d(n_waypoints),
            nn.ReLU(),
        ]
        c1 = n_waypoints
        for _ in range(4):
            c2 = c1 * 2
            cnn_layers.append(self.Block(c1, c2, stride=2))
            c1 = c2
            
        self.network = torch.nn.Sequential(*cnn_layers)

        self.linear = nn.Linear(576, 6)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        # print(x.shape)
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # raise NotImplementedError
        out = self.network(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out.view(out.size(0), self.n_waypoints, 2)


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024

# embed_dim = 126
# num_heads = 4


# model = MLPPlanner()
# k = torch.rand(16, 10, 2)
# v = torch.rand(16, 10, 2)
# output = model(k, v)
# print(output.shape)

# model = CNNPlanner()
# img = torch.rand(128, 3, 96, 128)
# output = model(img)
# print("Shape:", output.shape)