from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]

class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, target)

class CombinedLoss(nn.Module):
    def forward(self, logits1: torch.Tensor, target1: torch.LongTensor, logits2: torch.Tensor, target2: torch.LongTensor) -> torch.Tensor:
        loss1 = 0.5 * nn.functional.cross_entropy(logits1, target1)
        loss2 = 0.5 * nn.functional.mse_loss(logits2, target2)
        return loss1 + loss2

class Classifier(nn.Module):
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
            if (stride != 1 or in_channels != out_channels):
                # self.skip = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
                self.skip = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.skip = nn.Identity()

        def forward(self, x):
            return self.model(x) + self.skip(x)

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # TODO: implement
        # pass
        kernel_size = 11
        padding = (kernel_size - 1) // 2

        cnn_layers = [
            nn.Conv2d(3, in_channels, kernel_size=kernel_size, stride=2, padding=padding),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        ]
        c1 = in_channels
        for _ in range(4):
            c2 = c1 * 2
            cnn_layers.append(self.Block(c1, c2, stride=2))
            c1 = c2
        cnn_layers.append(nn.Flatten())
        cnn_layers.append(nn.Linear(192, num_classes))
        # cnn_layers.append(nn.Conv2d(c1, num_classes, kernel_size = 1))
        self.network = torch.nn.Sequential(*cnn_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # TODO: replace with actual forward pass
        # logits = torch.randn(x.size(0), 6)

        # return logits
        out = self.network(z)
        return out.view(out.size(0), -1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)


class Detector(nn.Module):
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
            self.bn2 = nn.BatchNorm2d(out_channels)

            if in_channels != out_channels:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size = 1)
            else:
                self.shortcut = nn.Identity()
        
        def forward(self, x):
            shortcut = self.shortcut(x)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x += shortcut
            x = self.relu(x)
            return x
    
    class FirstBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
            self.bn2 = nn.BatchNorm2d(out_channels)
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            return x

    
    class DownBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.down = nn.Sequential(
                Detector.ResidualBlock(in_channels, out_channels),
                # nn.MaxPool2d(2)
                nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2)
            )
        def forward(self, x):
            return self.down(x)
    
    class UpBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 2, stride = 2, padding = 0)
            self.conv = Detector.ResidualBlock(in_channels, out_channels)
        
        def forward(self, x, skip):
            x = self.up(x)
            x = torch.cat([x, skip], dim = 1)
            return self.conv(x)

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super(Detector, self).__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # TODO: implement
        # pass
        self.d1 = self.FirstBlock(in_channels, 16)
        self.d2 = self.DownBlock(16, 32)
        self.d3 = self.DownBlock(32, 64)
        self.d4 = self.DownBlock(64, 128)
        self.u3 = self.UpBlock(128, 64)
        self.u2 = self.UpBlock(64, 32)
        self.u1 = self.UpBlock(32, 16)
        self.logit = nn.Conv2d(16, num_classes, kernel_size = 1)
        self.depth = nn.Conv2d(16, 1, kernel_size = 1)
        # self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # TODO: replace with actual forward pass
        # logits = torch.randn(x.size(0), 3, x.size(2), x.size(3))
        # raw_depth = torch.rand(x.size(0), x.size(2), x.size(3))

        d1 = self.d1(z)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        u3 = self.u3(d4, d3)
        u2 = self.u2(u3,d2)
        u1 = self.u1(u2, d1)
        # logits = self.relu(self.logit(u1))
        # raw_depth = self.sigmoid(self.depth(u1))
        logits = self.logit(u1)
        raw_depth = self.depth(u1)
        return logits, raw_depth

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)

        # Optional additional post-processing for depth only if needed
        depth = raw_depth.mean(dim=1)

        return pred, depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
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
            m.load_state_dict(torch.load(model_path, map_location="cpu",weights_only=True))
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
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()
