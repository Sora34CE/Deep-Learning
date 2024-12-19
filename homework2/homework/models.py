"""
Implement the following models for classification.

Feel free to modify the arguments for each of model's __init__ function.
This will be useful for tuning model hyperparameters such as hidden_dim, num_layers, etc,
but remember that the grader will assume the default constructor!
"""

from pathlib import Path

import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Multi-class classification loss
        Hint: simple one-liner

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """
        return nn.functional.cross_entropy(logits, target)
        # raise NotImplementedError("ClassificationLoss.forward() is not implemented")


class LinearClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
    ):
        """
        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
        """
        super(LinearClassifier, self).__init__()
        self.model = nn.Linear(h * w * 3, num_classes)
        # raise NotImplementedError("LinearClassifier.__init__() is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        x = x.view(x.size(0), -1)
        return self.model(x)
        # raise NotImplementedError("LinearClassifier.forward() is not implemented")


class MLPClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
    ):
        """
        An MLP with a single hidden layer

        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
        """
        # super().__init__()
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(h * w * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        # raise NotImplementedError("MLPClassifier.__init__() is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        x = x.view(x.size(0), -1)
        return self.model(x)
        # raise NotImplementedError("MLPClassifier.forward() is not implemented")


class MLPClassifierDeep(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
    ):
        """
        An MLP with multiple hidden layers

        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int

        Hint - you can add more arguments to the constructor such as:
            hidden_dim: int, size of hidden layers
            num_layers: int, number of hidden layers
        """
        super(MLPClassifierDeep, self).__init__()
        layers = []
        hidden_sizes = [128, 128, 128]
        prev_size = h * w * 3
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, num_classes))
        self.model = nn.Sequential(*layers)
        # raise NotImplementedError("MLPClassifierDeep.__init__() is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        x = x.view(x.size(0), -1)
        return self.model(x)
        # raise NotImplementedError("MLPClassifierDeep.forward() is not implemented")


class MLPClassifierDeepResidual(nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.model = torch.nn.Sequential(
                torch.nn.Linear(in_channels, out_channels),
                torch.nn.LayerNorm(out_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(out_channels, out_channels),
                torch.nn.LayerNorm(out_channels),
                torch.nn.ReLU(),
            )
            if in_channels != out_channels:
                self.skip = torch.nn.Linear(in_channels, out_channels)
            else:
                self.skip = torch.nn.Identity()

        def forward(self, x):
            return self.skip(x) + self.model(x)

    def __init__(
        self, 
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        layer_size=[128, 128, 128]
    ):
        """
        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int

        Hint - you can add more arguments to the constructor such as:
            hidden_dim: int, size of hidden layers
            num_layers: int, number of hidden layers
        """
        super(MLPClassifierDeepResidual, self).__init__()
        layers = []
        layers.append(torch.nn.Flatten())
        c = h * w * 3
        layers.append(torch.nn.Linear(c, 128, bias=False))
        c = 128
        for s in layer_size:
            layers.append(self.Block(c, s))
            c = s
        layers.append(torch.nn.Linear(c, num_classes, bias=False))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
        # raise NotImplementedError("MLPClassifierDeepResidual.forward() is not implemented")


model_factory = {
    "linear": LinearClassifier,
    "mlp": MLPClassifier,
    "mlp_deep": MLPClassifierDeep,
    "mlp_deep_residual": MLPClassifierDeepResidual,
}


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def save_model(model):
    """
    Use this function to save your model in train.py
    """
    for n, m in model_factory.items():
        if isinstance(model, m):
            return torch.save(model.state_dict(), Path(__file__).resolve().parent / f"{n}.th")
    raise ValueError(f"Model type '{str(type(model))}' not supported")


def load_model(model_name: str, with_weights: bool = False, **model_kwargs):
    """
    Called by the grader to load a pre-trained model by name
    """
    r = model_factory[model_name](**model_kwargs)
    if with_weights:
        model_path = Path(__file__).resolve().parent / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        try:
            r.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # Limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(r)
    if model_size_mb > 10:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")
    print(f"Model size: {model_size_mb:.2f} MB")

    return r
