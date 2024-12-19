"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import RegressionLoss, load_model, save_model
from .datasets.road_dataset import load_data
from .metrics import PlannerMetric

# DISCLAIMER: Some of my code is inspired by online resources.

def train(
    exp_dir: str = "logs",
    model_name: str = "linear",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    transform_pipeline: str = "default",
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2, transform_pipeline=transform_pipeline)
    val_data = load_data("drive_data/val", shuffle=False)

    # create loss function and optimizer
    loss_func = RegressionLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr, 0.7)

    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}
    
    train_metric = PlannerMetric()
    val_metric = PlannerMetric()

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        train_metric.reset()
        val_metric.reset()
        for key in metrics:
            metrics[key].clear()

        model.train()

        for data in train_data:
            left = data.get("track_left").to(device)
            right = data.get("track_right").to(device)
            label = data.get("waypoints").to(device)
            label_mask = data.get("waypoints_mask").to(device)

            # TODO: implement training step
            pred = model(left, right)
            loss = loss_func(
                pred, 
                label,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_metric.add(
                pred, 
                label,
                label_mask
            )

            # raise NotImplementedError("Training step not implemented")

            global_step += 1

        epoch_train = train_metric.compute()
        epoch_train_l1 = epoch_train.get("l1_error")
        epoch_train_lat = epoch_train.get("lateral_error")
        epoch_train_long = epoch_train.get("longitudinal_error")


        # disable gradient computation and switch to evaluation mode

        with torch.inference_mode():
            model.eval()

            for data in val_data:
                left = data.get("track_left").to(device)
                right = data.get("track_right").to(device)
                label = data.get("waypoints").to(device)
                label_mask = data.get("waypoints_mask").to(device)

                # TODO: compute validation accuracy
                pred = model(left, right)
                val_metric.add(
                    pred, 
                    label,
                    label_mask
                )

        epoch_val = val_metric.compute()
        epoch_val_l1 = epoch_val.get("l1_error")
        epoch_val_lat = epoch_val.get("lateral_error")
        epoch_val_long = epoch_val.get("longitudinal_error")
        # log average train and val accuracy to tensorboard
        logger.add_scalar("train_l1", epoch_train_l1, epoch)
        logger.add_scalar("val_l1", epoch_val_l1, epoch)
        logger.add_scalar("train_lat", epoch_train_lat, epoch)
        logger.add_scalar("val_lat", epoch_val_lat, epoch)
        logger.add_scalar("train_long", epoch_train_long, epoch)
        logger.add_scalar("val_long", epoch_val_long, epoch)

        # raise NotImplementedError("Logging not implemented")

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_l1={epoch_train_l1:.4f} "
                f"train_lat={epoch_train_lat:.4f} "
                f"train_long={epoch_train_long:.4f} "
                f"\n"
                f"val_l1={epoch_val_l1:.4f} "
                f"val_lat={epoch_val_lat:.4f} "
                f"val_long={epoch_val_long:.4f} "
            )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--transform_pipeline", type=str, default="default")

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))


