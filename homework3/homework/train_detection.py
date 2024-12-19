import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import CombinedLoss, load_model, save_model
from .datasets.road_dataset import load_data
from .metrics import DetectionMetric

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

    train_data = load_data("road_data/train", shuffle=True, batch_size=batch_size, num_workers=2, transform_pipeline=transform_pipeline)
    val_data = load_data("road_data/val", shuffle=False)

    # create loss function and optimizer
    loss_func = CombinedLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr, 0.9)

    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}
    
    train_metric = DetectionMetric()
    val_metric = DetectionMetric()

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        train_metric.reset()
        for key in metrics:
            metrics[key].clear()

        model.train()

        for data in train_data:
            img = data.get("image").to(device)
            depth = data.get("depth").to(device)
            track = data.get("track").to(device)

            # TODO: implement training step
            logit, raw_depth = model(img)
            loss = loss_func(
                logit, 
                track,
                raw_depth.mean(dim=1),
                depth
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_metric.add(logit.argmax(dim=1),
                track,
                raw_depth.mean(dim=1),
                depth
            )

            # raise NotImplementedError("Training step not implemented")

            global_step += 1

        epoch_train_acc = train_metric.compute().get("accuracy")
        epoch_train_iou = train_metric.compute().get("iou")
        epoch_train_abs = train_metric.compute().get("abs_depth_error")
        epoch_train_tp = train_metric.compute().get("tp_depth_error")


        # disable gradient computation and switch to evaluation mode

        with torch.inference_mode():
            model.eval()

            for data in val_data:
                img = data.get("image").to(device)
                depth = data.get("depth").to(device)
                track = data.get("track").to(device)

                # TODO: compute validation accuracy
                logit, raw_depth = model(img)
                val_metric.add(logit.argmax(dim=1),
                    track,
                    raw_depth.mean(dim=1),
                    depth
                )

        epoch_val_acc = val_metric.compute().get("accuracy")
        epoch_val_iou = val_metric.compute().get("iou")
        epoch_val_abs = val_metric.compute().get("abs_depth_error")
        epoch_val_tp = val_metric.compute().get("tp_depth_error")
        # log average train and val accuracy to tensorboard
        logger.add_scalar("train_acc", epoch_train_acc, epoch)
        logger.add_scalar("val_acc", epoch_val_acc, epoch)
        logger.add_scalar("train_iou", epoch_train_iou, epoch)
        logger.add_scalar("val_iou", epoch_val_iou, epoch)
        logger.add_scalar("train_abs", epoch_train_abs, epoch)
        logger.add_scalar("val_abs", epoch_val_abs, epoch)
        logger.add_scalar("train_tp", epoch_train_tp, epoch)
        logger.add_scalar("val_tp", epoch_val_tp, epoch)

        # raise NotImplementedError("Logging not implemented")

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f} "
                f"train_iou={epoch_train_iou:.4f} "
                f"val_iou={epoch_val_iou:.4f} "
                f"\n"
                f"train_abs={epoch_train_abs:.4f} "
                f"val_abs={epoch_val_abs:.4f} "
                f"train_tp={epoch_train_tp:.4f} "
                f"val_tp={epoch_val_tp:.4f} "
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


