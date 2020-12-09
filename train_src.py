import os
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from absl import app, flags, logging
from torch.multiprocessing import Process

from dataloader import get_src_dataloader
from models.get_model import get_model
from utils import InfIterator, Logger, backup_code, check_args, get_optimizier

FLAGS = flags.FLAGS
# Training
flags.DEFINE_integer("num_run", 1, "The number of meta-training runs")
flags.DEFINE_integer("batch_size", 128, "Batch size")
flags.DEFINE_integer("train_steps", 2000, "Total training steps for a single run")
flags.DEFINE_enum("lr_schedule", "step_lr", ["step_lr", "cosine_lr"], "lr schedule")
flags.DEFINE_enum("opt", "adam", ["adam", "sgd", "rmsprop"], "optimizer")
flags.DEFINE_float("lr", 1e-3, "Learning rate")

# Model
flags.DEFINE_string("model", "resnet20", "Model")
flags.DEFINE_integer("clipval", 100, "clip value for perturb module")

# Data
flags.DEFINE_integer("img_size", 32, "Image size")
flags.DEFINE_string("data", "tiny_imagenet", "Data")
flags.DEFINE_integer("num_split", 10, "The number of splits")

# Misc
flags.DEFINE_string("tblog_dir", None, "Directory for tensorboard logs")
flags.DEFINE_string("code_dir", "./codes", "Directory for backup code")
flags.DEFINE_string("save_dir", "./checkpoints", "Directory for checkpoints")
flags.DEFINE_string("exp_name", "", "Experiment name")
flags.DEFINE_integer("print_every", 200, "Print period")
flags.DEFINE_integer("save_every", 1000, "Save period")
flags.DEFINE_list("gpus", "", "GPUs to use")
flags.DEFINE_string("port", "123456", "Port number for multiprocessing")
flags.DEFINE_integer("num_workers", 1, "The number of workers for dataloading")


def share_grads(params):
    tensors = torch.cat([p.grad.view(-1) for p in params])
    dist.all_reduce(tensors)
    tensors /= dist.get_world_size()

    idx = 0
    for p in params:
        p.grad.data.copy_(tensors[idx : idx + np.prod(p.grad.shape)].view(p.grad.size()))
        idx += np.prod(p.grad.shape)


def accuracy(y, y_pred):
    with torch.no_grad():
        pred = torch.max(y_pred, dim=1)
        return 1.0 * pred[1].eq(y).sum() / y.size(0)


def train_step(model, phi, train_iter, valid_iter, theta_opt, phi_opt, clipval, device, criterion, logger):
    model.train()

    # Sample data from training set
    x, y = next(train_iter)
    x, y = x.to(device), y.to(device)
    x = x.expand(-1, 3, -1, -1)

    # Update theta
    y_pred, metrics = model(x, clipval=clipval)
    y_pred = nn.LogSoftmax(dim=1)(y_pred)
    loss = criterion(y_pred, y)

    theta_opt.zero_grad()
    loss.backward()
    gradient_norm = [torch.norm(param.grad) for name, param in model.named_parameters() if "perturb" not in name]
    theta_opt.step()

    # Meter logs
    logger.meter("train", "ce_loss", loss)
    logger.meter("train", "accuracy", accuracy(y, y_pred))
    for i, metric in enumerate(metrics):
        if metric is None:
            continue
        for name, value in metric.items():
            logger.meter(f"train_{name}", f"layer_{i}", value)
    logger.meter("theta", "gradient_norm", torch.sum(torch.stack(gradient_norm)))

    # Stop accumulating running statistics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    # Sample data from validation set
    x, y = next(valid_iter)
    x, y = x.to(device), y.to(device)
    x = x.expand(-1, 3, -1, -1)

    # Update phi
    y_pred, metrics = model(x, clipval=clipval)
    y_pred = nn.LogSoftmax(dim=1)(y_pred)
    loss = criterion(y_pred, y)

    phi_opt.zero_grad()
    loss.backward()
    gradient_norm = [torch.norm(param.grad) for param in phi]
    share_grads(phi)
    phi_opt.step()

    # Meter logs
    logger.meter("valid", "ce_loss", loss)
    logger.meter("valid", "accuracy", accuracy(y, y_pred))
    for i, metric in enumerate(metrics):
        if metric is None:
            continue
        for name, value in metric.items():
            logger.meter(f"valid_{name}", f"layer_{i}", value)

    idx = 0
    for name, _ in model.named_parameters():
        if "perturb" in name:
            logger.meter(name[8:], "gradient_norm", gradient_norm[idx])
            idx += 1
    logger.meter("all", "gradient_norm", torch.sum(torch.stack(gradient_norm)))


def test(model, test_loader, device, criterion, logger):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x = x.expand(-1, 3, -1, -1)

            y_preds = torch.stack([nn.Softmax(dim=1)(model(x)) for _ in range(5)])
            y_pred = torch.log(torch.mean(y_preds, dim=0))

            loss = criterion(y_pred, y)
            logger.meter("test", "ce_loss", loss)

            pred = torch.max(y_pred, dim=1)
            correct += pred[1].eq(y).sum()
            total += y.size(0)
    logger.meter("test", "accuracy", 1.0 * correct / total)


def run_single_process(rank, model_name, backend="nccl"):
    dist.init_process_group(backend, rank=rank, world_size=FLAGS.num_split)
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpus[rank]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloader
    # Don't test in meta-training stage
    train_loader, valid_loader, test_loader, num_classes = get_src_dataloader(
        name=FLAGS.data,
        split=rank,
        total_split=FLAGS.num_split,
        img_size=FLAGS.img_size,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
    )
    train_iter = InfIterator(train_loader)
    valid_iter = InfIterator(valid_loader)

    if rank == 0:
        logging.info(f"Train dataset: {len(train_loader)} batches")
        logging.info(f"Total {FLAGS.train_steps//len(train_loader)} epochs")
        logging.info(f"Valid dataset: {len(valid_loader)} batches")
        logging.info(f"Total {FLAGS.train_steps//len(valid_loader)} epochs")
        # logging.info(f"Test dataset: {len(test_loader)} batches")

    for run in range(FLAGS.num_run):
        if rank == 0:
            logging.info(f"Run #{run+1}")
        # Model
        model = get_model(model_name=model_name, num_classes=num_classes, img_size=FLAGS.img_size, do_perturb=True)
        model = model.to(device)

        theta = [p for name, p in model.named_parameters() if "perturb" not in name]
        phi = [p for name, p in model.named_parameters() if "perturb" in name]

        # Synchronize phi at the beginning
        for p in phi:
            dist.all_reduce(p.data)
            p.data /= FLAGS.num_split

        # Criterion
        criterion = nn.NLLLoss().to(device)

        # Optimizers
        theta_opt = get_optimizier(FLAGS.opt, FLAGS.lr, theta)
        phi_opt = get_optimizier(FLAGS.opt, FLAGS.lr, phi)

        # Logger
        logger = Logger(
            exp_name=FLAGS.exp_name,
            log_dir=FLAGS.log_dir,
            save_dir=FLAGS.save_dir,
            exp_suffix=f"run_{run}_src/split_{rank+1}",
            print_every=FLAGS.print_every,
            save_every=FLAGS.save_every,
            total_step=FLAGS.train_steps,
            print_to_stdout=(rank == 0),
            use_wandb=True,
            wnadb_project_name="l2p",
            wandb_tags=[f"split_{rank+1}"],
            wandb_config=FLAGS,
        )
        logger.register_model_to_save(model, "model")
        logger.register_model_to_save(model.perturb, "perturb")

        # Training Loop
        logger.start()
        for i in range(FLAGS.train_steps):
            train_step(model, phi, train_iter, valid_iter, theta_opt, phi_opt, FLAGS.clipval, device, criterion, logger)
            if (i + 1) % FLAGS.save_every == 0:
                pass
            logger.step()
        logger.finish()


def run_multi_process(argv):
    del argv
    check_args(FLAGS)
    backup_code(os.path.join(FLAGS.code_dir, FLAGS.exp_name, datetime.now().strftime("%m-%d-%H-%M-%S")))

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = FLAGS.port
    os.environ["WANDB_SILENT"] = "true"
    processes = []

    for rank in range(FLAGS.num_split):
        p = Process(target=run_single_process, args=(rank, FLAGS.model))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    app.run(run_multi_process)
