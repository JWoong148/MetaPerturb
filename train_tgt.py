import os

import torch
import torch.nn as nn
from absl import app, flags, logging

from dataloader import get_tgt_dataloader
from models.get_model import get_model
from utils import check_args, InfIterator, Logger, get_optimizier, get_scheduler

FLAGS = flags.FLAGS
# Training
flags.DEFINE_integer("batch_size", 128, "Batch size")
flags.DEFINE_integer("train_steps", 10000, "Total training steps")
flags.DEFINE_enum("lr_schedule", "step_lr", ["step_lr", "cosine_lr"], "lr schedule")
flags.DEFINE_enum("opt", "adam", ["adam", "sgd", "rmsprop"], "optimizer")
flags.DEFINE_float("lr", 1e-3, "Learning rate")
flags.DEFINE_float("noise_coeff", 1.0, "Noise coefficient for noise annealing (Not used)")

# Model
flags.DEFINE_string("model", "resnet20", "Model")
flags.DEFINE_integer("clipval", 100, "clip value for perturb module")

# Data
flags.DEFINE_integer("img_size", 32, "Image size")
flags.DEFINE_string("data", "stl10", "Data")

# Misc
flags.DEFINE_string("tblog_dir", None, "Directory for tensorboard logs")
flags.DEFINE_string("code_dir", "./codes", "Directory for backup code")
flags.DEFINE_string("save_dir", "./checkpoints", "Directory for checkpoints")
flags.DEFINE_bool("fine_tune", False, "Fine tune or not")
flags.DEFINE_string("src_name", "", "Source name to use")
flags.DEFINE_string("src_steps", "10000", "Source training steps")
flags.DEFINE_string("exp_name", "", "Experiment name")
flags.DEFINE_integer("print_every", 200, "Print period")
flags.DEFINE_string("gpus", "", "GPUs to use")
flags.DEFINE_integer("num_workers", 3, "The number of workers for dataloading")


def accuracy(y, y_pred):
    with torch.no_grad():
        pred = torch.max(y_pred, dim=1)
        return 1.0 * pred[1].eq(y).sum() / y.size(0)


def train_step(model, noise_coeff, train_iter, theta_optimizer, device, criterion, logger):
    model.train()

    x, y = next(train_iter)
    x, y = x.to(device), y.to(device)
    x = x.expand(-1, 3, -1, -1)

    y_pred, metrics = model(x, clipval=FLAGS.clipval, noise_coeff=noise_coeff)
    y_pred = nn.LogSoftmax(dim=1)(y_pred)
    loss = criterion(y_pred, y)

    theta_optimizer.zero_grad()
    loss.backward()
    theta_optimizer.step()

    # Meter logs
    logger.meter("train", "ce_loss", loss)
    logger.meter("train", "accuracy", accuracy(y, y_pred))
    for i, metric in enumerate(metrics):
        if metric is None:
            continue
        for name, value in metric.items():
            logger.meter(f"train_{name}", f"layer_{i}", value)


def test(model, noise_coeff, test_loader, num_samples, device, criterion, logger):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x = x.expand(-1, 3, -1, -1)

            y_preds = torch.stack([nn.Softmax(dim=1)(model(x, noise_coeff=noise_coeff)[0]) for _ in range(num_samples)])
            y_pred = torch.log(torch.mean(y_preds, dim=0))

            loss = criterion(y_pred, y)
            logger.meter("test", "ce_loss", loss)

            pred = torch.max(y_pred, dim=1)
            correct += pred[1].eq(y).sum()
            total += y.size(0)

    logger.meter("test", "accuracy", 1.0 * correct / total)
    return (1.0 * correct / total).item()


def main(argv):
    del argv
    check_args(FLAGS)

    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpus
    os.environ["WANDB_SILENT"] = "true"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloader
    train_loader, test_loader, num_classes = get_tgt_dataloader(
        name=FLAGS.data, img_size=FLAGS.img_size, batch_size=FLAGS.batch_size, num_workers=FLAGS.num_workers
    )
    train_iter = InfIterator(train_loader)
    logging.info(f"Train dataset: {len(train_loader)} batches")
    logging.info(f"Total {FLAGS.train_steps//len(train_loader)} epochs")
    logging.info(f"Test dataset: {len(test_loader)} batches")

    # Model
    model = get_model(model_name=FLAGS.model, num_classes=num_classes, img_size=FLAGS.img_size, do_perturb=True)
    model = model.to(device)
    if FLAGS.fine_tune:
        src_model_state_dict = torch.load(f"{FLAGS.save_dir}/tgt/scratch_TIN_{FLAGS.model}/model_100000.pth")
        state_dict_wo_bn = {
            name: value for name, value in src_model_state_dict.items() if "bn" not in name and "fc" not in name
        }
        model.load_state_dict(state_dict_wo_bn, strict=False)
        logging.info(f"Model is loaded from {FLAGS.save_dir}/tgt/scratch_TIN_{FLAGS.model}/model_100000.pth")

    src_perturb_state_dict = torch.load(f"{FLAGS.save_dir}/{FLAGS.src_name}_src/split_1/perturb_{FLAGS.src_steps}.pth")
    model.perturb.load_state_dict(src_perturb_state_dict)
    logging.info(
        f"Perturb module is loaded from {FLAGS.save_dir}/{FLAGS.src_name}_src/split_1/perturb_{FLAGS.src_steps}.pth"
    )

    theta = [p for name, p in model.named_parameters() if "perturb" not in name]

    # Optimizer
    theta_opt = get_optimizier(FLAGS.opt, FLAGS.lr, theta)

    # Scheduler
    scheduler = get_scheduler(FLAGS.lr_schedule, theta_opt, FLAGS.train_steps)

    # Criterion
    criterion = nn.NLLLoss().to(device)

    # Logger
    logger = Logger(
        exp_name=FLAGS.exp_name,
        log_dir=FLAGS.log_dir,
        save_dir=FLAGS.save_dir,
        exp_suffix="tgt",
        print_every=FLAGS.print_every,
        save_every=FLAGS.train_steps,
        total_step=FLAGS.train_steps,
        use_wandb=True,
        wnadb_project_name="l2p",
        wandb_config=FLAGS,
    )
    logger.register_model_to_save(model, "model")
    logger.register_model_to_save(model.perturb, "perturb")

    # Training Loop
    logger.start()
    for step in range(1, FLAGS.train_steps + 1):
        # Noise annealing
        noise_coeff = FLAGS.noise_coeff * min(1.0, step / int(FLAGS.train_steps * 0.4))
        train_step(model, noise_coeff, train_iter, theta_opt, device, criterion, logger)
        scheduler.step()
        if step % FLAGS.print_every == 0:
            test(model, noise_coeff, test_loader, 5, device, criterion, logger)

        logger.step()

    # Test final model
    final_results = test(model, FLAGS.noise_coeff, test_loader, 100, device, criterion, logger)
    logger.write_log_individually("final_accuracy", final_results, FLAGS.train_steps)
    print(f"Final accuracy: {final_results}")
    logger.finish()


if __name__ == "__main__":
    app.run(main)
