FROM pytorch/pytorch:latest

WORKDIR /workspace
RUN pip install absl-py sklearn tensorboard wandb matplotlib gpustat
