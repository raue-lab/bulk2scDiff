import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from VAE.VAE_model import VAE
from guided_diffusion.cell_datasets_loader import load_data
from guided_diffusion import logger


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def build_pretrained_checkpoint_paths(base_dir):
    return {
        "encoder": os.path.join(base_dir, "encoder.ckpt"),
        "decoder": os.path.join(base_dir, "decoder.ckpt"),
    }


def prepare_vae(args, pretrained_paths=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = load_data(
        data_dir=args["data_dir"],
        batch_size=args["batch_size"],
        train_vae=True,
        sample_key=args["sample_key"],
        include_sample_ids_path=args["include_sample_ids_path"],
        exclude_sample_ids_path=args["exclude_sample_ids_path"],
    )
    autoencoder = VAE(
        num_genes=args["num_genes"],
        device=device,
        seed=args["seed"],
        hidden_dim=args["hidden_dim"],
    )

    if pretrained_paths is not None:
        print(f"loading pretrained model from:\n{args['state_dict']}")
        use_gpu = device == "cuda"
        autoencoder.encoder.load_state(pretrained_paths["encoder"], use_gpu=use_gpu)
        autoencoder.decoder.load_state(pretrained_paths["decoder"], use_gpu=use_gpu)

    return autoencoder, dataset


def train_vae(args, return_model=False):
    pretrained_paths = None
    if args["state_dict"]:
        pretrained_paths = build_pretrained_checkpoint_paths(args["state_dict"])

    autoencoder, dataset = prepare_vae(args, pretrained_paths)
    os.makedirs(args["save_dir"], exist_ok=True)
    log_dir = args["log_dir"] or str(Path("output") / "logs" / Path(args["save_dir"]).name)
    os.makedirs(log_dir, exist_ok=True)
    logger.configure(dir=log_dir)
    logger.log(f"logging VAE training metrics to {log_dir}")

    start_time = time.time()
    for step in range(args["max_steps"]):
        genes, _ = next(dataset)
        training_stats = autoencoder.train_step(genes)

        elapsed_minutes = (time.time() - start_time) / 60
        should_stop = elapsed_minutes > args["max_minutes"] or step == args["max_steps"] - 1
        should_save = step % args["checkpoint_freq"] == 0 or should_stop
        should_log = step % args["log_interval"] == 0 or should_stop

        if should_log:
            logger.logkv("step", step)
            logger.logkv("loss_reconstruction", training_stats["loss_reconstruction"])
            logger.logkv("elapsed_minutes", elapsed_minutes)
            logger.dumpkvs()

        if should_save:
            checkpoint_path = os.path.join(
                args["save_dir"],
                f"model_seed={args['seed']}_step={step}.pt",
            )
            torch.save(autoencoder.state_dict(), checkpoint_path)
            logger.log(f"saved checkpoint to {checkpoint_path}")
            if should_stop:
                break

    if return_model:
        return autoencoder, dataset


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train the scDiffusion VAE")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/share/data/transcriptomics/single cell/curated_mini_umi/AML_vanGalen_2019_NanoWell_AnnData.h5ad",
    )
    parser.add_argument("--num_genes", type=int, default=19616)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample_key", type=str, default="SampleID")
    parser.add_argument("--include_sample_ids_path", type=str, default="")
    parser.add_argument("--exclude_sample_ids_path", type=str, default="")
    parser.add_argument("--max_steps", type=int, default=200000)
    parser.add_argument("--max_minutes", type=int, default=3000)
    parser.add_argument("--checkpoint_freq", type=int, default=50000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--state_dict", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="output/checkpoint/AE/my_VAE")
    parser.add_argument("--log_dir", type=str, default="")
    return vars(parser.parse_args())


if __name__ == "__main__":
    args = parse_arguments()
    seed_everything(args["seed"])
    train_vae(args)
