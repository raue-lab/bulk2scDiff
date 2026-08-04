import argparse
import random
from pathlib import Path

import numpy as np
import torch

from guided_diffusion.cell_datasets_loader import infer_num_genes_from_vae_checkpoint, load_data
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)

def main():
    args = create_argparser().parse_args()
    setup_seed(args.seed)
    if args.cond_pseudobulk and args.pseudobulk_dim <= 0:
        args.pseudobulk_dim = infer_num_genes_from_vae_checkpoint(args.vae_path)
    if args.mmd_eval_interval > 0 and not args.cond_pseudobulk:
        raise ValueError(
            "periodic pseudobulk MMD evaluation requires --cond_pseudobulk True"
        )

    from guided_diffusion import dist_util, logger
    from guided_diffusion.pseudobulk_mmd_eval import PeriodicPseudobulkMMDEvaluator
    from guided_diffusion.resample import create_named_schedule_sampler
    from guided_diffusion.train_util import TrainLoop

    dist_util.setup_dist()
    log_dir = Path("output") / "logs" / args.model_name
    logger.configure(dir=str(log_dir))

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    mmd_evaluator = None
    if args.mmd_eval_interval > 0:
        mmd_evaluator = PeriodicPseudobulkMMDEvaluator(
            data_dir=args.data_dir,
            vae_path=args.vae_path,
            sample_key=args.sample_key,
            eval_interval=args.mmd_eval_interval,
            include_sample_ids_path=args.include_sample_ids_path,
            exclude_sample_ids_path=args.exclude_sample_ids_path,
            validation_sample_ids_path=args.mmd_eval_validation_sample_ids_path,
            max_cells_for_mmd=args.mmd_eval_max_cells,
            sampling_batch_size=args.mmd_eval_sampling_batch_size,
            encode_batch_size=args.encode_batch_size,
            use_ddim=args.mmd_eval_use_ddim,
            random_seed=args.seed,
            hidden_dim=args.input_dim,
            output_dir=str(log_dir),
        )

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        vae_path=args.vae_path,
        train_vae=False,
        sample_key=args.sample_key,
        include_pseudobulk=args.cond_pseudobulk,
        encode_batch_size=args.encode_batch_size,
        include_sample_ids_path=args.include_sample_ids_path,
        exclude_sample_ids_path=args.exclude_sample_ids_path,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        model_name=args.model_name,
        save_dir=args.save_dir,
        mmd_evaluator=mmd_evaluator,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="<path/to/data.h5ad>",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0001,
        lr_anneal_steps=800000,
        batch_size=128,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=200000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        vae_path="output/checkpoint/AE/model_seed=0_step=199999.pt",
        model_name="aml_pseudobulk",
        save_dir="output/checkpoint/backbone",
        sample_key="SampleID",
        encode_batch_size=1024,
        include_sample_ids_path="",
        exclude_sample_ids_path="",
        mmd_eval_interval=0,
        mmd_eval_validation_sample_ids_path="",
        mmd_eval_max_cells=1200,
        mmd_eval_sampling_batch_size=1000,
        mmd_eval_use_ddim=False,
        seed=1234,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    main()
