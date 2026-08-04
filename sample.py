import argparse
import random
from pathlib import Path

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion.cell_datasets_loader import (
    compute_pseudobulk,
    read_preprocessed_adata,
)
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


PSEUDOBULK_REDUCTION = "sum"


def sanitize_name(value):
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))


def resolve_output_path(sample_dir, sample_id):
    output_path = Path(sample_dir)
    if output_path.suffix == ".npz":
        return output_path
    filename = f"{output_path.name}_{sanitize_name(sample_id)}.npz"
    return output_path.parent / filename


def save_data(all_cells, output_path, sample_id, input_pseudobulk, real_num_cells):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        cell_gen=all_cells,
        source_sample_id=np.asarray(sample_id),
        input_pseudobulk=np.asarray(input_pseudobulk, dtype=np.float32),
        real_num_cells=np.asarray(real_num_cells, dtype=np.int64),
        pseudobulk_reduction=np.asarray(PSEUDOBULK_REDUCTION),
    )


def select_sample_condition(args):
    adata, raw_counts, cell_library_sums = read_preprocessed_adata(
        args.data_dir,
        return_raw_counts=True,
        return_library_sums=True,
    )
    if args.sample_key not in adata.obs:
        raise KeyError(f"'{args.sample_key}' not found in adata.obs")

    sample_values = adata.obs[args.sample_key].astype(str).to_numpy()
    pseudobulk_matrix, _, sample_ids, sample_counts = compute_pseudobulk(
        raw_counts,
        sample_values,
        sample_library_sums=cell_library_sums,
        normalize_and_log1p=True,
    )

    if args.sample_id:
        matches = np.where(sample_ids == args.sample_id)[0]
        if len(matches) == 0:
            raise ValueError(
                f"sample_id '{args.sample_id}' not found. Available examples: {sample_ids[:10].tolist()}"
            )
        sample_index = int(matches[0])
    else:
        if args.sample_index < 0 or args.sample_index >= len(sample_ids):
            raise ValueError(f"sample_index must be in [0, {len(sample_ids) - 1}]")
        sample_index = args.sample_index

    return (
        sample_ids[sample_index],
        pseudobulk_matrix[sample_index].astype(np.float32),
        int(sample_counts[sample_index]),
    )


def main():
    args = create_argparser().parse_args()
    setup_seed(args.seed)

    selected_sample_id, selected_pseudobulk, real_num_cells = select_sample_condition(args)
    if args.num_samples <= 0:
        args.num_samples = real_num_cells
    if args.pseudobulk_dim <= 0:
        args.pseudobulk_dim = int(selected_pseudobulk.shape[0])
    args.cond_pseudobulk = True

    from guided_diffusion import dist_util, logger

    dist_util.setup_dist()
    logger.configure(dir="output/logs/pseudobulk_sampling")

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log(
        f"sampling with pseudobulk condition from sample '{selected_sample_id}' "
        f"({real_num_cells} real cells, generating {args.num_samples})"
    )

    all_cells = []
    total_created = 0
    output_path = resolve_output_path(args.sample_dir, selected_sample_id)
    pseudobulk_batch = th.from_numpy(selected_pseudobulk).to(dist_util.dev()).unsqueeze(0)
    sample_fn = diffusion.ddim_sample_loop if args.use_ddim else diffusion.p_sample_loop

    while total_created < args.num_samples:
        current_batch_size = min(args.batch_size, args.num_samples - total_created)
        model_kwargs = {
            "pseudobulk": pseudobulk_batch.repeat(current_batch_size, 1),
        }
        sample, _ = sample_fn(
            model,
            (current_batch_size, args.input_dim),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            start_time=diffusion.betas.shape[0],
        )

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)
        batch_samples = [item.cpu().numpy() for item in gathered_samples]
        all_cells.extend(batch_samples)
        total_created = sum(item.shape[0] for item in all_cells)
        logger.log(f"created {total_created} latent samples")

    arr = np.concatenate(all_cells, axis=0)[: args.num_samples]
    if dist.get_rank() == 0:
        save_data(
            all_cells=arr,
            output_path=output_path,
            sample_id=selected_sample_id,
            input_pseudobulk=selected_pseudobulk,
            real_num_cells=real_num_cells,
        )
        logger.log(f"sampling complete: {output_path}")

    dist.barrier()


def create_argparser():
    from guided_diffusion.script_util import add_dict_to_argparser, model_and_diffusion_defaults

    defaults = dict(
        clip_denoised=False,
        num_samples=0,
        batch_size=1000,
        use_ddim=False,
        model_path="output/checkpoint/backbone/aml_pseudobulk/model800000.pt",
        sample_dir="output/simulated_samples/aml_pseudobulk",
        sample_key="SampleID",
        sample_id="",
        sample_index=0,
        seed=1234,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults["cond_pseudobulk"] = True
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="<path/to/data.h5ad>",
    )
    add_dict_to_argparser(parser, defaults)
    return parser


def setup_seed(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.backends.cudnn.deterministic = True


if __name__ == "__main__":
    main()
