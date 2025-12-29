import argparse
import sys
from functools import partial

import numpy as np
import torch
import xarray as xr
import matplotlib.pyplot as plt  # noqa: F401

sys.path.append('/leap/DB_scratch/physicsnemo/research/corrdiff_27_Nov')
from datasets.nyc import NYCDataset
from physicsnemo.utils.corrdiff import regression_step, diffusion_step
from physicsnemo.utils.diffusion import stochastic_sampler
from physicsnemo import Module


def full_generate_func(dataset, net_reg, net_res, len_to_gen=None, num_ensembles=4, device="cuda"):
    if len_to_gen is None:
        len_to_gen = len(dataset)

    num_in_channels = len(dataset.input_channels())
    num_out_channels = len(dataset.output_channels())
    img_shape = dataset.image_shape()

    input_phys_all = np.zeros((len_to_gen, num_in_channels) + img_shape, dtype=np.float32)
    output_phys_all = np.zeros((len_to_gen, num_ensembles, num_out_channels) + img_shape, dtype=np.float32)
    target_phys_all = np.zeros((len_to_gen, num_out_channels) + img_shape, dtype=np.float32)

    for idx in range(len_to_gen):
        print(f'Generating sample {idx+1}/{len_to_gen}...')
        image_tar, image_lr = dataset[idx]
        image_tar_t = torch.from_numpy(image_tar)
        image_lr_t = torch.from_numpy(image_lr)

        img_lr_batch = image_lr_t.unsqueeze(0).to(device=device, memory_format=torch.channels_last)

        num_channels = len(dataset.output_channels())
        seeds = list(range(num_ensembles))

        image_reg = regression_step(
            net=net_reg,
            img_lr=img_lr_batch,
            latents_shape=(1, num_channels, 128, 128),
            lead_time_label=None
        )

        sampler_fn = partial(stochastic_sampler, patching=None)
        image_res = diffusion_step(
            net=net_res,
            sampler_fn=sampler_fn,
            img_shape=(128, 128),
            img_out_channels=num_channels,
            rank_batches=[seeds],
            img_lr=img_lr_batch.expand(num_ensembles, -1, -1, -1),
            rank=0,
            device=device,
            mean_hr=image_reg[0:1],
            lead_time_label=None
        )

        output_combined = image_reg + image_res

        input_phys_all[idx] = dataset.denormalize_input(image_lr_t.cpu().numpy())
        output_phys_all[idx] = dataset.denormalize_output(output_combined.cpu().numpy())
        target_phys_all[idx] = dataset.denormalize_output(image_tar_t.cpu().numpy())

    ds_input_phys = xr.DataArray(
        input_phys_all, dims=['time', 'variable', 'y', 'x'],
        coords=dataset.input.isel(time=slice(0, len_to_gen)).coords
    ).to_dataset(dim='variable')

    ds_output_phys = xr.DataArray(
        output_phys_all, dims=['time', 'ensemble', 'variable', 'y', 'x'],
        coords=dataset.output.isel(time=slice(0, len_to_gen)).coords
    ).to_dataset(dim='variable')

    ds_target_phys = xr.DataArray(
        target_phys_all, dims=['time', 'variable', 'y', 'x'],
        coords=dataset.output.isel(time=slice(0, len_to_gen)).coords
    ).to_dataset(dim='variable')

    return ds_input_phys, ds_output_phys, ds_target_phys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--num-ensembles", type=int, default=10)
    parser.add_argument("--len-to-gen", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data-path", type=str, default="/leap/NYC_data_128/4_dec/")
    parser.add_argument("--file-path", type=str, default="ERA5_hrrr_interp_128_1945_to_2005.nc")
    parser.add_argument("--out-dir", type=str, default="/leap/NYC_data_128/generated_1945_2005")
    args = parser.parse_args()

    year = args.year

    dataset = NYCDataset(
        data_path=args.data_path,
        stats_path='',
        preload=True,
        sel_time=None,
        file_path=args.file_path,
        ERA_only=True,
        sel_year=year
    )

    print(dataset.time()[0], dataset.time()[-1])

    net_reg = Module.from_checkpoint("/leap/DB_scratch/physicsnemo/research/corrdiff_27_Nov/results_nyc_3/checkpoints/checkpoints_regression/UNet.0.10000384.mdlus")
    net_reg = net_reg.eval().to(device=args.device, memory_format=torch.channels_last)

    net_res = Module.from_checkpoint('/leap/DB_scratch/physicsnemo/research/corrdiff_27_Nov/results_nyc_3/checkpoints/checkpoints_diffusion/EDMPrecondSuperResolution.0.10000384.mdlus')
    net_res = net_res.eval().to(device=args.device, memory_format=torch.channels_last)

    ds_in, ds_out, ds_tar = full_generate_func(
        dataset, net_reg, net_res,
        len_to_gen=args.len_to_gen,
        num_ensembles=args.num_ensembles,
        device=args.device
    )

    # Convert precip units
    ds_in['total_precipitation'] = ds_in['total_precipitation'] * 1000
    ds_in['total_precipitation'].attrs['units'] = 'kg m-2'

    out_dir = args.out_dir
    ds_in.to_netcdf(f'{out_dir}/input_{year}.nc')
    ds_tar.to_netcdf(f'{out_dir}/target_{year}.nc')
    ds_out.to_netcdf(f'{out_dir}/output_{year}.nc')


if __name__ == "__main__":
    main()