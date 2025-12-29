import datetime
import math
from typing import List, Tuple, Union

import json
import numpy as np
from numba import jit, prange
import xarray as xr
import pandas as pd

from physicsnemo.utils.diffusion import convert_datetime_to_cftime

from datasets.base import ChannelMetadata, DownscalingDataset

class NYCDataset(DownscalingDataset):
    """Reader for NYC dataset"""

    def __init__(
        self,
        data_path: str,
        stats_path: str = None,
        input_variables: Union[List[str], None] = None,
        output_variables: Union[List[str], None] = None,
        preload=False, 
        sel_time = None,
        file_path: str = None,
        ERA_only: bool = False,
        sel_year = None,
    ):
        self.preload = preload        
        # self.input, self.output, self.input_variables, self.output_variables = self._load_dataset(data_path)
        
        # if input_variables is not None:
        #     self.input_variables = input_variables
        # if output_variables is not None:
        #     self.output_variables = output_variables

            
        # self.input_mean, self.input_std, self.output_mean, self.output_std = _load_stats(data_path, self.input_variables, self.output_variables)

       
        # if sel_time is not None: 
        #     sel_time = slice(*sel_time)
        #     self.input = self.input.isel(time=sel_time)
        #     self.output = self.output.isel(time=sel_time)

        self.input, self.output, self.input_variables, self.output_variables = self._load_dataset(data_path, file_path, ERA_only)
        

        self.input_mean, self.input_std, self.output_mean, self.output_std = self._load_stats(data_path, self.input_variables, self.output_variables)

        if sel_time is not None: 
            sel_time = slice(*sel_time)
            self.input = self.input.isel(time=sel_time)
            self.output = self.output.isel(time=sel_time)

        if sel_year is not None:
            years = np.array([pd.to_datetime(t).year for t in self.input.time.values])
            year_mask = np.isin(years, sel_year)
            self.input = self.input.isel(time=year_mask)
            self.output = self.output.isel(time=year_mask)

        self.longitude_ = self.input.longitude.values
        self.latitude_ = self.input.latitude.values
        self.time_array = self.input.time.values
        self.image_shape_ = self.input.shape[-2:]

        # Preload into memory
        
        if self.preload:
            print("Preloading data into memory...")
            # to_array() stacks variables into a new dimension, gives (C, N, H, W)
            # transpose to (N, C, H, W)
            self.input.load()
            self.output.load()

            # Normalize once upfront
            #self.input_data = (self.input_data - self.input_mean) / self.input_std
            #self.output_data = (self.output_data - self.output_mean) / self.output_std
            print(f"Loaded input: {self.input.shape}, output: {self.output.shape}")


    def __getitem__(self, idx):
        input_data = self.input.isel(time=idx).values
        output_data = self.output.isel(time=idx).values
        
        x = self.normalize_input(input_data)
        y = self.normalize_output(output_data)

        #return x.astype(np.float32), y.astype(np.float32)
        return y.astype(np.float32), x.astype(np.float32)
    
    def __len__(self) -> int:
        """Number of samples in the dataset."""
        return len(self.time_array)

    def input_channels(self) -> List[ChannelMetadata]:
        """Metadata for the input channels. A list of ChannelMetadata, one for each channel"""
        return [ChannelMetadata(name=v) for v in self.input_variables]
    
    def output_channels(self) -> List[ChannelMetadata]:
        """Metadata for the output channels. A list of ChannelMetadata, one for each channel"""
        return [ChannelMetadata(name=v) for v in self.output_variables]
    
    def longitude(self):
        return self.longitude_

    def latitude(self):
        return self.latitude_

    def image_shape(self):
        return self.image_shape_

    def time(self):
        """Get time values from the dataset."""
        datetimes = (
            datetime.datetime.utcfromtimestamp(t.tolist() / 1e9) 
            for t in self.time_array
        )
        return [convert_datetime_to_cftime(t) for t in datetimes]
    #def time(self):
    #    return self.input.time.values

    def normalize_input(self, x: np.ndarray) -> np.ndarray:
        """Convert input from physical units to normalized data."""
        return (x - self.input_mean) / self.input_std

    def denormalize_input(self, x: np.ndarray) -> np.ndarray:
        """Convert input from normalized data to physical units."""
        return x * self.input_std + self.input_mean

    def normalize_output(self, x: np.ndarray) -> np.ndarray:
        """Convert output from physical units to normalized data."""
        return (x - self.output_mean) / self.output_std

    def denormalize_output(self, x: np.ndarray) -> np.ndarray:
        """Convert output from normalized data to physical units."""
        return x * self.output_std + self.output_mean


## Load data function
    def _load_dataset(self, data_path, file_path=None, ERA_only=False):
        """Load dataset from zarr files"""
    
        if self.preload is True and ERA_only is True:
            print('Using NC files')
            path_input = data_path+file_path#'/leap/NYC_data_128/4_dec/ERA5_hrrr_interp_128_2020_2025.nc'
            ds_input =  xr.open_dataset(path_input)

            # Load a reference HRRR dataset to get output variable names and shapes
            path_output_ref = data_path+'hrrr_NYC_128_2020_2025.nc'
            ds_output_ref = xr.open_dataset(path_output_ref)
            output_vars = list(ds_output_ref.keys())
            
            # Create zero-filled output matching ERA5 time axis
            ds_output = xr.Dataset()
            for var in output_vars:
                # Get the shape from reference (excluding time dimension)
                ref_shape = ds_output_ref[var].isel(time=0).shape
                # Create zeros with ERA5 time dimension
                zeros = np.zeros((len(ds_input.time), *ref_shape), dtype=np.float32)
                ds_output[var] = (('time', 'latitude', 'longitude'), zeros)
            
            # Copy coordinates from ds_input
            ds_output = ds_output.assign_coords({
                'time': ds_input.time,
                'latitude': ds_input.latitude,
                'longitude': ds_input.longitude
            })
        elif self.preload is True and ERA_only is False:
            print('Using NC files')
            path_input = data_path+'ERA5_hrrr_interp_128_2020_2025.nc'#'/leap/NYC_data_128/4_dec/ERA5_hrrr_interp_128_2020_2025.nc'
            path_output = data_path+'hrrr_NYC_128_2020_2025.nc'#'/leap/NYC_data_128/4_dec/hrrr_NYC_128_2020_2025.nc'
            ds_input =  xr.open_dataset(path_input)
            ds_output =  xr.open_dataset(path_output)
        else: 
            print('Using zarr files')
            zarr_path_input = data_path+'/leap/NYC_data_128/4_dec/ERA5_hrrr_interp_128_2020_2025.zarr'
            zarr_path_output = data_path+'/leap/NYC_data_128/4_dec/hrrr_NYC_128_2020_2025.zarr'
    
            ds_input =  xr.open_zarr(zarr_path_input)
            ds_output =  xr.open_zarr(zarr_path_output)
        # drop specific times when the data is corrupted
        input_vars = list(ds_input.keys())
        output_vars = list(ds_output.keys())
    
        ds = xr.merge([ds_input, ds_output], compat='override')
        
        if ERA_only is False:
            ds = _drop_nan_timesteps(ds, data_path)
    
        ds_input = ds[input_vars].to_array()
        ds_output = ds[output_vars].to_array()

    
        return ds_input, ds_output, input_vars, output_vars

## Function to drop NaN timesteps
    

## Load stats 
    def _load_stats(self, data_path, input_variables, output_variables):

        ds_mean = xr.open_dataset(data_path+'all_mean.nc')
        ds_std = xr.open_dataset(data_path+'all_std.nc')

        input_mean = ds_mean[input_variables].to_array().values.reshape(-1, 1, 1)
        input_std = ds_std[input_variables].to_array().values.reshape(-1, 1, 1)

        output_mean = ds_mean[output_variables].to_array().values.reshape(-1, 1, 1)
        output_std = ds_std[output_variables].to_array().values.reshape(-1, 1, 1)

        return input_mean, input_std, output_mean, output_std




def _drop_nan_timesteps(ds, data_path):
    """Drop all timesteps where any variable has any NaN."""
    valid_time = xr.open_dataset(data_path+'valid_time.nc')
    # Keep only times with no NaNs
    return ds.sel(time=valid_time.time)