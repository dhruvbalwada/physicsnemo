import datetime
import math
from typing import List, Tuple, Union

import json
import numpy as np
from numba import jit, prange
import xarray as xr

from physicsnemo.utils.diffusion import convert_datetime_to_cftime

from datasets.base import ChannelMetadata, DownscalingDataset

class NYCwindowDataset(DownscalingDataset):
    """Reader for NYC dataset with a time windowing approach"""

    def __init__(
        self,
        data_path: str,
        stats_path: str = None,
        input_variables: Union[List[str], None] = None,
        output_variables: Union[List[str], None] = None,
        preload=False, 
        sel_time = None
    ):
        self.preload = preload        
        self.input, self.output, self.input_variables, self.output_variables = self._load_dataset(data_path)
        

        self.input_mean, self.input_std, self.output_mean, self.output_std = self._load_stats(data_path, self.input_variables, self.output_variables)

        if sel_time is not None: 
            sel_time = slice(*sel_time)
            self.input = self.input.isel(time=sel_time)
            self.output = self.output.isel(time=sel_time)

        self.longitude_ = self.input.longitude.values
        self.latitude_ = self.input.latitude.values
        self.time_array = self.input.time.values
        self.image_shape_ = self.input.shape[-2:] 
        # Preload into memory
        
        if self.preload:
            print("Preloading data into memory...")
            # to_array() stacks variables into a new dimension, gives (C, N, H, W)
            # transpose to (N, C, H, W)
            #self.input_data = self.input[self.input_variables].to_array().transpose('time', 'variable', 'local_time_window',...)
            #self.output_data = self.output[self.output_variables].to_array().transpose('time', 'variable', 'local_time_window',...)
            
            # Normalize once upfront
            #self.input = (self.input.values - self.input_mean) / self.input_std
            #self.output = (self.output.values - self.output_mean) / self.output_std
            self.input.load()
            self.output.load()

            print(f"Loaded input: {self.input.shape}, output: {self.output.shape}")

    def __getitem__(self, idx):
        # if self.preload:
        #     # Fast: just index into arrays
        #     #return self.input_data[idx].astype(np.float32), self.output_data[idx].astype(np.float32)
        #     input_data = self.input[idx].values
        #     output_data = self.output[idx].values

        #     x = self.normalize_input(input_data)
        #     y = self.normalize_output(output_data)

        #     return y.astype(np.float32), x.astype(np.float32)
        # else:
            # Lazy load from zarr (slow)
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
    def _load_dataset(self, data_path):
        """Load dataset from zarr files"""
    
        path_input_data = data_path+'/leap/NYC_data_128/data_input_expanded.nc'
        path_output_data = data_path+'/leap/NYC_data_128/data_output_expanded.nc'
        
        ds_input =  xr.open_dataset(path_input_data)
        ds_output =  xr.open_dataset(path_output_data)
        
        ds_input = ds_input['__xarray_dataarray_variable__']
        ds_output = ds_output['__xarray_dataarray_variable__']

        input_vars = list(ds_input.channel.values)
        output_vars = list(ds_output.channel.values)
    
        return ds_input, ds_output, input_vars, output_vars


    ## Load stats 
    def _load_stats(self, data_path, input_variables, output_variables):

        stats_in = xr.open_dataset('/leap/NYC_data_128/input_stats_expanded.nc')
        stats_out = xr.open_dataset('/leap/NYC_data_128/output_stats_expanded.nc')

        input_mean = stats_in['mean'].values.reshape(-1, 1, 1)
        input_std = stats_in['std'].values.reshape(-1, 1, 1)
        output_mean = stats_out['mean'].values.reshape(-1, 1, 1)
        output_std = stats_out['std'].values.reshape(-1, 1, 1)

        return input_mean, input_std, output_mean, output_std
    

class NYCwindowDataset_old(DownscalingDataset):
    """Reader for NYC dataset with a time windowing approach"""

    def __init__(
        self,
        data_path: str,
        stats_path: str = None,
        input_variables: Union[List[str], None] = None,
        output_variables: Union[List[str], None] = None,
        preload=False, 
        sel_time = None
    ):
        self.preload = preload        
        self.input, self.output, self.input_variables, self.output_variables = self._load_dataset(data_path)
        
        if input_variables is not None:
            self.input_variables = input_variables
        if output_variables is not None:
            self.output_variables = output_variables

            
        self.input_mean, self.input_std, self.output_mean, self.output_std = self._load_stats(data_path, self.input_variables, self.output_variables)

       
        if sel_time is not None: 
            sel_time = slice(*sel_time)
            self.input = self.input.isel(time=sel_time)
            self.output = self.output.isel(time=sel_time)

        # Preload into memory
        
        if self.preload:
            print("Preloading data into memory...")
            # to_array() stacks variables into a new dimension, gives (C, N, H, W)
            # transpose to (N, C, H, W)
            self.input_data = self.input[self.input_variables].to_array().transpose('time', 'variable', 'local_time_window',...)
            self.output_data = self.output[self.output_variables].to_array().transpose('time', 'variable', 'local_time_window',...)
            
            # Normalize once upfront
            self.input_norm = (self.input_data - self.input_mean) / self.input_std
            self.output_norm = (self.output_data - self.output_mean) / self.output_std

            # Center the window coord
            #half = self.input_norm.sizes["local_time_window"] // 2
            #self.input_norm = self.input_norm.assign_coords(
            #    local_time_window=np.arange(-half, half + 1)
            #)
            #self.output_norm = self.output_norm.assign_coords(
            #    local_time_window=np.arange(-half, half + 1)
            #)

            # Stack and create "var|t-2" ... "var|t+2" labels
            self.input_stacked = (
                self.input_norm
                .stack(channel=("variable", "local_time_window"))
                .transpose("time", "channel", "y", "x")
            )
            self.input_stacked = self.input_stacked.reset_index("channel")
            labels = np.array(
                [f"{v}|t{lw:+d}" for v, lw in zip(self.input_stacked["variable"].values,
                                                self.input_stacked["local_time_window"].values)],
                dtype=object,
            )
            self.input_stacked = (
                self.input_stacked
                .assign_coords(channel=("channel", labels))
                .drop_vars(["variable", "local_time_window"])
            )

            self.output_stacked = (
                self.output_norm
                .stack(channel=("variable", "local_time_window"))
                .transpose("time", "channel", "y", "x")
            )
            self.output_stacked = self.output_stacked.reset_index("channel")
            labels = np.array(
                [f"{v}|t{lw:+d}" for v, lw in zip(self.output_stacked["variable"].values,
                                                self.output_stacked["local_time_window"].values)],
                dtype=object,
            )
            self.output_stacked = (
                self.output_stacked
                .assign_coords(channel=("channel", labels))
                .drop_vars(["variable", "local_time_window"])
            )

            self.input_data = self.input_stacked.values
            self.output_data = self.output_stacked.values

            self.input_channel_names = self.input_stacked.channel.values
            self.output_channel_names = self.output_stacked.channel.values

            print(f"Loaded input: {self.input_data.shape}, output: {self.output_data.shape}")


    def __getitem__(self, idx):
        if self.preload:
            # Fast: just index into arrays
            #return self.input_data[idx].astype(np.float32), self.output_data[idx].astype(np.float32)
            return self.output_data[idx].astype(np.float32), self.input_data[idx].astype(np.float32)
        else:
            # Lazy load from zarr (slow)
            input_data = self.input[self.input_variables].isel(time=idx)
            output_data = self.output[self.output_variables].isel(time=idx)
            x = np.stack([input_data[v].values for v in self.input_variables], axis=0)
            y = np.stack([output_data[v].values for v in self.output_variables], axis=0)
            x = self.normalize_input(x)
            y = self.normalize_output(y)
            #return x.astype(np.float32), y.astype(np.float32)
            return y.astype(np.float32), x.astype(np.float32)
    
    def __len__(self) -> int:
        """Number of samples in the dataset."""
        return len(self.input.time)

    def input_channels(self) -> List[ChannelMetadata]:
        """Metadata for the input channels. A list of ChannelMetadata, one for each channel"""
        return [ChannelMetadata(name=v) for v in self.input_channel_names]
    
    def output_channels(self) -> List[ChannelMetadata]:
        """Metadata for the output channels. A list of ChannelMetadata, one for each channel"""
        return [ChannelMetadata(name=v) for v in self.output_channel_names]
    
    def longitude(self):
        return self.input.longitude.values

    def latitude(self):
        return self.input.latitude.values

    def image_shape(self):
        return self.output[self.output_variables[0]].shape[1:3]  # H, W

    def time(self):
        """Get time values from the dataset."""
        datetimes = (
            datetime.datetime.utcfromtimestamp(t.tolist() / 1e9) 
            for t in self.input.time.values
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
    def _load_dataset(self, data_path):
        """Load dataset from zarr files"""
    
        path_input_vars = data_path+'/leap/NYC_data_128/ERA5_hrrr_interp_128_2020_2025.nc'
        path_output_vars = data_path+'/leap/NYC_data_128/hrrr_NYC_128_2020_2025.nc'
        path_data = data_path+'/leap/NYC_data_128/combined_window_5_NYC_128_2020_2025.nc'
        
        ds_input =  xr.open_dataset(path_input_vars)
        ds_output =  xr.open_dataset(path_output_vars)
        ds_data =  xr.open_dataset(path_data)
        # drop specific times when the data is corrupted
        input_vars = list(ds_input.keys())
        output_vars = list(ds_output.keys())
    
        ds_input = ds_data[input_vars]
        ds_output = ds_data[output_vars]
    
        return ds_input, ds_output, input_vars, output_vars


    ## Load stats 
    def _load_stats(self, data_path, input_variables, output_variables):

        ds_mean = xr.open_dataset('/leap/NYC_data_128/all_mean.nc')
        ds_std = xr.open_dataset('/leap/NYC_data_128/all_std.nc')

        input_mean = ds_mean[input_variables].to_array()#.values.reshape(-1, 1, 1)
        input_std = ds_std[input_variables].to_array()#.values.reshape(-1, 1, 1)
    
        output_mean = ds_mean[output_variables].to_array()#.values.reshape(-1, 1, 1)
        output_std = ds_std[output_variables].to_array()#.values.reshape(-1, 1, 1)

        return input_mean, input_std, output_mean, output_std

