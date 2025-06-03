from datasets import Dataset
from cosyvoice.utils.audio_utils import ResamplerDict
import librosa as lb
import soundfile as sf
import numpy as np
import torchaudio
import torch

arrow_fpath = "/proj/gpu_d_09023_MR_dataset_augmented/emilia/en/arrow_for_taste/emilia-dataset-train-01696-of-04908-taste.arrow"

ds = Dataset.from_file(arrow_fpath)
# ds = ds.map(
#     lambda x: x,
#     batched=True,
#     keep_in_memory=True,
# )
print(ds)
print(ds[0])
# _mp3 = ds[0]['mp3']
# path, data, sr = _mp3['path'], _mp3['array'], _mp3['sampling_rate']
# print(path, sr)
# print(data, data.shape, data.dtype, type(data))
# # data_pt = torch.tensor(data, dtype=torch.float32).view(1, -1)
# data_pt = torch.tensor(data, dtype=torch.float32)
# print(data_pt, data_pt.shape, data_pt.dtype, data_pt.dim())
# resampler_dict = ResamplerDict()
# resampler = resampler_dict[(24000, 22050)]
# new_data = torchaudio.transforms.Resample(
#                 orig_freq=24000, new_freq=22050)(data_pt)
# print(new_data, new_data.shape, new_data.dtype)
# _new_data = resampler(data_pt)
# # print(_new_data, _new_data.shape, _new_data.dtype)
# # print((new_data == _new_data).sum())
# data = np.expand_dims(data, axis=0)
# print(data, data.shape, data.dtype)
# sf.write(path, data, sr)
# for x in ds:
