from collections import defaultdict
import torchaudio

class ResamplerDict(defaultdict):
    """
    Create new resampler once the orig -> new freq is different
    """
    def __missing__(self, key):
        orig_freq, new_freq = key
        new_sampler = torchaudio.transforms.Resample(
            orig_freq=orig_freq, new_freq=new_freq
        )
        self[key] = new_sampler
        return new_sampler
