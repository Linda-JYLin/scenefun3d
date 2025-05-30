# Modify https://github.com/scannetpp/scannetpp/blob/5b8c07183748f75ad1d0f6c7d14cede940956b14/common/utils/rle.py
import numpy as np


def rle_encode(mask):
    """Encode RLE (Run-length-encode) from 1D binary mask.

    Args:
        mask (np.ndarray): 1D binary mask
    Returns:
        counts (list): encoded counts RLE
    """
    length = mask.shape[0]
    mask = np.concatenate([[0], mask, [0]])
    runs = np.where(mask[1:] != mask[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    counts = " ".join(str(x) for x in runs)
    return counts


def rle_decode(counts, length):
    """Decode rle to get binary mask.

    Args:
        rle (dict): rle of encoded mask
    Returns:
        mask (np.ndarray): decoded mask
    """
    s = counts.split()
    starts, nums = [np.asarray(x, dtype=np.int64) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + nums
    mask = np.zeros(length, dtype=np.uint16)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask
