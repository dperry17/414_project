import cv2
import numpy as np
import pywt
import sys

from collections import Counter
from pathlib import Path
from scipy import stats
from shutil import rmtree


def find_splices(vid, alpha=0.05, wavelet="db1"):
    interval_contains = lambda i1, i2: (i1[0] <= i2[0] and i2[1] <= i1[1])

    dec_len = pywt.Wavelet(wavelet).dec_len
    level = int(np.floor(np.log(len(vid)) / np.log(dec_len)))

    coeffs = pywt.wavedec(vid.astype(np.float32), axis=0, wavelet=wavelet)

    outliers = []
    for lvl in range(level):
        c = coeffs[lvl + 1]
        stride = dec_len ** (level - lvl)
        n = len(vid) // stride

        if n == 1:
            continue

        sq = np.square(c)[:n]
        if len(vid.shape) == 3:
            sum_sq = np.sum(np.reshape(sq, (n, -1)), axis=1)
        else:
            sum_sq = np.sum(np.reshape(sq, (n, -1, vid.shape[-1])), axis=1)

        z = stats.zscore(sum_sq)
        p = stats.norm.sf(np.abs(z)) * 2
        if len(vid.shape) > 3:
            p = np.min(p, axis=1)

        is_outlier = p < alpha
        outlier_indices = np.where(is_outlier)[0]

        intervals = [(j * stride, (j + 1) * stride) for j in outlier_indices]

        outliers = [
            (i1, *_)
            for i1, *_ in outliers
            if all(not interval_contains(i1, i2) for i2 in intervals)
        ] + [
            (intervals[i], sq[outlier_indices][i], p[outlier_indices][i])
            for i in range(len(intervals))
        ]

    return outliers


if __name__ == "__main__":
    video_name = sys.argv[1] if len(sys.argv) > 1 else "cars_2s_spliced.mp4"

    video_path = Path("input") / video_name
    output_dir = Path("output") / "find_splices" / video_path.stem
    if output_dir.exists():
        rmtree(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    vid = np.array([cap.read()[1] for _ in range(frame_count)])
    cap.release()

    args = dict()
    if len(sys.argv) > 2:
        args["alpha"] = float(sys.argv[2])
    if len(sys.argv) > 3:
        args["wavelet"] = sys.argv[3]

    for (start, end), sq, p in find_splices(vid, **args):
        cv2.imwrite(
            output_dir / f"frame_{start}_{end}.png",
            cv2.resize(
                cv2.normalize(sq, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_LINEAR,
            ),
        )
        print(
            f"Potential splice at frames {start}-{end} (confidence: {(1 - p) * 100:.2f}%)"
        )
