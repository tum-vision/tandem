import sys
import numpy as np
import argparse
from os.path import join, basename, splitext
from glob import glob
import cv2


def load_depth(f, factor=1000.0):
    d = cv2.imread(f, -1)
    assert d is not None, "Could not read " + str(f)
    return d.astype(np.float) / factor


def comp_depth(*args):
    mask = args[0] > 0
    for x in args:
        mask = np.logical_and(mask, x > 0)

    return [x[mask] for x in args]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Align the scale using depth maps.')
    parser.add_argument('data_dir', help='Data folder (must contain associate.txt)')
    parser.add_argument('dso_dir', help='Result Folder of DSO')
    parser.add_argument('--depth_dir', help='depth directory if not data_dir/depth')
    parser.add_argument('--num_frames', help='Number of frames used for estimation (default: 20)', type=int, default=20)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    assoc = {}
    with open(join(args.data_dir, 'associate.txt'), "r") as fp:
        lines = fp.readlines()

    for line in lines:
        s = line.strip().split(" ")
        t_rgb, f_rgb, t_dpt, f_dpt = s[0], s[1], s[2], s[3]

        assoc[t_rgb] = t_dpt

    depth_pred_dir = join(args.dso_dir, "mvs_depth")
    ts = sorted([splitext(basename(x))[0] for x in glob(join(depth_pred_dir, "*.png"))])
    ts = [t for t in ts if t in assoc]
    sub = len(ts) // (args.num_frames - 1)
    ts = ts[::sub]

    if args.depth_dir is not None:
        depth_gt_dir = args.depth_dir
    else:
        depth_gt_dir = join(args.data_dir, "depth")

    preds = []
    gts = []
    for i, t in enumerate(ts):
        pred = load_depth(join(depth_pred_dir, t + ".png"), factor=1000)
        gt = load_depth(join(depth_gt_dir, assoc[t] + ".png"), factor=5000)

        pred, gt = comp_depth(pred, gt)
        preds.append(pred)
        gts.append(gt)

    preds = np.concatenate(preds)
    gts = np.concatenate(gts)

    scale = np.median(gts / preds)

    abs_abs = np.mean(np.abs(scale * preds - gts))

    print("--scale " + str(scale))
    if args.verbose:
        print("ABS:   " + str(abs_abs) + " m")
