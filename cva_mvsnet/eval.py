import argparse
import numpy as np
import torch
import random
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from models import Tandem
from models.datasets import MVSDataset
from models.module import eval_errors
from models.utils.helpers import tensor2numpy, to_device
from models.utils import epoch_end_mean
import cv2
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("ckpt", type=str, help="Path to pytorch lightning ckpt.")
parser.add_argument("--data_dir", help="Path to replica data.", type=str, default='data')
parser.add_argument("--num_save_images", help="Number of images to be saved for viz.", type=int, default=10)

parser.add_argument("--seed", help="Seed.", type=int, default=1)
parser.add_argument("--device", help="Torch device.", type=str, choices=('cpu', 'cuda'), default='cuda')
parser.add_argument("--batch_size", help="Batch size.", type=int, default=4)
parser.add_argument("--num_workers", help="Number of workers.", type=int, default=4)

parser.add_argument("--tuples_ext", help="Tuples Extension.", type=str, default="dso_gs")
parser.add_argument("--pose_ext", help="Pose Extension.", type=str, default="dso", choices=("dso", "gt"))

parser.add_argument("--height", help="Image height.", type=int, default=480)
parser.add_argument("--width", help="Image width.", type=int, default=640)
parser.add_argument("--depth_min", help="Depth minimum.", type=float, default=0.01)
parser.add_argument("--depth_max", help="Depth maximum.", type=float, default=10.0)

parser.add_argument("--split", help="Split file", type=str, default="val")


def main(args: argparse.Namespace):
    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    model = Tandem.load_from_checkpoint(args.ckpt)  # type: Tandem
    model = model.to(device)
    model.eval()
    outputs_to_dict = model.cva_mvsnet.outputs_to_dict

    dataset = MVSDataset(
        root_dir=args.data_dir,
        split=args.split,
        pose_ext=args.pose_ext,
        tuples_ext=args.tuples_ext,
        ignore_pose_scale=args.pose_ext == "gt",
        height=args.height,
        width=args.width,
        tuples_default_flag=False,
        tuples_default_frame_num=-1,
        tuples_default_frame_dist=-1,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        dtype="float32",
        transform=None,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=6)
    errors = []

    if args.num_save_images > 0:
        image_save_ids = tuple((np.arange(args.num_save_images) * (len(dataset) // args.num_save_images)).tolist())
    else:
        image_save_ids = tuple()
    images = []

    start = time.time()
    num_processed = 0
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader)):
                batch = to_device(batch, device=device)
                outputs = outputs_to_dict(model(batch))
                errors.append(eval_errors(outputs=outputs, batch=batch))
                num_processed += args.batch_size

                for i, idx in enumerate(range(batch_idx * args.batch_size, (batch_idx + 1) * args.batch_size)):
                    if idx in image_save_ids:
                        gt = tensor2numpy(batch['depth']['stage3'][i]).astype(np.float64) / args.depth_max
                        est = tensor2numpy(outputs['stage3']['depth'][i]).astype(np.float64) / args.depth_max
                        images.append(np.concatenate((gt, est), axis=0))

    except KeyboardInterrupt:
        pass

    elapsed = time.time() - start
    fps = num_processed / elapsed
    ms_per_frame = 1000.0 / fps

    errors = epoch_end_mean(errors)
    errors = tensor2numpy(errors)

    # Save errors
    with open(args.ckpt.rstrip('.ckpt') + '.pkl', 'wb') as fp:
        pickle.dump(obj=errors, file=fp)

    # Save images
    if len(images) > 0:
        image = np.concatenate(images, axis=1)
        if not np.all((image >= 0) & (image <= 1)):
            print(f"Image out of bounds: min/max/median = {np.amin(image)}/{np.amax(image)}/{np.median(image)}")
            image = np.clip(image, 0, 1)
        image = (image * float(np.iinfo(np.uint16).max)).astype(np.uint16)
        cv2.imwrite(args.ckpt.rstrip('.ckpt') + '.png', image)

    # Save output to file too
    with open(args.ckpt.rstrip('.ckpt') + '.txt', 'w') as fp:
        # Stage Table
        error_names = ('abs_rel', 'abs', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3')
        header = ' ' * 14 + ("{:>8s}   " * len(error_names)).format(*error_names)
        fmt_str = "{:>11s}:  " + "{:8.3f}   " * len(error_names)
        print(header, file=fp)
        for stage in errors:
            err = tuple(errors[stage][n].item() for n in error_names)
            print(fmt_str.format(stage.upper(), *err), file=fp)

        # Performance
        print(f"Performance: {fps:5.2f} FPS,  {int(ms_per_frame):5d} ms per image.", file=fp)

        # Eigen String
        print(
            f"Eigen et. al (delta <1.25, <1.25**2, <1.25**3): {errors['stage3']['d1'].item()} {errors['stage3']['d2'].item()} {errors['stage3']['d3'].item()}",
            file=fp)

        # Google Sheets String
        name = args.ckpt.rstrip(".ckpt")
        header = " " * (len(name) + 3)
        header += ("{:>8s}   " * (len(error_names) + 5)).format(
            *error_names, 'width', 'height', 'd_min', 'd_max', 'seed')[:-3]
        fmt_str = "{:>10s}   " + "{:8.4f}   " * len(error_names) + "{:8d}   {:8d}   {:8.4f}   {:8.4f}   {:8d}"
        print("\nPaste last line into Google Sheets", file=fp)
        print("" + header, file=fp)
        err = tuple(errors['stage3'][n].item() for n in error_names)
        print(fmt_str.format(name, *err, args.width, args.height, args.depth_min, args.depth_max, args.seed), file=fp)


if __name__ == "__main__":
    main(parser.parse_args())
