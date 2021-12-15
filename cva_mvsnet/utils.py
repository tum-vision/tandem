from typing import Tuple
import numpy as np
import torchvision.utils as vutils
import random
import argparse
import config as cfg
from plyfile import PlyElement, PlyData
import os
import socket
from os.path import join, isdir
import logging
import nvidia_smi
from hurry.filesize import size as filesize
from hurry.filesize import si as filesize_si
import time
import glob
import psutil
import shutil

nvidia_smi.nvmlInit()


def slurm_ddp_setup():
    num_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
    tasks_per_node = int(os.environ['SLURM_NTASKS_PER_NODE'])
    cuda_visible_devices = tuple(int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(","))
    cuda_num_devices = len(cuda_visible_devices)

    if cuda_num_devices != tasks_per_node:
        print(cuda_num_devices, tasks_per_node)
        logging.error(
            f"On node {socket.gethostname()} we have {cuda_num_devices} visible devices {cuda_visible_devices}, "
            f"but we have {tasks_per_node} tasks per node. "
            f"In your sbatch file --gres=gpu:X and --ntasks-per-node=X must be the same.")
        exit(1)

    return num_nodes, cuda_num_devices


def writer_add_scalars(writer, *, config, global_step, avg_loss, count, avg_accs_abs, batch_size, start, stages,
                       batch_idx, loader, optimizer, epoch, device, training, **kwargs):
    with torch.no_grad():
        writer.add_scalar("AA/Loss", avg_loss / count, global_step)
        writer.add_scalar(
            f"AA/Accuracy{config.IO.ACC_ABS_THRESHOLDS[config.IO.ACC_ABS_MAIN_INDEX]}",
            avg_accs_abs['stage3'][config.IO.ACC_ABS_MAIN_INDEX] / count, global_step)

        for stage in stages:
            for t, acc in zip(config.IO.ACC_ABS_THRESHOLDS, avg_accs_abs[stage]):
                writer.add_scalar(f"AccuracyAbs{stage.title()}/ErrorSmaller{t}", acc / count, global_step)

        fps = count * batch_size / (time.time() - start)
        writer.add_scalar("Sys/FPS", fps, global_step)
        writer_add_sys_stats(writer, device, global_step)

        if training:
            writer.add_scalar("Train/HoursTillEpoch",
                              (len(loader.dataset) - batch_idx * batch_size) / (fps * 3600), global_step)
            writer.add_scalar("Train/Epoch", epoch + batch_idx * batch_size / len(loader.dataset),
                              global_step)
            writer.add_scalar("Train/lr", optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar("Train/batch_size", batch_size, global_step)


def writer_add_summaries(writer, *, config, global_step, batch, outputs, loader, **kwargs):
    with torch.no_grad():
        i_batch = 0

        def scale_depth(x):
            return x / loader.dataset.depth_max

        if 'image' in config.IO.SUMMARIES:
            multi_view_image = torch.cat(torch.unbind(batch['image'][i_batch], 0), -1)  # (3, H, W*V)
            writer.add_image('multi_view_image', multi_view_image, global_step, dataformats='CHW')

        if 'depth' in config.IO.SUMMARIES:
            for stage in ('stage1', 'stage2', 'stage3'):
                depth_gt = batch['depth'][stage][i_batch]
                depth_pred = outputs[stage]['depth'][i_batch]
                error = torch.abs(depth_gt - depth_pred)
                mask = batch['mask'][stage][i_batch]

                writer.add_image(f'depth/{stage}/gt', scale_depth(depth_gt)[None, :, :],
                                 global_step, dataformats='CHW')
                writer.add_image(f'depth/{stage}/pred', scale_depth(depth_pred)[None, :, :],
                                 global_step, dataformats='CHW')
                writer.add_image(f'depth/{stage}/error_meter', scale_depth(error * mask)[None, :, :],
                                 global_step, dataformats='CHW')
                writer.add_image(f'depth/{stage}/error_rel', (error * mask)[None, :, :] / torch.max(error * mask),
                                 global_step, dataformats='CHW')

        if 'confidence' in config.IO.SUMMARIES:
            for stage in ('stage1', 'stage2', 'stage3'):
                writer.add_image(f'confidence/{stage}/confidence', outputs[stage]['confidence'][i_batch][None, :, :],
                                 global_step, dataformats='CHW')
                writer.add_image(f'confidence/{stage}/mask', batch['mask'][stage][i_batch][None, :, :],
                                 global_step, dataformats='CHW')


def format_global_step(global_step):
    s = filesize(global_step, system=filesize_si)
    if s[-1] == "B":
        s = s[:-1] + " "
    return s


def mem():
    proc = psutil.Process()
    _mem = proc.memory_info()
    return str(filesize(_mem.rss))


def mem_gb():
    proc = psutil.Process()
    _mem = proc.memory_info()
    return _mem.rss / 2 ** 30


def gpu_stats(device):
    if device.type != 'cuda':
        return None

    data = {}
    try:
        device_count = nvidia_smi.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            name = nvidia_smi.nvmlDeviceGetName(handle).decode()
            res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            data[name] = res

    except nvidia_smi.NVMLError as error:
        logging.error(error)
        return None

    device_name = torch.cuda.get_device_name(device)
    if device_name not in data:
        logging.warning(f"The device name {device_name} was not found. Possibilities {tuple(data.keys())}.")
        return None

    return data[device_name]


def writer_add_sys_stats(writer, device, step):
    writer.add_scalar("Sys/RAM_RSS", mem_gb(), step)

    res = gpu_stats(device)
    if res:
        writer.add_scalar(f"Sys/GPU_Mem", res.memory, step)
        writer.add_scalar(f"Sys/GPU_Util", res.gpu, step)


def load_model(ckpt, model):
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])


def load_optimizer(ckpt, optimizer, device):
    if 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])

        if ckpt['device.type'] != device.type:
            logging.warning(
                f"The model was saved on the "
                f"{ckpt['device.type']} and should now be executed on the {device.type}."
                f" We try the best, but this might not work properly. See source code."
            )
            # https://github.com/pytorch/pytorch/issues/2830
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)


def load_scheduler(ckpt, scheduler):
    if 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])


def str_tensor(x):
    import torch
    if torch.is_tensor(x):
        return f"{x.dtype}: {list(x.shape)}"

    if isinstance(x, list):
        if len(x) == 0:
            return "[]"
        s = "["
        for y in x:
            s += str_tensor(y) + ", "
        return s[:-2] + "]"

    if isinstance(x, tuple):
        if len(x) == 0:
            return "()"
        s = "("
        for y in x:
            s += str_tensor(y) + ", "
        return s[:-2] + ")"

    if isinstance(x, dict):
        if len(x) == 0:
            return "{}"
        s = "{"
        for key in x:
            s += f"'{key}': " + str_tensor(x[key]) + ", "
        return s[:-2] + "}"

    return str(x)


def st(x):
    return str_tensor(x)


def print_tensor(*args):
    s = " ".join(str_tensor(arg) for arg in args)
    print(s)


def parse_args(parser: argparse.ArgumentParser) -> Tuple[str, dict, str, argparse.Namespace]:
    args = parser.parse_args()

    out_dir = os.path.realpath(args.out_dir)
    if args.rm_out_dir:
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)

    # This is ddp save
    os.makedirs(out_dir, exist_ok=True)
    regular_files = tuple(sorted(glob.iglob(join(out_dir, '**/*'), recursive=True)))
    regular_files = tuple(x for x in regular_files if not isdir(x))
    if len(regular_files) != 0:
        msg = f"The output directory {out_dir} is not empty, it has {len(regular_files)} files." \
              f" Due to the specifics of pytorch lightning and ddp we only support" \
              f" empty/non-existent output directories.\n\n" \
              f"The files are: {regular_files}."
        raise RuntimeError(msg)

    config = cfg.default()
    config_path = args.config

    if config_path is not None:
        cfg.merge_from_file(config, config_path)
    if args.opts is not None:
        cfg.merge_from_list(config, args.opts)

    return str(args.out_dir), config, args.pretrained, args


# print arguments
def print_args(args):
    print("################################  args  ################################")
    for k, v in args.__dict__.items():
        print("{0: <10}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
    print("########################################################################")


# torch.no_grad warpper for functions
def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper


# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars, **kwargs):
        if isinstance(vars, list):
            return [wrapper(x, **kwargs) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x, **kwargs) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v, **kwargs) for k, v in vars.items()}
        else:
            return func(vars, **kwargs)

    return wrapper


@make_recursive_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type {} for tensor2float".format(type(vars)))


@make_recursive_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


@make_recursive_func
def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.to(torch.device("cuda"))
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


@make_recursive_func
def to_device(x, *, device):
    if torch.is_tensor(x):
        return x.to(device)
    elif isinstance(x, str):
        return x
    else:
        raise NotImplementedError(f"Invalid type for to_device: {type(x)}")


def save_scalars(logger, mode, scalar_dict, global_step):
    scalar_dict = tensor2float(scalar_dict)
    for key, value in scalar_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_scalar(name, value, global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_scalar(name, value[idx], global_step)


def save_images(logger, mode, images_dict, global_step):
    images_dict = tensor2numpy(images_dict)

    def preprocess(name, img):
        if not (len(img.shape) == 3 or len(img.shape) == 4):
            raise NotImplementedError("invalid img shape {}:{} in save_images".format(name, img.shape))
        if len(img.shape) == 3:
            img = img[:, np.newaxis, :, :]
        img = torch.from_numpy(img[:1])
        return vutils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True)

    for key, value in images_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_image(name, preprocess(name, value), global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_image(name, preprocess(name, value[idx]), global_step)


class DictAverageMeter(object):
    def __init__(self):
        self.data = {}
        self.count = 0

    def update(self, new_input):
        self.count += 1
        if len(self.data) == 0:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] = v
        else:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] += v

    def mean(self):
        return {k: v / self.count for k, v in self.data.items()}


# a wrapper to compute metrics for each image individually
def compute_metrics_for_each_image(metric_func):
    def wrapper(depth_est, depth_gt, mask, *args):
        batch_size = depth_gt.shape[0]
        results = []
        # compute result one by one
        for idx in range(batch_size):
            ret = metric_func(depth_est[idx], depth_gt[idx], mask[idx], *args)
            results.append(ret)
        return torch.stack(results).mean()

    return wrapper


@make_nograd_func
@compute_metrics_for_each_image
def Thres_metrics(depth_est, depth_gt, mask, thres):
    assert isinstance(thres, (int, float))
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    errors = torch.abs(depth_est - depth_gt)
    err_mask = errors > thres
    return torch.mean(err_mask.float())


# NOTE: please do not use this to build up training loss
@make_nograd_func
@compute_metrics_for_each_image
def AbsDepthError_metrics(depth_est, depth_gt, mask, thres=None):
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    error = (depth_est - depth_gt).abs()
    if thres is not None:
        error = error[(error >= float(thres[0])) & (error <= float(thres[1]))]
        if error.shape[0] == 0:
            return torch.tensor(0, device=error.device, dtype=error.dtype)
    return torch.mean(error)


import torch.distributed as dist


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def reduce_scalar_outputs(scalar_outputs):
    world_size = get_world_size()
    if world_size < 2:
        return scalar_outputs
    with torch.no_grad():
        names = []
        scalars = []
        for k in sorted(scalar_outputs.keys()):
            names.append(k)
            scalars.append(scalar_outputs[k])
        scalars = torch.stack(scalars, dim=0)
        dist.reduce(scalars, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            scalars /= world_size
        reduced_scalars = {k: v for k, v in zip(names, scalars)}

    return reduced_scalars


import torch
from bisect import bisect_right


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,
            gamma=0.1,
            warmup_factor=1.0 / 3,
            warmup_iters=500,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        # print("base_lr {}, warmup_factor {}, self.gamma {}, self.milesotnes {}, self.last_epoch{}".format(
        #    self.base_lrs[0], warmup_factor, self.gamma, self.milestones, self.last_epoch))
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def local_pcd(depth, intr):
    nx = depth.shape[1]  # w
    ny = depth.shape[0]  # h
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='xy')
    x = x.reshape(nx * ny)
    y = y.reshape(nx * ny)
    p2d = np.array([x, y, np.ones_like(y)])
    p3d = np.matmul(np.linalg.inv(intr), p2d)
    depth = depth.reshape(1, nx * ny)
    p3d *= depth
    p3d = np.transpose(p3d, (1, 0))
    p3d = p3d.reshape(ny, nx, 3).astype(np.float32)
    return p3d


def generate_pointcloud(rgb, depth, ply_file, intr, scale=1.0):
    """
    Generate a colored point cloud in PLY format from a color and a depth image.

    Input:
    rgb_file -- filename of color image
    depth_file -- filename of depth image
    ply_file -- filename of ply file

    """
    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
    points = []
    for v in range(rgb.shape[0]):
        for u in range(rgb.shape[1]):
            color = rgb[v, u]  # rgb.getpixel((u, v))
            Z = depth[v, u] / scale
            if Z == 0: continue
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            points.append("%f %f %f %d %d %d 0\n" % (X, Y, Z, color[0], color[1], color[2]))
    file = open(ply_file, "w")
    file.write('''ply
            format ascii 1.0
            element vertex %d
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            property uchar alpha
            end_header
            %s
            ''' % (len(points), "".join(points)))
    file.close()
    print("save ply, fx:{}, fy:{}, cx:{}, cy:{}".format(fx, fy, cx, cy))


def write_ply(fname, vert, vert_colors, dtype, text):
    assert dtype in ('float32', 'float64')
    dtype = {'float32': 'f4', 'float64': 'f8'}[dtype]
    assert vert.shape[1] == 3 and vert_colors.shape[1] == 3
    assert vert.shape[0] == vert_colors.shape[0]
    assert vert_colors.dtype == np.uint8

    vert = vert.astype(dtype)

    # Convert to plyfile compatible type
    vert = np.array([tuple(v) for v in vert],
                    dtype=[('x', dtype), ('y', dtype), ('z', dtype)])
    vert_colors = np.array(
        [tuple(v) for v in vert_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vert_all = np.empty(len(vert), vert.dtype.descr + vert_colors.dtype.descr)

    for prop in vert.dtype.names:
        vert_all[prop] = vert[prop]
    for prop in vert_colors.dtype.names:
        vert_all[prop] = vert_colors[prop]

    el = PlyElement.describe(vert_all, 'vertex')
    PlyData([el], text=text).write(fname)


def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def unproject(K, depth, half_pixel_centers, dtype):
    K, depth = np.squeeze(K), np.squeeze(depth)
    assert K.shape == (3, 3)
    assert len(depth.shape) == 2

    h, w = depth.shape
    x = np.arange(w, dtype=dtype)
    y = np.arange(h, dtype=dtype)

    if half_pixel_centers:
        x += 0.5
        y += 0.5

    X, Y = np.meshgrid(x, y)
    pixel_homog = np.stack((X.reshape(-1), Y.reshape(-1), np.ones_like(Y.reshape(-1))), -1)
    xyz = depth.reshape(-1, 1) * np.dot(pixel_homog, np.linalg.inv(K.T))

    return xyz


def make_homog(xyz):
    assert len(xyz.shape) == 2
    assert xyz.shape[-1] == 3

    return np.concatenate((xyz, np.ones_like(xyz[:, 2:3])), -1)


def unproject_tuple(batch, index, half_pixel_centers, dtype):
    num_views = batch['cam_to_world'].size(1)
    stage = 'stage3'

    xyzs = tuple(unproject(
        K=_to_np(batch['intrinsics'][stage]['K'][index]),
        depth=_to_np(batch['depth'][index, i]),
        half_pixel_centers=half_pixel_centers,
        dtype=dtype) for i in range(num_views))

    # Make Homog
    xyzs = tuple(make_homog(xyz) for xyz in xyzs)

    # Transform all to world: pose is T_cam_to_world
    xyzs = tuple(
        np.dot(xyzs[i], _to_np(batch['cam_to_world'][index, i]).T) for i in range(num_views))

    # Transform all to cam 0
    xyzs = tuple(
        np.dot(xyzs[i], np.linalg.inv(_to_np(batch['cam_to_world'][index, 0]).T)) for i in range(num_views))

    # Remove homog
    xyzs = tuple(xyz[:, :3] for xyz in xyzs)

    return np.concatenate(xyzs, 0)


def write_tuple_to_ply(fname, batch, index, half_pixel_centers, dtype, text, view_colors):
    assert dtype in ('float64', 'float32')
    num_views = batch['cam_to_world'].size(1)
    xyz = unproject_tuple(batch, index, half_pixel_centers, dtype)

    image = _to_np(255.0 * batch['image'][index]).astype(np.uint8)  # (V, C, H, W)
    image = np.transpose(image, (0, 2, 3, 1))  # (V, H, W, C)

    if not view_colors:
        colors = np.concatenate([image[i].reshape(-1, 3) for i in range(num_views)], 0)
    else:
        colors = []
        assert num_views == 3
        for i in range(num_views):
            col = image[i].reshape(-1, 3)
            col = col // 2
            col[:, i] += 80
            colors.append(col)
        colors = np.concatenate(colors, 0)

    write_ply(fname, xyz, colors, dtype=dtype, text=text)
