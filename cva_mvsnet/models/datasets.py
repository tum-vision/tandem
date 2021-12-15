import re
from os.path import join, exists, splitext
from typing import Optional, Union
from copy import deepcopy

import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import cv2

import logging

import torch
from torch import nn
import torchvision.transforms as transforms

try:
    import kornia.augmentation as kornia_aug
except ImportError:
    kornia_aug = None

cv2.setNumThreads(0)


class AugmentationPipeline(nn.Module):
    def __init__(self, hparams: dict) -> None:
        super(AugmentationPipeline, self).__init__()

        self.hparams = hparams
        modules = []

        if self._hget("AUG.COLOR_JITTER") is not None:
            brightness, contrast, saturation, hue = self._hget("AUG.COLOR_JITTER")
            modules.append(kornia_aug.ColorJitter(brightness, contrast, saturation, hue, **self._kwargs))

        if self._hget("AUG.MOTION_BLUR") is not None:
            kernel_size, angle, direction = self._hget("AUG.MOTION_BLUR")
            modules.append(kornia_aug.RandomMotionBlur(kernel_size, angle, direction))

        self.transform = nn.Sequential(*modules)

    def _hget(self, key: str):
        return self.hparams.get(key, None) if self.hparams.get("AUG.ANY", False) else None

    @property
    def _kwargs(self):
        return {"same_on_batch": self.hparams["AUG.SAME_ON_VIEWS"]}

    def forward(self, batch: dict):
        batch['image'] = torch.stack([self.transform(x) for x in torch.unbind(batch['image'])])
        return batch


def preprocess(data: dict):
    do_color_aug = random.random() > 0.5

    if not do_color_aug:
        return data

    # transforms.ColorJitter(...) creates new transformation every time, so it gives different color transformation
    # on each img of the tuple.
    # Here we use transforms.ColorJitter.get_params() to ensure that all images in the tuple has the same transformation
    # However, it might worth testing which one is better.
    b, c, s, h = 0.4, 0.4, 0.4, 0.1
    # To deal with different torchvision version
    try:
        brightness = (max(0, 1 - b), 1 + b)
        contrast = (max(0, 1 - c), 1 + c)
        saturation = (max(0, 1 - s), 1 + s)
        hue = (-h, h)
        color_trans = transforms.ColorJitter.get_params(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
    except TypeError:
        brightness = b
        contrast = c
        saturation = s
        hue = h
        color_trans = transforms.ColorJitter.get_params(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    color_trans = transforms.Compose([
        transforms.ToPILImage(),
        color_trans,
    ])

    for img_ind in range(0, data['image'].shape[0]):
        img_ori = (data['image'][img_ind, ...].transpose(1, 2, 0) * 255.0).astype(np.uint8)
        img_aug = color_trans(img_ori)
        img_out = np.transpose(np.array(img_aug), (2, 0, 1)).astype(data['image'][img_ind, ...].dtype) / 255.0
        data['image'][img_ind] = img_out

    return data


def cam_intrinsics(height: Optional[int] = None,
                   width: Optional[int] = None,
                   fx: Optional[float] = None,
                   cx: Optional[float] = None,
                   fy: Optional[float] = None,
                   cy: Optional[float] = None,
                   cam: Optional[dict] = None,
                   dtype: Optional[Union[np.dtype, str]] = None) -> dict:
    """Make camera intrinsics dict from parameters. This assumes half_pixel_centers=False.
    If any value is not filled (=None) it will be taken from cam which must then be not None.

    :param height:
    :param width:
    :param fx:
    :param cx:
    :param fy:
    :param cy:
    :param cam:
        input camera intrinsics
    :param dtype:
        Dtype of output array. Either given here or through cam.
    :return: {'K': (3, 3), 'height': int, 'width': int}
    """
    dtype = dtype if dtype is not None else cam['K'].dtype

    height = height if height is not None else _height(cam)
    width = width if width is not None else _width(cam)

    fx = fx if fx is not None else _fx(cam)
    cx = cx if cx is not None else _cx(cam)
    fy = fy if fy is not None else _fy(cam)
    cy = cy if cy is not None else _cy(cam)

    return {
        'K': np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=dtype),
        'height': height,
        'width': width
    }


def cam_resize(cam: dict,
               height: int,
               width: int):
    """Convert to new camera intrinsics for resize of image from original camera.
    :param cam:
        camera intrinsics
    :param height:
        height of resized frame
    :param width:
        width of resized frame
    :return:
        camera intrinsics for resized frame
    """
    center_x = 0.5 * float(_width(cam) - 1)
    center_y = 0.5 * float(_height(cam) - 1)

    orig_cx_diff = _cx(cam) - center_x
    orig_cy_diff = _cy(cam) - center_y

    scaled_center_x = 0.5 * float(width - 1)
    scaled_center_y = 0.5 * float(height - 1)

    scale_x = float(width) / float(_width(cam))
    scale_y = float(height) / float(_height(cam))

    fx = scale_x * _fx(cam)
    fy = scale_y * _fy(cam)
    cx = scaled_center_x + scale_x * orig_cx_diff
    cy = scaled_center_y + scale_y * orig_cy_diff

    return cam_intrinsics(height=height, width=width, fx=fx, cx=cx, fy=fy, cy=cy, dtype=cam['K'].dtype)


def cam_stack(cams: list):
    assert len(cams) > 0
    cam0 = cams[0]
    assert all((cam0["width"] == cam["width"] and cam0["height"] == cam["height"]) for cam in cams)

    return {'K': np.stack([cam['K'] for cam in cams]), 'width': cam0['width'], 'height': cam0['height']}


def _fx(cam: dict) -> float:
    return cam['K'][0, 0]


def _cx(cam: dict) -> float:
    return cam['K'][0, 2]


def _fy(cam: dict) -> float:
    return cam['K'][1, 1]


def _cy(cam: dict) -> float:
    return cam['K'][1, 2]


def _height(cam: dict) -> int:
    return cam['height']


def _width(cam: dict) -> int:
    return cam['width']


def readlines(*args, num_lines=None):
    fname = join(*args)
    with open(fname, 'r') as fp:
        lines = fp.readlines()

    lines_out = []
    for line in lines:
        if line.startswith('#'):
            continue
        if len(line) == 0:
            continue
        lines_out.append(line.rstrip())

    if num_lines is not None:
        assert len(lines_out) == num_lines, f"The file at {fname}" \
                                            f" is supposed to have {num_lines} lines but has {len(lines_out)} lines."

    return lines_out


def sample_tuple(t: tuple, num=1) -> tuple:
    t = np.array(t)
    dists = np.diff(t)
    ds = (dists - 1) // 2

    t_min = np.zeros_like(t)
    t_min[0] = t[0]
    t_min[1:] = t[1:] - ds

    t_max = np.zeros_like(t)
    t_max[-1] = t[-1]
    t_max[:-1] = t[:-1] + ds

    if num > 1:
        t_min, t_max = np.tile(t_min, [num, 1]), np.tile(t_max, [num, 1])
    return tuple(np.random.randint(t_min, t_max + 1))


def split_index(start_indices: np.ndarray, index: int):
    scene_index = np.searchsorted(start_indices, index, side='right') - 1
    inner_index = index - start_indices[scene_index]

    return scene_index, inner_index


def fix_extension(fname: str, ext: str):
    if not fname.endswith(ext):
        return fname + ext
    return fname


def resize(img: np.ndarray, height: Optional[int], width: Optional[int], interpolation: Optional[int]) -> np.ndarray:
    if width is None or height is None or interpolation is None:
        return img
    if img.shape[0] == height and img.shape[1] == width:
        return img
    return cv2.resize(img, (width, height), interpolation=interpolation)


def mask_depth(depth, depth_min, depth_max):
    mask = np.logical_and(depth >= depth_min, depth <= depth_max)
    depth[np.logical_not(mask)] = 0
    mask = mask.astype(depth.dtype)

    return depth, mask


class MVSScene(object):
    def __init__(self, scene_dir: str, pose_ext: str, height: Optional[int], width: Optional[int],
                 tuples_ext: Optional[str], ignore_pose_scale: bool,
                 tuples_default_flag: bool, tuples_default_frame_num: int, tuples_default_frame_dist: int,
                 depth_min: float, depth_max: float, dtype: str, interpolation: int):
        self.scene_dir = scene_dir
        self.pose_ext = pose_ext
        self.poses_file = fix_extension('poses_' + self.pose_ext, '.txt')
        if tuples_ext is not None:
            self.tuples_file = fix_extension('tuples_' + tuples_ext, '.txt')
        else:
            self.tuples_file = fix_extension('tuples_' + self.pose_ext, '.txt')
        self.dtype = dtype
        self.height = height
        self.width = width
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.interpolation = interpolation
        self.depth_scale = float(readlines(self.scene_dir, 'depths', 'scale.txt', num_lines=1)[0])
        del scene_dir, pose_ext, dtype

        self.cam_base = self.read_camera(self.scene_dir, self.dtype)
        self.height = self.height if self.height is not None else self.cam_base['height']
        self.width = self.width if self.width is not None else self.cam_base['width']
        assert self.height % 4 == 0 and self.width % 4 == 0

        self.poses = self.read_poses(self.scene_dir, self.poses_file, self.dtype)
        if tuples_default_flag:
            self.scales = None
            self.tuples = self.generate_tuples(self.poses, tuples_default_frame_num, tuples_default_frame_dist)
        else:
            self.tuples, self.scales = self.read_tuples(self.scene_dir, self.tuples_file,
                                                        ignore_scale=ignore_pose_scale)

        if tuples_ext != "dso_optimization_windows":
            self.num_views = len(self.tuples[0])

            self.ref_index = self.num_views // 2
            self.out_indices = (self.ref_index,) + tuple(i for i in range(self.num_views) if i != self.ref_index)
            # e.g. (1, 0, 2) or (2, 0, 1, 3, 4)

            if len(self.out_indices) == 3:
                assert self.out_indices == (1, 0, 2)
            if len(self.out_indices) == 5:
                assert self.out_indices == (2, 0, 1, 3, 4)
        else:
            self.num_views = len(self.tuples[0])
            self.ref_index = self.num_views - 2  # One before last
            self.out_indices = (self.ref_index,) + tuple(i for i in range(self.num_views) if i != self.ref_index)

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        cam_base = cam_resize(self.cam_base, height=self.height, width=self.width)

        poses_out = []
        _depths_out = []
        images_out = []
        cams_out = []

        current_tuple = self.tuples[idx]
        for view_index in self.out_indices:
            frame_index = current_tuple[view_index]
            p = self.poses[frame_index]
            if self.scales is not None:
                p = self.scale_pose(p, self.scales[idx])
            poses_out.append(p)

            images_out.append(self.read_image(frame_index))
            _depths_out.append(self.read_depth(frame_index))
            cams_out.append(deepcopy(cam_base))

        poses_out = np.stack(poses_out, 0)
        images_out = np.stack(images_out, 0)

        # resize depth
        _depths_out = np.stack(_depths_out, 0)
        depth_out_stage3 = np.copy(_depths_out[0])
        depth_out_stage2 = resize(depth_out_stage3, height=cam_base['height'] // 2, width=cam_base['width'] // 2,
                                  interpolation=self.interpolation)
        depth_out_stage1 = resize(depth_out_stage3, height=cam_base['height'] // 4, width=cam_base['width'] // 4,
                                  interpolation=self.interpolation)

        # mask depth
        _depths_out, _masks_out = mask_depth(_depths_out, self.depth_min, self.depth_max)
        depth_out_stage3, mask_out_stage3 = mask_depth(depth_out_stage3, self.depth_min, self.depth_max)
        depth_out_stage2, mask_out_stage2 = mask_depth(depth_out_stage2, self.depth_min, self.depth_max)
        depth_out_stage1, mask_out_stage1 = mask_depth(depth_out_stage1, self.depth_min, self.depth_max)

        item = {
            'intrinsics': {
                'stage3': cam_stack(cams_out),
                'stage2': cam_stack(
                    [cam_resize(cam, height=cam['height'] // 2, width=cam['width'] // 2) for cam in cams_out]),
                'stage1': cam_stack(
                    [cam_resize(cam, height=cam['height'] // 4, width=cam['width'] // 4) for cam in cams_out]),
            },
            'depth': {
                'stage3': depth_out_stage3,
                'stage2': depth_out_stage2,
                'stage1': depth_out_stage1
            },
            'mask': {
                'stage3': mask_out_stage3,
                'stage2': mask_out_stage2,
                'stage1': mask_out_stage1
            },
            'cam_to_world': poses_out,
            'image': images_out,
            'depth_min': np.dtype(self.dtype).type(self.depth_min),
            'depth_max': np.dtype(self.dtype).type(self.depth_max),
            'view_index': np.array(self.out_indices, dtype=np.int64),
        }

        return item

    @staticmethod
    def scale_pose(pose: np.ndarray, scale: float):
        pose_out = np.copy(pose)
        pose_out[:3, 3] *= scale
        return pose_out

    def read_depth(self, frame_index: int):
        fname = join(self.scene_dir, 'depths', f"{frame_index:06d}.png")

        depth = cv2.imread(fname, -1)
        assert depth.shape[:2] == (self.cam_base['height'], self.cam_base['width']), \
            f"Depth size and intrinsics must agree"

        # Assert depth sizes
        data_depth_min = self.depth_scale
        data_depth_max = self.depth_scale * np.iinfo(depth.dtype).max
        assert self.depth_min >= 2 * data_depth_min, \
            f"The min depth {self.depth_min} is not appropriate for the dataformat which has min {data_depth_min}."
        assert (self.depth_max <= data_depth_max) or np.allclose(self.depth_max, data_depth_max), \
            f"The max depth {self.depth_max} is not appropriate for the dataformat which has max {data_depth_max}."

        depth = resize(depth, height=self.height, width=self.width, interpolation=self.interpolation)
        depth = self.depth_scale * depth.astype(self.dtype)

        return depth

    def read_image(self, frame_index: int):
        fname = join(self.scene_dir, 'images', f"{frame_index:06d}.jpg")
        if not exists(fname):
            fname = splitext(fname)[0] + '.png'
        image = cv2.imread(fname, -1)
        assert image is not None, f"Couldn't load {fname}"
        assert image.shape[:2] == (self.cam_base['height'], self.cam_base['width']), \
            f"Image size and intrinsics must agree"
        image = resize(image, height=self.height, width=self.width, interpolation=self.interpolation)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        assert image.dtype == np.uint8
        if len(image.shape) == 2:
            image = image[:, :, None]

        image = np.transpose(image, (2, 0, 1)).astype(self.dtype) / 255.0
        return image

    @staticmethod
    def read_camera(scene_dir: str, dtype: str):
        lines = readlines(scene_dir, 'camera.txt')
        line = lines[0]
        line_split = line.split(" ")
        try:
            float(line_split[0])
            assert len(line_split) == 5, f"{scene_dir}: Misformed camera.txt"
            assert int(line_split[-1]) == 0, f"{scene_dir}: Misformed camera.txt"
            fx, fy, cx, cy, _ = [float(x) for x in line_split]
        except ValueError:
            assert line_split[0].lower() == 'pinhole', f"{scene_dir}: Misformed camera.txt"
            assert int(line_split[-1]) == 0, f"{scene_dir}: Misformed camera.txt"
            fx, fy, cx, cy, _ = [float(x) for x in line_split[1:]]

        intr = np.zeros((3, 3), dtype=dtype)

        assert cy > 1 and fx > 1 and fy > 1, f"{scene_dir}: Misformed camera.txt"
        intr[(0, 0, 1, 1, 2), (0, 2, 1, 2, 2)] = (fx, cx, fy, cy, 1)

        line = lines[1]
        line_split = line.split(" ")
        width, height = int(line_split[0]), int(line_split[1])

        return {
            'K': intr,
            'width': width,
            'height': height
        }

    @staticmethod
    def read_poses(scene_dir: str, poses_file: str, dtype: str):
        lines = readlines(scene_dir, poses_file)

        poses = {}
        for line in lines:
            line_split = line.split(" ")
            frame_index = int(line_split[0])
            assert frame_index not in poses, f"{scene_dir}: {frame_index} twice in {poses_file}"
            poses[frame_index] = np.reshape(np.array([float(line_split[i]) for i in range(1, 17)], dtype=dtype), (4, 4))

        return poses

    @staticmethod
    def read_tuples(scene_dir: str, tuples_file: str, ignore_scale=False):
        lines = readlines(scene_dir, tuples_file)
        num_views = int(lines[0].split(" ")[0])
        scale_available = len(lines[0].split(" ")) == num_views + 2
        scale_pose = scale_available and not ignore_scale

        tuples = []
        scales = [] if scale_pose else None
        for line in lines:
            line_split = line.split(" ")
            assert num_views == int(line_split[0]), f"{scene_dir}: Only the same number of views is supported. " \
                                                    f"Got {num_views} and {int(line_split[0])}"
            tuples.append(tuple(int(line_split[i + 1]) for i in range(num_views)))
            if scale_pose:
                assert len(line_split) == 1 + num_views + 1, \
                    f"{scene_dir}: Need {num_views + 2} values but got {len(line_split)}"
                scales.append(float(line_split[-1]))

        return tuple(tuples), tuple(scales) if scale_pose else None

    @staticmethod
    def generate_tuples(poses: dict, tuples_default_frame_num: int, tuples_default_frame_dist: int) -> tuple:
        assert tuples_default_frame_num > 1
        assert tuples_default_frame_dist > 0

        min_frame_index = min(poses.keys())
        max_frame_index = max(poses.keys())
        frame_num = max_frame_index - min_frame_index + 1
        spaced_frame_num = 1 + (frame_num - 1) // tuples_default_frame_dist
        tuple_num = spaced_frame_num - tuples_default_frame_num + 1

        tuples = tuple(
            tuple((i + j) * tuples_default_frame_dist for j in range(tuples_default_frame_num))
            for i in range(tuple_num)
        )

        for tup in tuples:
            for frame_index in tup:
                assert frame_index in poses, f"For default tuples {frame_index} does not have a pose"

        return tuples


class MVSDataset(Dataset):
    def __init__(self, root_dir: str, split: str, pose_ext: str, height: Optional[int], width: Optional[int],
                 tuples_ext: Optional[str], ignore_pose_scale: bool,
                 tuples_default_flag: bool, tuples_default_frame_num: int, tuples_default_frame_dist: int,
                 depth_min: float, depth_max: float, dtype: str = 'float32',
                 interpolation: int = cv2.INTER_NEAREST, transform=None):
        super(MVSDataset, self).__init__()
        self.root_dir = root_dir
        self.split = fix_extension(split, '.txt')
        self.pose_ext = pose_ext
        self.interpolation = interpolation
        self.dtype = dtype
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.transform = transform
        del root_dir, split, pose_ext, dtype, transform

        self.scene_names = self.read_scene_names(self.root_dir, self.split)
        self.scenes = tuple(
            MVSScene(
                join(self.root_dir, scene_name), self.pose_ext, height=height, width=width,
                depth_min=depth_min, depth_max=depth_max, dtype=self.dtype, interpolation=self.interpolation,
                tuples_ext=tuples_ext, ignore_pose_scale=ignore_pose_scale,
                tuples_default_flag=tuples_default_flag, tuples_default_frame_num=tuples_default_frame_num,
                tuples_default_frame_dist=tuples_default_frame_dist
            ) for scene_name in self.scene_names)
        tmp = np.cumsum([len(scene) for scene in self.scenes])
        self.scene_start_indices = np.zeros_like(tmp)
        self.scene_start_indices[1:] = tmp[:-1]
        self.len = tmp[-1]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        scene_index, inner_index = split_index(self.scene_start_indices, idx)
        data = self.scenes[scene_index][inner_index]
        if self.transform is not None:
            data = self.transform(data)

        data['_idx'] = idx
        return data

    @staticmethod
    def read_scene_names(root_dir, split):
        scenes = readlines(root_dir, split, num_lines=1)
        return tuple(scene for scene in scenes[0].split(" ") if len(scene) > 0)


class NamedDataset(Dataset):
    def __init__(self, *, name: str, dataset: Dataset):
        super(NamedDataset, self).__init__()
        self.name = name
        self.dataset = dataset

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        item = self.dataset.__getitem__(idx)
        assert isinstance(item, dict)
        assert "dataset_name" not in item, f"item.keys() = {list(item.keys())}"
        item["dataset_name"] = self.name
        return item

    def __repr__(self):
        return f"NamedDataset: name={self.name}, dataset={self.dataset.__repr__()}"

    def __str__(self):
        return f"NamedDataset: name={self.name}, dataset={self.dataset.__str__()}"


class TruncatedDataset(Dataset):
    def __init__(self, *, length: int, dataset: Dataset, front: bool = False):
        super(TruncatedDataset, self).__init__()
        self.length = min(length, len(dataset))
        self.dataset = dataset
        self.offset = len(dataset) - self.length if front else 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < self.length:
            return self.dataset.__getitem__(self.offset + idx)
        raise IndexError(f"Index {idx} out of bounds for TruncatedDataset of length {self.length}")

    def __repr__(self):
        return f"TruncatedDataset: length={self.length}, offset={self.offset}, dataset={self.dataset.__repr__()}"

    def __str__(self):
        return f"TruncatedDataset: length={self.length}, offset={self.offset}, dataset={self.dataset.__str__()}"


def _identity(x):
    return x


def make_dataloader(hparams: dict, split: str, truncate=None):
    if hparams['DATA.NAME'] == 'replica':
        ds = MVSDataset(
            root_dir=hparams["DATA.ROOT_DIR"],
            split=split,
            pose_ext=hparams["DATA.POSE_EXT"],
            tuples_ext=hparams["DATA.TUPLES_EXT"],
            ignore_pose_scale=hparams.get("DATA.IGNORE_POSE_SCALE", False),
            height=hparams["DATA.IMG_HEIGHT"],
            width=hparams["DATA.IMG_WIDTH"],
            tuples_default_flag=hparams["DATA.TUPLES_DEFAULT_FLAG"],
            tuples_default_frame_num=hparams["DATA.TUPLES_DEFAULT_FRAME_NUM"],
            tuples_default_frame_dist=hparams["DATA.TUPLES_DEFAULT_FRAME_DIST"],
            depth_min=hparams["DATA.DEPTH_MIN"],
            depth_max=hparams["DATA.DEPTH_MAX"],
            dtype=hparams["DATA.DTYPE"],
            transform=preprocess if split == 'train' else None
        )
        if truncate is not None:
            ds = TruncatedDataset(length=truncate, dataset=ds)
        ds = NamedDataset(name=hparams['DATA.NAME'], dataset=ds)
        batch_fun = _identity
    else:
        raise NotImplementedError(f"Dataset {hparams['DATA.NAME']} not implemented.")

    dataloader = DataLoader(
        dataset=ds,
        batch_size=hparams["TRAIN.BATCH_SIZE"],
        shuffle=hparams["TRAIN.SHUFFLE"] and split == 'train',
        num_workers=hparams["TRAIN.NUM_WORKERS"],
        drop_last=hparams["TRAIN.DROP_LAST"] and split == 'train'
    )

    return dataloader, batch_fun
