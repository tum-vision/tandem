from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_only
import csv
import os
import numpy as np
from os.path import join
from argparse import Namespace
from typing import Optional, Dict, Union, Any
from warnings import warn

import torch
from pkg_resources import parse_version
from torch.utils.tensorboard import SummaryWriter

import matplotlib.cm


class TBLogger(LightningLoggerBase):
    NAME_CSV_TAGS = 'meta_tags.csv'

    def __init__(self,
                 out_dir: str,
                 hparams: dict,
                 i_batch=0,
                 **kwargs):
        super().__init__()
        self.out_dir = out_dir

        self._experiment = None  # type: Optional[Dict[str, SummaryWriter]]
        self._sub_loggers = ('train', 'val', 'train_epoch', 'val_epoch')
        self.tags = {}
        self._kwargs = kwargs
        self.hparams = hparams

        self.i_batch = i_batch
        self.stages = ('stage1', 'stage2', 'stage3')

    @property
    def experiment(self) -> Optional[Dict[str, SummaryWriter]]:
        if self._experiment is not None:
            return self._experiment

        self._experiment = {}
        for sub in self._sub_loggers:
            os.makedirs(join(self.out_dir, sub), exist_ok=True)
            self._experiment[sub] = SummaryWriter(log_dir=join(self.out_dir, sub), **self._kwargs)

        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        sanitized_params = self._sanitize_params(params)

        if parse_version(torch.__version__) < parse_version("1.3.0"):
            warn(
                f"Hyperparameter logging is not available for Torch version {torch.__version__}."
                " Skipping log_hyperparams. Upgrade to Torch 1.3.0 or above to enable"
                " hyperparameter logging."
            )
        else:
            from torch.utils.tensorboard.summary import hparams
            exp, ssi, sei = hparams(sanitized_params, {})
            writer = self.experiment[self._sub_loggers[0]]._get_file_writer()
            writer.add_summary(exp)
            writer.add_summary(ssi)
            writer.add_summary(sei)

        # some alternative should be added
        self.tags.update(sanitized_params)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        return

    @rank_zero_only
    def save(self) -> None:
        super().save()
        for _, exp in self.experiment.items():
            try:
                exp.flush()
            except AttributeError:
                # you are using PT version (<v1.2) which does not have implemented flush
                exp._get_file_writer().flush()

        # prepare the file path
        meta_tags_path = os.path.join(self.out_dir, self._sub_loggers[0], self.NAME_CSV_TAGS)

        # save the metatags file
        with open(meta_tags_path, 'w', newline='') as csvfile:
            fieldnames = ['key', 'value']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'key': 'key', 'value': 'value'})
            for k, v in self.tags.items():
                writer.writerow({'key': k, 'value': v})

    @rank_zero_only
    def finalize(self, status: str) -> None:
        self.save()

    @property
    def name(self) -> str:
        return self.out_dir

    @property
    def version(self) -> int:
        return 0

    def _scale_depth(self, x, depth_max=None):
        if depth_max is None:
            return x / self.hparams["DATA.DEPTH_MAX"]
        return x / depth_max

    def _should_log_scalars(self, tag, batch_idx, global_step) -> bool:
        del batch_idx
        assert tag != 'val', "Scalars should be logged only in validation_epoch_end."

        if tag in ('val_epoch', 'train_epoch'):
            return True
        if tag == 'train':
            return global_step % self.hparams["IO.LOG_INTERVAL"] == 0
        raise NotImplementedError("This scenario has not been implemented.")

    def _should_log_summaries(self, tag, batch_idx, global_step) -> bool:
        assert tag not in ('train_epoch', 'val_epoch'), "Summaries should not be logged in *_epoch_end methods."

        if tag == 'val':
            return batch_idx % self.hparams["IO.LOG_INTERVAL"] == 0
        if tag == 'train':
            return global_step % self.hparams["IO.LOG_INTERVAL"] == 0
        raise NotImplementedError("This scenario has not been implemented.")

    @rank_zero_only
    def add_scalars(self, tag, losses, errors, batch_idx, global_step, prefix=""):
        if not self._should_log_scalars(tag, batch_idx, global_step):
            return

        global_sample = self.hparams["IO.SAMPLES_PER_STEP"] * global_step
        del global_step

        writer = self._experiment[tag]  # type: SummaryWriter

        for key, val in losses.items():
            writer.add_scalar(f"0.Main/{prefix}{key}", val, global_sample)

        for i, (k, v) in enumerate(errors['stage3'].items()):
            writer.add_scalar(f"0.Main/{prefix}{i}.{k}", v, global_sample)

        for i_stage, stage in enumerate(self.stages):
            for i, (k, v) in enumerate(errors[stage].items()):
                writer.add_scalar(f"{3 - i_stage}.Errors{stage.title()}/{prefix}{k}", v, global_sample)

    @rank_zero_only
    def add_lr(self, tag, optimizers, batch_idx, global_step):
        if not self._should_log_scalars(tag, batch_idx, global_step):
            return

        global_sample = self.hparams["IO.SAMPLES_PER_STEP"] * global_step
        del global_step

        writer = self._experiment[tag]  # type: SummaryWriter

        if isinstance(optimizers, (list, tuple)) and len(optimizers) == 1:
            optimizers = optimizers[0]

        if not isinstance(optimizers, (list, tuple)):
            writer.add_scalar("4.Train/Lr", optimizers.param_groups[0]['lr'], global_sample)
        else:
            for i in range(len(optimizers)):
                writer.add_scalar(f"4.Train/Lr{i}", optimizers[i].param_groups[0]['lr'], global_sample)

    @rank_zero_only
    def add_summaries(self, tag, batch, outputs, batch_idx, global_step):
        if not self._should_log_summaries(tag, batch_idx, global_step):
            return

        global_sample = self.hparams["IO.SAMPLES_PER_STEP"] * global_step
        del global_step

        writer = self._experiment[tag]  # type: SummaryWriter

        if 'image' in self.hparams['IO.SUMMARIES']:
            multi_view_image = torch.cat(torch.unbind(batch['image'][self.i_batch], 0), -1)  # (3, H, W*V)
            writer.add_image('0.multi_view_image', multi_view_image, global_sample,
                             dataformats='CHW')
            multi_view_image_noaug = torch.cat(torch.unbind(batch['image_noaug'][self.i_batch], 0), -1)  # (3, H, W*V)
            writer.add_image('0.multi_view_image_noaug', multi_view_image_noaug, global_sample,
                             dataformats='CHW')
            # photometric loss is turned on
            if 'img_warped' in outputs['stage3']:
                writer.add_image('0.warped', torch.cat(outputs['stage3']['img_warped'], -1)[self.i_batch],
                                 global_sample,
                                 dataformats='CHW')

        if 'depth' in self.hparams['IO.SUMMARIES']:
            for stage in self.stages:
                depth_gt = batch['depth'][stage][self.i_batch]
                depth_pred = outputs[stage]['depth'][self.i_batch]
                error = torch.abs(depth_gt - depth_pred)
                if 'mask_total' in batch:
                    mask = batch['mask_total'][stage][self.i_batch]
                else:
                    mask = batch['mask'][stage][self.i_batch]

                depth_max = batch['depth_max'][self.i_batch]
                writer.add_image(f'1.depth_gt/{stage}', self._scale_depth(depth_gt, depth_max)[None, :, :],
                                 global_sample, dataformats='CHW')
                writer.add_image(f'2.depth_pred/{stage}',
                                 self._scale_depth(depth_pred, depth_max)[None, :, :],
                                 global_sample, dataformats='CHW')
                writer.add_image(f'3.depth_err_abs/{stage}',
                                 self._scale_depth(error * mask, depth_max)[None, :, :],
                                 global_sample, dataformats='CHW')
                writer.add_image(f'4.depth_err_rel/{stage}',
                                 (error * mask)[None, :, :] / torch.max(error * mask),
                                 global_sample, dataformats='CHW')

        if 'confidence' in self.hparams['IO.SUMMARIES']:
            for stage in ('stage1', 'stage2', 'stage3'):
                writer.add_image(f'5.confidence/{stage}',
                                 outputs[stage]['confidence'][self.i_batch][None, :, :],
                                 global_sample, dataformats='CHW')
                writer.add_image(f'6.mask/{stage}',
                                 batch['mask'][stage][self.i_batch][None, :, :],
                                 global_sample, dataformats='CHW')

        if 'warp' in self.hparams['IO.SUMMARIES']:
            for stage in ('stage1', 'stage2', 'stage3'):
                writer.add_image(f'7.warp_image/{stage}',
                                 torch.cat(torch.unbind(batch['warp_image'][stage][self.i_batch], 0), -1),
                                 global_sample, dataformats='CHW')
                writer.add_image(f'8.warp_mask/{stage}',
                                 torch.cat(torch.unbind(batch['warp_mask'][stage][self.i_batch], 0), -1),
                                 global_sample, dataformats='CHW')
                error = torch.unbind(batch['warp_image'][stage][self.i_batch], 0)  # (V)(C, H, W)
                error = [torch.mean(torch.abs(x - batch['warp_image'][stage][self.i_batch][0]), 0) for x in
                         error]  # [V](H, W)
                for i in range(len(error)):
                    error[i][batch['mask_total'][stage][self.i_batch, i, 0] < 0.5] = 0.0
                    error[i] = colorize(error[i])
                writer.add_image(f'9.warp_error/{stage}',
                                 torch.cat(error, -1),
                                 global_sample, dataformats='CHW')


_cm = matplotlib.cm.get_cmap("plasma")
_colors = _cm(np.arange(256))[:, :3]
_colors = torch.from_numpy(_colors)


# https://gist.github.com/jimfleming/c1adfdb0f526465c99409cc143dea97b#gistcomment-2398882
def colorize(value):
    """
    A utility function for Torch/Numpy that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.
    Arguments:
      - value: 2D Tensor of shape [height, width]
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: Matplotlib default colormap)

    Returns a 4D uint8 tensor of shape [height, width, 4].
    """

    global _colors
    _colors = _colors.to(value.device)

    size = list(value.size())
    idx = torch.reshape((255 * value).to(torch.long), (-1,))
    idx = idx.unsqueeze(1).expand(-1, 3)

    out = torch.gather(_colors, 0, idx)
    out = torch.reshape(out, size + [3])

    out = torch.stack(torch.unbind(out, -1), 0)

    return out
