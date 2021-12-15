import logging
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.optim
from .module import compute_loss, eval_errors

from .cva_mvsnet import CvaMVSNet, StageTensor
from .datasets import make_dataloader, AugmentationPipeline
from .utils import epoch_end_mean, TBLogger, WarmupMultiStepLR, epoch_end_mean_named


class Tandem(pl.LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()
        self.hparams = hparams
        self.cva_mvsnet = CvaMVSNet(
            depth_num=self.hparams["MODEL.DEPTH_NUM"],
            depth_interval_ratio=self.hparams["MODEL.DEPTH_INTERVAL_RATIO"],
            cost_volume_base_channels=self.hparams["MODEL.COST_VOLUME_BASE_CHANNELS"],
            feature_net_base_channels=self.hparams["MODEL.FEATURE_NET_BASE_CHANNELS"],
            view_aggregation=self.hparams.get("MODEL.VIEW_AGGREGATION", False),
            conv2d_normalization=self.hparams.get("MODEL.CONV2D_NORMALIZATION", "batchnorm"),
            conv2d_use_bn_skip=self.hparams.get("MODEL.CONV2D_USE_BN_SKIP", False),
            conv3d_normalization=self.hparams.get("MODEL.CONV3D_NORMALIZATION", "batchnorm"),
        )
        self.augmentation_pipeline = AugmentationPipeline(hparams=self.hparams)

        self.train_batch_fun = None
        self.val_batch_fun = None
        self.test_dir = None

    def forward(self, batch: dict, stage1_depth=None):
        batch['image_noaug'] = batch['image']

        if self.training:
            batch = self.augmentation_pipeline(batch)

        outputs = self.cva_mvsnet(
            image=batch['image'],
            intrinsic_matrix=StageTensor(*[batch['intrinsics'][s]['K'] for s in self.cva_mvsnet.stages]),
            cam_to_world=batch['cam_to_world'],
            depth_min=batch['depth_min'],
            depth_max=batch['depth_max'],
        )

        return outputs

    def train_dataloader(self) -> DataLoader:
        train_loader, self.train_batch_fun = make_dataloader(self.hparams, split='train')
        return train_loader

    def val_dataloader(self):
        val_loader, self.val_batch_fun = make_dataloader(self.hparams, split='val')
        return val_loader

    def configure_optimizers(self):
        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["TRAIN.LR"])

        if self.hparams['DATA.NAME'] == 'dtu':
            epoch_splits = [10, 12, 14]
            world_size = self.hparams.get("DDP.WORLD_SIZE", 1)
            steps_per_epoch = len(self.train_dataloader()) / world_size
            logging.info(f"MVSModel: World size adapted steps per epoch: {steps_per_epoch}, world size: {world_size}")
            milestones = [int(steps_per_epoch * int(l)) for l in epoch_splits]

            batch_size_per_rank = self.hparams["TRAIN.BATCH_SIZE"]
            batch_size = batch_size_per_rank * world_size
            warmup_iter = int(500 * (16 / batch_size))
            logging.info(f"MVSModel: Warmup iter adapted to batch size: {warmup_iter}")

            lr_gamma = 1 / 2
            scheduler = WarmupMultiStepLR(optimizer, milestones, gamma=lr_gamma, warmup_factor=1.0 / 3,
                                          warmup_iters=warmup_iter)
        else:
            world_size = self.hparams.get("DDP.WORLD_SIZE", 1)
            steps_per_epoch = len(self.train_dataloader()) / world_size
            total_steps = self.hparams["TRAIN.EPOCHS"] * steps_per_epoch
            logging.info(
                f"MVSModel: World size adapted steps per epoch: {steps_per_epoch}, world size: {world_size}, total steps: {total_steps}")

            batch_size_per_rank = self.hparams["TRAIN.BATCH_SIZE"]
            batch_size = batch_size_per_rank * world_size
            warmup_iter = int(500 * (16 / batch_size))
            logging.info(f"MVSModel: Warmup iter adapted to batch size: {warmup_iter}")

            def lr_fun(step):
                # for step = 0 this is 0, for step  = total_steps - 1 this is 1
                frac = step / (total_steps - 1)
                factor = 1.0 * (1 - frac) + self.hparams["TRAIN.LR_SCHEDULE_FINAL_FRACTION"] * frac
                return factor

            # noinspection PyTypeChecker
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fun)

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def training_step(self, batch, batch_idx):
        batch = self.train_batch_fun(batch)

        outputs = self(batch)
        outputs = self.cva_mvsnet.outputs_to_dict(outputs)

        loss, losses = compute_loss(outputs=outputs, batch=batch,
                                    weights=self.hparams["LOSS.STAGE_WEIGHTS"],
                                    loss_terms=self.hparams["LOSS.TERMS"],
                                    term_weights=self.hparams["LOSS.TERM_WEIGHTS"])

        if torch.isnan(loss):
            print(f"Loss is nan, loss: {loss}, batch_idx={batch_idx}")

        with torch.no_grad():
            errors = eval_errors(outputs=outputs, batch=batch)

            logger = self.logger  # type: TBLogger
            logger.add_scalars('train', losses, errors, batch_idx, self.global_step)

            logger.add_lr('train', self.trainer.optimizers, batch_idx, self.global_step)
            logger.add_summaries('train', batch, outputs, batch_idx, self.global_step)
        return {'loss': loss, 'losses': losses, 'errors': errors}

    def validation_step(self, batch, batch_idx):
        batch = self.val_batch_fun(batch)
        outputs = self(batch)
        outputs = self.cva_mvsnet.outputs_to_dict(outputs)

        loss, losses = compute_loss(outputs=outputs, batch=batch,
                                    weights=self.hparams["LOSS.STAGE_WEIGHTS"],
                                    loss_terms=self.hparams["LOSS.TERMS"],
                                    term_weights=self.hparams["LOSS.TERM_WEIGHTS"],
                                    keep_batch='dataset_name' in batch)
        with torch.no_grad():
            errors = eval_errors(outputs=outputs, batch=batch, keep_batch='dataset_name' in batch)

            logger = self.logger  # type: TBLogger
            logger.add_summaries('val', batch, outputs, batch_idx, self.global_step)
            out = {'val_loss': loss, 'val_losses': losses, 'val_errors': errors}
            if 'dataset_name' in batch:
                out['dataset_name'] = batch['dataset_name']
        return out

    def validation_epoch_end(self, outputs):
        logger = self.logger  # type: TBLogger

        if 'dataset_name' in outputs[0]:
            names = [out.pop('dataset_name') for out in outputs]
            all_mean_output, mean_output = epoch_end_mean_named(outputs, names=names)
            logger.add_scalars('val_epoch', all_mean_output['val_losses'], all_mean_output['val_errors'], None,
                               self.global_step)
            for name in sorted(mean_output):
                logger.add_scalars('val_epoch', mean_output[name]['val_losses'], mean_output[name]['val_errors'], None,
                                   self.global_step, prefix=name + "/")
        else:
            mean_output = epoch_end_mean(outputs)
            logger.add_scalars('val_epoch', mean_output['val_losses'], mean_output['val_errors'], None,
                               self.global_step)
        return mean_output

    def training_epoch_end(self, outputs):
        mean_output = epoch_end_mean(outputs)
        logger = self.logger  # type: TBLogger
        logger.add_scalars('train_epoch', mean_output['losses'], mean_output['errors'], None, self.global_step)
        return mean_output
