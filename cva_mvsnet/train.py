import argparse
import logging
import numpy as np
from os.path import join
import torch
from utils import parse_args
import warnings
import random
import time
import subprocess
from typing import Optional
from packaging import version

warnings.filterwarnings("ignore", message='Displayed epoch numbers in the progress bar start from "1" until v0.6.x,'
                                          ' but will start from "0" in v0.8.0.')
import pytorch_lightning as pl  # noqa
from pytorch_lightning.callbacks import ModelCheckpoint  # noqa
from utils import slurm_ddp_setup  # noqa

from models import Tandem  # noqa
from models.utils import TBLogger  # noqa
from models.utils.load_ckpt import load_ckpt  # noqa

parser = argparse.ArgumentParser()
parser.add_argument("out_dir", help="Output directory.", type=str)
parser.add_argument("--config", help="Path to config file.", required=True)
parser.add_argument("--pretrained", help="Path to pretrained ckpt.")
parser.add_argument("--rm_out_dir", action="store_true", help="Remove out dir. Not DDP save.")
parser.add_argument("opts", nargs=argparse.REMAINDER,
                    help="Modify hparams. Example: train.py resume out_dir TRAIN.BATCH_SIZE 2")


def main(out_dir: str, hparams: dict, pretrained: Optional[str], args):
    del args
    logging.info(f"PyTorch Lightning Training: {out_dir}")

    # Seed
    random.seed(hparams["TRAIN.SEED"])
    np.random.seed(hparams["TRAIN.SEED"])
    torch.manual_seed(hparams["TRAIN.SEED"])
    torch.cuda.manual_seed_all(hparams["TRAIN.SEED"])

    trainer_kwargs = {
        'log_row_interval': hparams["IO.LOG_INTERVAL"],
        'progress_bar_refresh_rate': hparams["IO.LOG_INTERVAL"],
        'log_save_interval': 10 * hparams["IO.LOG_INTERVAL"],
        'weights_summary': hparams["IO.WEIGHTS_SUMMARY"],
        'default_root_dir': out_dir,
        'default_save_path': out_dir,
        'min_epochs': hparams['TRAIN.EPOCHS'],
        'max_epochs': hparams['TRAIN.EPOCHS'],
        'benchmark': hparams['TRAIN.CUDNN_BENCHMARK']
    }

    if version.parse(pl.__version__) > version.parse("0.7.4-dev"):
        # TODO: Also other stuff breaks
        trainer_kwargs.pop('log_row_interval')

    hparams["IO.SAMPLES_PER_STEP"] = hparams["TRAIN.BATCH_SIZE"]
    if hparams["TRAIN.DEVICE"] == 'cuda':
        trainer_kwargs.update({'gpus': 1})
    if hparams["TRAIN.DEVICE"] in ('slurm-ddp', 'slurm_ddp'):
        num_nodes, gpus_per_node = slurm_ddp_setup()
        hparams['DDP.WORLD_SIZE'] = num_nodes * gpus_per_node
        trainer_kwargs.update({
            'gpus': gpus_per_node, 'num_nodes': num_nodes, 'distributed_backend': 'ddp'
        })
        logging.info(f"SLURM DDP: using {num_nodes} nodes with {gpus_per_node} GPUs each")
        hparams["IO.SAMPLES_PER_STEP"] = num_nodes * gpus_per_node * hparams["TRAIN.BATCH_SIZE"]
        if hparams['TRAIN.LR_DDP_SCALE_WITH_BATCH_SIZE'] is True:
            logging.info(f"SLURM DDP: scaling learning rate by {num_nodes * gpus_per_node}")
            hparams['TRAIN.LR'] = num_nodes * gpus_per_node * hparams['TRAIN.LR']
        # TODO: This is a quick fix to avoid race conditions with multiple nodes
        time.sleep(60)
    if hparams["TRAIN.DEVICE"].startswith('debug-ddp'):
        info = hparams["TRAIN.DEVICE"].split("-")
        assert len(info) == 4
        num_nodes, gpus_per_node = int(info[2]), int(info[3])
        hparams['DDP.WORLD_SIZE'] = num_nodes * gpus_per_node
        trainer_kwargs.update({
            'gpus': gpus_per_node, 'num_nodes': num_nodes, 'distributed_backend': 'ddp'
        })
        logging.info(f"DEBUG DDP: using {num_nodes} nodes with {gpus_per_node} GPUs each")
        hparams["IO.SAMPLES_PER_STEP"] = num_nodes * gpus_per_node * hparams["TRAIN.BATCH_SIZE"]
        if hparams['TRAIN.LR_DDP_SCALE_WITH_BATCH_SIZE'] is True:
            logging.info(f"DEBUG DDP: scaling learning rate by {num_nodes * gpus_per_node}")
            hparams['TRAIN.LR'] = num_nodes * gpus_per_node * hparams['TRAIN.LR']
        # TODO: This is a quick fix to avoid race conditions with multiple nodes
        time.sleep(10)

    # Add git hash
    git_commit = subprocess.check_output(['git', 'rev-parse', '--verify', 'HEAD'], stderr=subprocess.STDOUT)
    hparams['GIT.COMMIT'] = git_commit.decode().rstrip()

    # Just for us to know if we loaded weights
    hparams['TRAIN.PRETRAINED'] = pretrained

    # Hparams are now fixed
    logging.info(f"HPARAMS:")
    for k, v in sorted(hparams.items()):
        logging.info(f"  {k}: {v}")

    model = Tandem(hparams=hparams)

    # Seed again because different model architectures change seed. Make train samples consistent.
    # https://discuss.pytorch.org/t/shuffle-issue-in-dataloader-how-to-get-the-same-data-shuffle-results-with-fixed-seed-but-different-network/45357/9
    random.seed(hparams["TRAIN.SEED"])
    np.random.seed(hparams["TRAIN.SEED"])
    torch.manual_seed(hparams["TRAIN.SEED"])
    torch.cuda.manual_seed_all(hparams["TRAIN.SEED"])

    if pretrained:
        logging.info(f"Loading pretrained model weights model from {pretrained}")
        map_location = torch.device('cpu') if hparams['TRAIN.DEVICE'] == 'cpu' else None
        load_ckpt(model, pretrained, map_location=map_location)

    logger = TBLogger(out_dir=out_dir, hparams=hparams)
    checkpoint_callback = ModelCheckpoint(filepath=join(out_dir, 'ckpt', '{epoch:03d}'), save_top_k=-1)

    trainer = pl.Trainer(checkpoint_callback=checkpoint_callback, logger=logger, **trainer_kwargs)

    trainer.fit(model)


if __name__ == "__main__":
    main(*parse_args(parser))
