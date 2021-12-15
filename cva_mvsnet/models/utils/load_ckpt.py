import torch
from ..tandem import Tandem
from ..cva_mvsnet import CvaMVSNet
import logging


def _change_key(key, m):
    matches = tuple(k for k in m if key.startswith(k))
    assert len(matches) == 1, f"Existance and Uniqueness Error." \
                              f" Matches = {matches}, key = {key}"
    return m[matches[0]] + key[len(matches[0]):]


def _map_dict(d, m):
    return {_change_key(key, m): d[key] for key in d}


def _load_cascade_stereo_ckpt(model, state_dict):
    if isinstance(model, Tandem):
        model = model.cva_mvsnet
    elif isinstance(model, CvaMVSNet):
        pass
    else:
        logging.warning(f"Model is of unknown type {type(model)}")
    map_old_to_new = {f"cost_regularization.{i}": f"cost_regularization_net.stage{i + 1}" for i in range(3)}
    map_old_to_new.update({f"feature.conv{i}": f"feature_net.conv{i}" for i in range(3)})
    map_old_to_new.update({f"feature.out{i}": f"feature_net.out.stage{i}" for i in range(1, 4)})
    map_old_to_new.update({f"feature.inner{i}": f"feature_net.skip.stage{i + 1}" for i in range(1, 3)})

    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict=_map_dict(state_dict, map_old_to_new), strict=True)
    assert len(missing_keys) == 0 and len(unexpected_keys) == 0


def load_ckpt(model: Tandem, fname: str, map_location=None):
    ckpt = torch.load(fname, map_location=map_location)  # type: dict
    if 'state_dict' in ckpt and 'hparams' in ckpt:
        # Pytorch Lightning Checkpoint
        missing_keys, unexpected_keys = model.load_state_dict(ckpt['state_dict'], strict=True)
        assert len(missing_keys) == 0 and len(unexpected_keys) == 0

    elif set(ckpt.keys()) == {'epoch', 'model', 'optimizer'}:
        # Cascade Stereo Checkpoint
        _load_cascade_stereo_ckpt(model, state_dict=ckpt['model'])

    else:
        raise RuntimeError(
            f"Could not load ckpt {fname} because it seems to neither from pytorch lightning nor from cascade stereo.")
