import os
from os.path import join
import argparse
import numpy as np
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader
from models.datasets import MVSDataset
from models import StageTensor, Tandem
from models.utils.helpers import to_device, tensor2numpy
import warnings

import cv2
import config as cfg

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", help="Path to save outputs to.", type=str, required=True)
parser.add_argument("--model", help="Path to .ckpt file.", type=str)
parser.add_argument("--data_dir", help="Path to replica data.", type=str)
parser.add_argument("--tuples_ext", help="Pose Extension.",
                    type=str, default="dso_optimization_windows")

parser.add_argument("--seed", help="Seed.", type=int, default=1)
parser.add_argument("--device", help="Torch device.",
                    type=str, choices=('cuda',), default='cuda')
parser.add_argument("--batch_size", help="Batch size.", type=int, default=1)

parser.add_argument("--view_num", help="Number of views", type=int, default=7)
parser.add_argument("--height", help="Image height.", type=int, default=480)
parser.add_argument("--width", help="Image width.", type=int, default=640)
parser.add_argument("--depth_min", help="Depth minimum.", type=float, default=0.01)
parser.add_argument("--depth_max", help="Depth maximum.", type=float, default=10.0)

parser.add_argument("--jit_freeze", action='store_true')
parser.add_argument("--jit_run_frozen_optimizations", action='store_true')
parser.add_argument("--jit_optimize_for_inference", action="store_true")

parser.add_argument("--profile", action="store_true")



# --- Some ressources ---#
#
# TorchScript Docs: https://pytorch.org/docs/stable/jit.html
# Tensor container: https://github.com/pytorch/pytorch/issues/36568#issuecomment-613638898
#
# Export Model on correct device:
#   https://github.com/pytorch/pytorch/blob/master/docs/source/jit.rst#frequently-asked-questions
#
# Issues related to tracing with input/output dict/tuple
#   Support output dict: https://github.com/pytorch/pytorch/issues/27743
#   Support output dict: https://github.com/pytorch/pytorch/pull/31860

class TensorContainer(nn.Module):
    def __init__(self, tensor_dict):
        super().__init__()
        for key, value in tensor_dict.items():
            setattr(self, key, value)


def tensors_save(fname: str, tensor_dict: dict):
    tensors = TensorContainer(tensor_dict)
    tensors = torch.jit.script(tensors)
    tensors.save(fname)


def list_inverse(l: list) -> list:
    assert sorted(l) == list(range(len(l)))
    y = [None] * len(l)
    for i, v in enumerate(l):
        y[v] = i

    for v in y:
        assert v is not None

    return y


def main(args: argparse.Namespace):
    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    hparams = cfg.default()
    hparams['DATA.ROOT_DIR'] = args.data_dir
    hparams['DATA.IMG_HEIGHT'] = args.height
    hparams['DATA.IMG_WIDTH'] = args.width
    hparams['TRAIN.BATCH_SIZE'] = args.batch_size
    hparams['TRAIN.DEVICE'] = args.device

    device = torch.device(hparams["TRAIN.DEVICE"])

    tandem = Tandem.load_from_checkpoint(args.model)
    tandem = tandem.to(device)
    tandem = tandem.eval()

    dataset = MVSDataset(
        root_dir=args.data_dir,
        split="val",
        pose_ext='dso',
        tuples_ext=args.tuples_ext,
        ignore_pose_scale=False,
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
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, drop_last=False, num_workers=6)
    for batch in loader:
        break
    batch = to_device(batch, device=device)

    # Adapt batch to number of views
    view_index = batch["view_index"][0].tolist()
    assert len(view_index) >= args.view_num
    start_index = len(view_index) - args.view_num
    inverse_view_index = list_inverse(view_index)[start_index:]
    view_index = [args.view_num - 2] + \
        list(range(args.view_num - 2)) + [args.view_num - 1]
    selection_index = [inverse_view_index[i] for i in view_index]

    model = tandem.cva_mvsnet
    del tandem

    for s in model.stages:
        if batch['intrinsics'][s]['K'].ndim == 4:
            assert all(torch.equal(batch['intrinsics'][s]['K'][:, 0], x) for x in torch.unbind(batch['intrinsics'][s]['K'], 1)), "Non equal intrinsics. C++ export not possible."
            batch['intrinsics'][s]['K'] = batch['intrinsics'][s]['K'][:, 0]

    batch = {
        'image': batch['image'][:, selection_index],
        'intrinsics': batch['intrinsics'],
        'cam_to_world': batch['cam_to_world'][:, selection_index],
        'depth_min': batch['depth_min'],
        'depth_max': batch['depth_max'],
        'view_index': torch.tensor([view_index])
    }
    del view_index, inverse_view_index, selection_index

    depth_filter_discard_percentage = torch.tensor([2.5])

    example_inputs = (
        batch['image'],
        StageTensor(*[batch['intrinsics'][s]['K'] for s in model.stages]),
        batch['cam_to_world'],
        batch['depth_min'],
        batch['depth_max'],
        depth_filter_discard_percentage
    )

    with torch.no_grad():
        outputs = model(*example_inputs)

    view_index = batch["view_index"][0].tolist()
    inverse_view_index = list_inverse(view_index)
    tensor_dict = {
        'image': batch['image'][:, inverse_view_index],
        'intrinsic_matrix.stage1': batch['intrinsics']['stage1']['K'],
        'intrinsic_matrix.stage2': batch['intrinsics']['stage2']['K'],
        'intrinsic_matrix.stage3': batch['intrinsics']['stage3']['K'],
        'cam_to_world': batch['cam_to_world'][:, inverse_view_index],
        'depth_min': batch['depth_min'],
        'depth_max': batch['depth_max'],
        'discard_percentage': depth_filter_discard_percentage
    }

    for i, stage in enumerate(model.stages):
        for j, key in enumerate(("depth", "confidence")):
            tensor_dict["outputs." + stage + "." + key] = outputs[i][j]

    os.makedirs(args.out_dir, exist_ok=True)
    tensors_save(join(args.out_dir, "sample_inputs.pt"), tensor_dict)

    depth = (tensor_dict["outputs.stage3.depth"][0] - tensor_dict['depth_min'][0]) / (
        tensor_dict['depth_max'][0] - tensor_dict['depth_min'][0])
    depth = tensor2numpy(depth)
    cv2.imwrite(join(args.out_dir, "depth.png"), (255 * depth).astype(np.uint8))

    confidence = tensor_dict["outputs.stage3.confidence"][0]
    confidence = tensor2numpy(confidence)
    cv2.imwrite(join(args.out_dir, "confidence.png"),
                (255 * confidence).astype(np.uint8))

    with torch.no_grad():
        traced_script_module = torch.jit.trace(model, example_inputs=example_inputs, check_trace=True,)
        if args.jit_freeze:
            print("--- jit_freeze")
            traced_script_module = torch.jit.freeze(traced_script_module)
            assert len(list(traced_script_module.named_parameters())) == 0

        if args.jit_run_frozen_optimizations:
            assert args.jit_freeze
            print("--- jit_run_frozen_optimizations (Currently this seems to have no perf effect)")
            torch.jit.run_frozen_optimizations(traced_script_module)

        if args.jit_optimize_for_inference:
            assert args.jit_freeze
            print("--- jit_optimize_for_inference (Currently this sometimes even has a negative perf effect)")
            traced_script_module = torch.jit.optimize_for_inference(traced_script_module)

        traced_script_module.save(join(args.out_dir, "model.pt"))

    if args.profile:
        from torch.profiler import profile, record_function, ProfilerActivity

        def trace_handler(p):
            output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
            print(output)
            p.export_chrome_trace(join(args.out_dir, "trace.json"))

        act = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        sched = torch.profiler.schedule(wait=1, warmup=1, active=2)
        with profile(activities=act, schedule=sched, on_trace_ready=trace_handler) as p:
            with torch.no_grad():
                for idx in range(8):
                    traced_script_module(*example_inputs)
                    # profiler will trace iterations 2 and 3, and then 6 and 7 (counting from zero)
                    p.step()


if __name__ == "__main__":
    main(parser.parse_args())
