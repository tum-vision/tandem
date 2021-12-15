from typing import Optional, NamedTuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import uniform_depth_range, adaptive_depth_range, FeatureNet, CostRegNet, depth_prediction, \
    depth_filter_edges

StageOutputs = NamedTuple(
    "StageOutputs",
    [("depth", torch.Tensor), ("confidence", torch.Tensor), ("depth_dense", torch.Tensor),
     ("confidence_dense", torch.Tensor)]
)
Outputs = NamedTuple(
    "Outputs",
    [("stage1", StageOutputs), ("stage2", StageOutputs), ("stage3", StageOutputs)]
)
StageTensor = NamedTuple(
    "StageTensor",
    [("stage1", torch.Tensor), ("stage2", torch.Tensor), ("stage3", torch.Tensor)]
)


class CvaMVSNet(nn.Module):
    def __init__(self,
                 depth_num: tuple = (48, 32, 8),
                 depth_interval_ratio: tuple = (1.0, 0.5, 0.25),
                 feature_net_base_channels: int = 8,
                 cost_volume_base_channels: tuple = (8, 8, 8),
                 view_aggregation: bool = False,
                 conv2d_normalization: str = "batchnorm",
                 conv2d_use_bn_skip: bool = False,
                 conv3d_normalization: str = "batchnorm"):
        """
        All tuples have the order: (stage1, stage2, stage3)
        :param depth_num: tuple, number of depths per stage
        :param depth_interval_ratio: tuple, interval_ration: depth_interval = base_interval * ratio[stage]
            Base_interval = (depth_max-depth_min) / (depth_num[0] - 1) and depth_max, depth_min are supplied in the
            forward method. Therefore, depth_interval_ratio is w.r.t. the depth interval of stage1 and we thus require
            depth_interval_ratio[0] == 1.
        :param feature_net_base_channels: int
        :param cost_volume_base_channels: tuple, number of base channels of the respective cost volume
        """
        assert len(depth_num) == 3, "Currently only implemented for 3 stages"
        assert len(depth_num) == len(depth_interval_ratio) == len(cost_volume_base_channels), "Same number of stages"
        assert depth_interval_ratio[0] == 1, f"Interval ratio is w.r.t. stage 1. See docstring of this method."

        super(CvaMVSNet, self).__init__()
        self.stage_num = len(depth_num)
        self.stages = tuple(f"stage{idx}" for idx in range(1, self.stage_num + 1))

        self.depth_num = {stage: depth_num[idx] for idx, stage in enumerate(self.stages)}
        self.depth_interval_ratio = {stage: depth_interval_ratio[idx] for idx, stage in enumerate(self.stages)}
        self.cost_volume_base_channels = {
            stage: cost_volume_base_channels[idx] for idx, stage in enumerate(self.stages)}
        self.scale = {stage: 2 ** (self.stage_num - idx - 1) for idx, stage in enumerate(self.stages)}

        self.view_aggregation = view_aggregation

        self.feature_net = FeatureNet(base_channels=feature_net_base_channels,
                                      normalization=conv2d_normalization,
                                      use_bn_skip=conv2d_use_bn_skip,
                                      last_stage=3)
        self.cost_regularization_in_channels = self.feature_net.out_channels

        self.cost_regularization_net = nn.ModuleDict({stage: CostRegNet(
            in_channels=self.cost_regularization_in_channels[stage],
            base_channels=self.cost_volume_base_channels[stage],
            normalization=conv3d_normalization,
            has_four_depths=self.depth_num[stage] == 4
        ) for stage in self.stages})

        if self.view_aggregation:
            in_channels = {stage: self.cost_regularization_in_channels[stage] for stage in self.stages}

            self.volume_gates = nn.ModuleDict({stage: nn.Sequential(
                nn.Conv3d(in_channels[stage], 1, kernel_size=1, stride=1, bias=True),
                nn.BatchNorm3d(1),
                nn.ReLU(inplace=True),
                nn.Conv3d(1, 1, kernel_size=1, stride=1, bias=True),
                nn.BatchNorm3d(1),
                nn.ReLU(inplace=True)
            ) for stage in self.stages})

    @staticmethod
    def stage_prev(stage: str):
        assert stage.startswith('stage') and len(stage) == 6 and int(stage[-1]) > 1
        return f"stage{int(stage[-1]) - 1}"

    def outputs_to_dict(self, outputs: Outputs) -> Dict[str, Dict[str, torch.Tensor]]:
        res = {}
        for i, stage in enumerate(self.stages):
            out = outputs[i]  # type: StageOutputs
            res[stage] = {'depth': out.depth, 'confidence': out.confidence}

        return res

    def forward(self,
                image: torch.Tensor,
                intrinsic_matrix: StageTensor,
                cam_to_world: torch.Tensor,
                depth_min: torch.Tensor,
                depth_max: torch.Tensor,
                depth_filter_discard_percentage: Optional[torch.Tensor] = None) -> Outputs:
        """
        :param image: (B, V, C, H, W)
        :param intrinsic_matrix: StageTensor [stage] : (B, 3, 3) intrinsic matrix
        :param cam_to_world: (B, V, 4, 4)
            cam_to_world in SE(3) cam_to_world[0, 0] = [R t; 0 1] R is 3x3, t is 3x1, 0 is 1x3, 1 is 1x1
        :param depth_min: (B, ) Min depth value of the scene
        :param depth_max: (B, ) Max depth value of the scene
        :return:
        """
        # Setup
        align_corners_range = False
        batch_size, view_num, channels, height, width = list(image.size())  # (B, V, C, H, W)

        # Image Stack
        image_stack = torch.reshape(image, (
            batch_size * view_num, channels, height, width))  # (B*V, C_stage, H_stage, W_stage)

        # Feature Extraction
        features = {}
        feature_stack = self.feature_net(image_stack)  # (B*V, F, H_stage, W_stage)

        for stage in feature_stack:
            _, c_stage, h_stage, w_stage = list(feature_stack[stage].shape)
            feature_stack_stage = torch.reshape(feature_stack[stage], (
                batch_size, view_num, c_stage, h_stage, w_stage))  # (B, V, C_stage, H_stage, W_stage)
            features[stage] = torch.unbind(feature_stack_stage, dim=1)  # (V) (B, C_stage, H_stage, W_stage)

        outputs, depth_base_interval = {}, None
        for stage_idx, stage in enumerate(self.stages):
            # Depth Range Sampling
            if stage == 'stage1':
                # For the first stage use uniform depth samples
                depth_range_samples, depth_base_interval = uniform_depth_range(
                    depth_num=self.depth_num[stage], depth_min=depth_min, depth_max=depth_max,
                    height=height // self.scale[stage],
                    width=width // self.scale[stage])  # (B, D_stage, H_stage, W_stage), (B,)
            else:
                # For later stages use adaptive depth samples
                curr_depth = outputs[self.stage_prev(stage)]['depth'].detach()  # (B, H_stage//2, W_stage//2)
                curr_depth = F.interpolate(
                    curr_depth.unsqueeze(1), (height // self.scale[stage], width // self.scale[stage]),
                    mode='bilinear', align_corners=align_corners_range
                ).squeeze(1)  # (B, H_stage, W_stage)

                depth_range_samples = adaptive_depth_range(
                    depth=curr_depth, depth_num=self.depth_num[stage],
                    interval=self.depth_interval_ratio[stage] * depth_base_interval
                )  # (B, D_stage, H_stage, W_stage)

            # Warping and Cost Regularization
            outputs[stage] = depth_prediction(
                features=features[stage],
                depth_in=depth_range_samples,
                intrinsics=intrinsic_matrix[stage_idx],
                cam_to_world=cam_to_world,
                cost_regularization=self.cost_regularization_net[stage],
                training=self.training, half_pixel_centers=False,
                volume_gate=self.volume_gates[stage] if self.view_aggregation else None,
            )  # {'depth': (B, H_stage, W_stage), 'confidence':  (B, H_stage, W_stage)}

        # Must be done AFTER all the stages
        if depth_filter_discard_percentage is not None:
            for stage in self.stages:
                outputs[stage]['depth_dense'] = 1.0 * outputs[stage]['depth']
                outputs[stage]['confidence_dense'] = 1.0 * outputs[stage]['confidence']

                outputs[stage]['depth'], mask = depth_filter_edges(outputs[stage]['depth'],
                                                                   discard_percentage=depth_filter_discard_percentage)
                outputs[stage]['confidence'][mask] = 0
        else:
            for stage in self.stages:
                outputs[stage]['depth_dense'] = 1.0 * outputs[stage]['depth']
                outputs[stage]['confidence_dense'] = 1.0 * outputs[stage]['confidence']

        # Convert to NamedTuple
        out = Outputs(
            *[StageOutputs(depth=outputs[s]['depth'], confidence=outputs[s]['confidence'],
                           depth_dense=outputs[s]['depth_dense'], confidence_dense=outputs[s]['confidence_dense'],
                           ) for s in self.stages])
        return out
