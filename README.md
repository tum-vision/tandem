<h1 align="center">TANDEM: Tracking and Dense Mapping<br>in Real-time using Deep Multi-view Stereo</h1>
<p align="center">
    <a href="https://lukaskoestler.com">Lukas Koestler</a><sup>1*</sup> &emsp;&emsp;
    <a href="https://vision.in.tum.de/members/yangn">Nan Yang</a><sup>1,2*,&dagger;</sup> &emsp;&emsp;
    <a href="https://www.niclas-zeller.de">Niclas Zeller</a><sup>2,3</sup> &emsp;&emsp;
    <a href="https://vision.in.tum.de/members/cremers">Daniel Cremers</a><sup>1,2</sup>
</p>

<p align="center">
    <sup>*</sup>equal contribution&emsp;&emsp;&emsp;
    <sup>&dagger;</sup>corresponding author
</p>

<p align="center">
    <sup>1</sup>Technical University of Munich&emsp;&emsp;&emsp;
    <sup>2</sup>Artisense<br>
    <sup>3</sup>Karlsruhe University of Applied Sciences
</p>

<p align="center">
    Conference on Robot Learning (CoRL) 2021, London, UK
</p>
<p align="center">
    <a href="https://3dv2021.surrey.ac.uk/prizes">3DV 2021 Best Demo Award</a>
</p>
<p align="center">
    <a href="https://arxiv.org/abs/2111.07418">arXiv</a> |
    <a href="https://youtu.be/L4C8Q6Gvl1w">Video</a> |
    <a href="https://openreview.net/forum?id=FzMHiDesj0I">OpenReview</a> |
    <a href="https://go.vision.in.tum.de/tandem">Project Page</a>
</p>

## Overview
This repository is mostly based on [this](https://github.com/rancheng/tandem) repository by rancheng and official [TANDEM](https://github.com/tum-vision/tandem) repository.. Main changes include adding a `custom_dockerfile` to complete the existing `image` and adding support for automatically cloning and building [tandem](https://github.com/tum-vision/tandem) repository.

Instruction can be found inside `/docker`.

## Code and Data
- [x] 📣 C++ code released before Christmas! Please check [tandem/](tandem/).
- [x] 📣 CVA-MVSNet released! Please check [cva_mvsnet/](cva_mvsnet/).
- [x] 📣 Replica training data released! Please check [replica/](replica/).
- [x] Minor improvements throughout January. **Contributions are highly welcomed!**
- [x] Release of the ScanNet-trained model 
- [ ] Docker image for TANDEM. **Contributions are highly welcomed!**


### Abstract
<p align="justify">In this paper, we present TANDEM a real-time monocular tracking and dense mapping framework. For pose estimation, TANDEM performs photometric bundle adjustment based on a sliding window of keyframes. To increase the robustness, we propose a novel tracking front-end that performs dense direct image alignment using depth maps rendered from a global model that is built incrementally from dense depth predictions. To predict the dense depth maps, we propose Cascade View-Aggregation MVSNet (CVA-MVSNet) that utilizes the entire active keyframe window by hierarchically constructing 3D cost volumes with adaptive view aggregation to balance the different stereo baselines between the keyframes. Finally, the predicted depth maps are fused into a consistent global map represented as a truncated signed distance function (TSDF) voxel grid. Our experimental results show that TANDEM outperforms other state-of-the-art traditional and learning-based monocular visual odometry (VO) methods in terms of camera tracking. Moreover, TANDEM shows state-of-the-art real-time 3D reconstruction performance.</p>


### Poster
<p align="center">
  <img src="assets/tandem_poster.jpg">
</p>
