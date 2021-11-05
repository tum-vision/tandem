<center><h1>TANDEM: Tracking and Dense Mapping<br>in Real-time using Deep Multi-view Stereo</h1></center>
<center>Lukas Koestler<sup>1\*</sup> &emsp;&emsp;Nan Yang<sup>1,2\*</sup> &emsp;&emsp;Niclas Zeller<sup>2,3</sup> &emsp;&emsp;Daniel Cremers<sup>1,2</sup></center>

<center>
    <sup>1</sup>Technical University of Munich&emsp;&emsp;&emsp;
    <sup>2</sup>Artisense<br>
    <sup>3</sup>Karlsruhe University of Applied Sciences
</center>

<center>
    <span style="color:gray">Conference on Robot Learning (CoRL) 2021</span>
</center>


## Abstract
<div style="text-align: justify;">In this paper, we present TANDEM a real-time monocular tracking and dense mapping framework. For pose estimation, TANDEM performs photometric bundle adjustment based on a sliding window of keyframes. To increase the robustness, we propose a novel tracking front-end that performs dense direct image alignment using depth maps rendered from a global model that is built incrementally from dense depth predictions. To predict the dense depth maps, we propose Cascade View-Aggregation MVSNet (CVA-MVSNet) that utilizes the entire active keyframe window by hierarchically constructing 3D cost volumes with adaptive view aggregation to balance the different stereo baselines between the keyframes. Finally, the predicted depth maps are fused into a consistent global map represented as a truncated signed distance function (TSDF) voxel grid. Our experimental results show that TANDEM outperforms other state-of-the-art traditional and learning-based monocular visual odometry (VO) methods in terms of camera tracking. Moreover, TANDEM shows state-of-the-art real-time 3D reconstruction performance.</div>


## Poster
<p align="center">
  <img src="assets/tandem_poster.jpg">
</p>


For more details, please see:

* [OpenReview](https://openreview.net/forum?id=FzMHiDesj0I) for the full paper and supplementary material.

* Webpage: [go.vision.in.tum.de/tandem](https://go.vision.in.tum.de/tandem) for further information.

* Demo Video: [YouTube](https://youtu.be/L4C8Q6Gvl1w) for a live presentation of TANDEM.

* Code and data coming soon. We are currently preparing a live demo for CoRL and will update this repository afterwards. Thank you for your patience.
