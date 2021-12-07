# Replica-TANDEM-Ext
This dataset is an extension of the excellent [Replica dataset](https://github.com/facebookresearch/Replica-Dataset) by Straub et al. Replica-TANDEM-Ext extends the original Replica dataset by generating realistic trajectories for visual SLAM. For convenience, we offer folders including images, depth maps, the train-val split, and poses. Note that all files derived from the Replica dataset are subject to their license given at the end of the document. Please consider the dataset specification below.

## Download
The following table gives the download links for each version together with a version description. The first version number gives the major release, e.g. `1.` for the TANDEM CoRL paper. The second version number indicates the minor release, related to bug fixes or additional files. The description would indicate breaking changes, however, we hope to avoid these. Releases that end in `.beta` are beta and can be changed or deleted, while other releases will stay available. We give the MD5 sum computed with `md5sum (GNU coreutils) 8.30` in the last column. Upon clicking the link the download starts automatically and you can use `wget` or `curl` to download.

| Version Number | Version Description | Link | `md5sum` |
|:---------------|:--------------------|:-----|:---------|
| 1.1.beta | Initial Beta Release. | https://vision.in.tum.de/webshare/g/tandem/data/tandem_replica_1.1.beta.zip | `4696c986ed3947ada0b07dcb43d58c69` |



## Dataset Format
The root folder contains one folder for each scene, e.g. `apartment_0` and `hotel_0`. It also contains the `LICENSE` file and the train-val split in `train.txt` and `val.txt`. Each scene contains the folders `images` and `depths`. Each scene contains the camera intrinsics `camera.txt`, ground truth poses `poses_gt.txt`, poses from Direct Sparse Odometry (DSO) `poses_dso.txt` and tuple files, e.g. `tuples_dso_optimization_windows.txt`. A tuple file gives the frame indices that should be used for training and evaluation of the Multi-view Stereo (MVS) methods.

**General**

* All indices start at 0 and are zero padded to 6 places.
* In all text files, all lines that start with `#` are treated as comments.
* Any pose is given by a 4x4 SE(3) matrix unrolled row-wise separated by space. All poses are `CamToWorld`.

```
tandem_replica/
    train.txt
    val.txt
    LICENSE
    scene_name/
        camera.txt
        poses_gt.txt
        poses_dso.txt
        tuples_dso_optimization_windows.txt
        tuples_dso_optimization_windows_last3.txt
        depths/
            scale.txt
            frame_index.png
        images/
            frame_index.jpg
```

#### camera.txt
The intrinsic camera parameters in the ["DSO format"](https://github.com/JakobEngel/dso#calibration-file-for-pre-rectified-images) that we copy below for convenience. The last two lines are only used by DSO and can be ignored, the images in the dataset have size `1280 x 960`.
```
Pinhole fx fy cx cy 0
in_width in_height
"crop" / "full" / "none" / "fx fy cx cy 0"
out_width out_height
```

#### poses_gt.txt
Each line has the frame index followed by the pose, so one int and then 16 floats.

#### poses_dso.txt
The output from running DSO in the same format as `poses_gt.txt`. Because DSO is a monocular method there is a SIM(3) (rotation, translation, and scale) transform between the ground truth and this trajectory.

#### tuples_dso_optimization_windows.txt
Each line contains one MVS configuration for training. The first column is the number of views in this configuration, which is 8 because we use the optimization windows. Then the indices of the views. The final number is the scale between the ground truth poses and the DSO poses in the tuple. This is necessary to correctly scale the depth. The scale is obtained by aligning the DSO trajectory with the ground truth trajectory by using the [TUM RGB-D evaluation tools](https://vision.in.tum.de/data/datasets/rgbd-dataset/tools).

#### tuples_dso_optimization_windows_last3.txt
Same format as `tuples_dso_optimization_windows.txt`. Contains just the last 3 indices for each line. This file is used to test the influence of using 3 views or the whole optimization window.

#### depths
The depth is stored as `uint16` and the global scale is a single float in the scale.txt. The depth is loaded like `depth_meters = scale * cv2.imread('depth/000000.png', -1).astype(np.float)`.

## Bibtex
Please consider citing Replica and TANDEM if you use this dataset.
```
@article{replica19arxiv,
  title =   {The {R}eplica Dataset: A Digital Replica of Indoor Spaces},
  author =  {Julian Straub and Thomas Whelan and Lingni Ma and Yufan Chen and Erik Wijmans and Simon Green and Jakob J. Engel and Raul Mur-Artal and Carl Ren and Shobhit Verma and Anton Clarkson and Mingfei Yan and Brian Budge and Yajie Yan and Xiaqing Pan and June Yon and Yuyang Zou and Kimberly Leon and Nigel Carter and Jesus Briales and  Tyler Gillingham and  Elias Mueggler and Luis Pesqueira and Manolis Savva and Dhruv Batra and Hauke M. Strasdat and Renzo De Nardi and Michael Goesele and Steven Lovegrove and Richard Newcombe },
  journal = {arXiv preprint arXiv:1906.05797},
  year =    {2019}
}
@inproceedings{tandem2021corl,
 author = {Lukas Koestler and Nan Yang and Niclas Zeller and Daniel Cremers},
 title = {{TANDEM}: Tracking and Dense Mapping in Real-time using Deep Multi-view Stereo},
 booktitle = {Conference on Robot Learning ({CoRL})},
 year = {2021}
}
```

## License
Our additions to the Replica dataset are published under the BSD-3-clause license. All files derived from the original replica dataset, e.g. the images and depth maps, are licensed under the original Replica license, which can be found [here](https://github.com/facebookresearch/Replica-Dataset/blob/main/LICENSE) and which we replicate here for convenience. The full license file can be found in the downloaded data.
```
Replica Dataset Research Terms

Before Facebook Technologies, LLC (“FB”) is able to offer you (“Researcher” or
“You”) access to the Replica Dataset (the “Dataset”), please read the following
agreement (“Agreement”).

By accessing, and in exchange for receiving permission to access, the Dataset,
Researcher hereby agrees to the following terms and conditions:
1.  Researcher may use, modify, improve and/or publish the Dataset only in
connection with a research or educational purpose that is non-commercial or
not-for-profit in nature, and not for any other purpose.
2.  Researcher may provide research associates and colleagues with access to the
Dataset provided that they first agree to be bound by these terms and
conditions.
3.  Researcher may use the Dataset in the scope of their employment at a
for-profit or commercial entity provided that Researcher complies with Section 1
of this Agreement. If Researcher is employed by a for-profit or commercial
entity, Researcher's employer shall also be bound by these terms and conditions,
and Researcher hereby represents that they are fully authorized to enter into
this agreement on behalf of such employer.
4.  THE DATASET IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL FB OR ANY
CONTRIBUTOR BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE DATASET OR THE USE OR OTHER DEALINGS IN THE DATASET.
5.  The law of the State of California shall apply to all disputes related to
this Dataset.

FAQ:
Q: I work for a commercial research organization. Can I use this data?
A: Yes! We intend for employees of commercial research organizations to use
this data as long as the purpose in using the data is for a research or
educational purpose.
```
