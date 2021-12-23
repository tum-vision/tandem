# TANDEM
This folder contains the C++ TANDEM code, which can be used for evaluation of the tracking performance and mesh reconstruction quality. Most of this code is based on the excellent [DSO](https://github.com/JakobEngel/dso) by Jakob Engel and thus inherits it's GPL-3.0 license. The subfolder `libdr` contains code that is not derived from DSO and hence has different licenses.

### 1. Installation
We list the required dependencies for TANDEM in `1.1` and further dependencies in `1.2`. Also consider the full [README](https://github.com/JakobEngel/dso/blob/master/README.md) of DSO.

#### 1.1 TANDEM Dependencies
For the paper we used CUDA 11.1, cuDNN 8.0.5, and `libtorch-1.9.0+cu111` with CUDA support and CXX11 ABI. However, we assume that the code should work with a broad range of versions because it doesn't use version-specific features. We can sadly not offer a convenient installation script due to (a) different CUDA installation options and (b) the cuDNN download method that needs user input. You have to:

+ Install **CUDA** from [nvidia.com](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html). Make sure that this doesn't interfere with other packages on your system and maybe ask your system administrator. CUDA 11.1 should be in your path, e.g. setting `CUDA_HOME`, `LD_LIBRARY_PATH` and `PATH` should be sufficient. You can also set the symlink `/usr/local/cuda`.

+ Install **cuDNN** from [nvidia.com](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html). Make sure to install a version that exactly matches your CUDA and PyTorch versions.
```
export TANDEM_CUDNN_LIBRARY=/path/to/cudnn/lib64
export TANDEM_CUDNN_INCLUDE_PATH=/path/to/cudnn/include
```

+ Install **LibTorch** from [pytorch.org](https://pytorch.org/get-started/locally/). For our exact version you can use
```
wget https://download.pytorch.org/libtorch/cu111/libtorch-cxx11-abi-shared-with-deps-1.9.0%2Bcu111.zip
export TANDEM_LIBTORCH_DIR=/path/to/unziped/libtorch
```


#### 1.2 Further Dependencies

##### suitesparse and eigen3 (required).
We use suitesparse version `1:5.1.2-2` and eigen version `3.3.7` corresponding to `worl.major.minor`.

    sudo apt install libsuitesparse-dev libeigen3-dev libboost-all-dev


##### OpenCV (required).
We use version `3.2.0`.

    sudo apt install libopencv-dev


##### Pangolin (optional: highly recommended).
Used for 3D visualization & the GUI. Install from [https://github.com/stevenlovegrove/Pangolin](https://github.com/stevenlovegrove/Pangolin).
We use version `0.5`.


### 2. Build
CMake build in release mode
```
mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=$TANDEM_LIBTORCH_DIR \
    -DCUDNN_LIBRARY=$TANDEM_CUDNN_LIBRARY \
    -DCUDNN_INCLUDE_PATH=$TANDEM_CUDNN_INCLUDE_PATH
make -j
```

**Notes:**

* Building CUDA Code: Correctly setting up CUDA and CMake is not easy, especially since `FindCUDA` was deprecated in cmake version 3.10. We will try to clean up the CMakeLists.txt and maybe offer a docker-based installation to ease these issues in the future. If you have good resources or know how to do it well, please reach out. Because CUDA isn't cross-platform and thus the full functionality of cmake isn't necessary, maybe Makefiles would have been the better option.
* Sometimes the code crashes with this error `OpenGL Error: XX (502)`, which we couldn't resolve so far (help appreciated). Usually, a reboot fixes the issue.

### 3. Evaluation
For evaluation on EuRoC we provide pre-processed data, which follows the replica-TANDEM-Ext format, and running and evaluation scripts. 

* Download and unzip `euroc_tandem_format` and `export EUROC_TANDEM_FORMAT=/path/to/euroc_tandem_format` (see below)
* Setup a python 2 environment with `numpy matplotlib` that can run the `tum_rgbd_eval_tools` and `export TANDEM_PY2=$HOME/miniconda3/envs/py2/bin/python`
* **Warning**: The scripts will create and remove folders in `results/tracking/dense/euroc`. The tracking script will overwrite previous results
* Run the tracking from the base folder `scripts/tracking_euroc.bash`
* Run the evaluation from the base folder `scripts/tracking_euroc_eval.bash`
* Run the runtime experiment from the base folder `scripts/runtime_euroc.bash`. The FPS will be given at the end of `out.txt` in the `result_folder`.

**Download:**

The following table gives the download links for each version together with a version description. The first version number gives the major release, e.g. `1.` for the TANDEM CoRL paper. The second version number indicates the minor release, related to bug fixes or additional files. The description would indicate breaking changes, however, we hope to avoid these. Releases that end in `.beta` are beta and can be changed or deleted, while other releases will stay available. We give the MD5 sum computed with `md5sum (GNU coreutils) 8.30` in the last column. Upon clicking the link the download starts automatically and you can use `wget` or `curl` to download.

| Name | Version Number | Version Description | Link | `md5sum` |
|:-----|:---------------|:--------------------|:-----|:---------|
| EuRoC| 1.1.beta | Initial Beta Release. | https://vision.in.tum.de/webshare/g/tandem/data/euroc_tandem_format_1.1.beta.zip | `9793f342df0c4cc66cac25ee135290c3` |

### 4. General Usage
Run on a general sequence. Section 4.1 that describes the dataset format is copied from the original DSO for convenience. The Commandline options described in Section 4.2 are different from DSO.
```
    build/bin/tandem_dataset \
      preset=dataset \
      result_folder=path/to/save/results \
      files=path/to/scene/images \
      calib=path/to/scene/camera.txt \
      mvsnet_folder=path/to/exported/mvsnet \
      mode=1
```

#### 4.1 Dataset Format.
The format assumed is that of [https://vision.in.tum.de/mono-dataset](https://vision.in.tum.de/mono-dataset).
However, it should be easy to adapt it to your needs, if required. The binary is run with:

- `files=XXX` where XXX is either a folder or .zip archive containing images. They are sorted *alphabetically*.

- `gamma=XXX` where XXX is a gamma calibration file, containing a single row with 256 values, mapping [0..255] to the respective irradiance value, i.e. containing the *discretized inverse response function*. See TUM monoVO dataset for an example.

- `vignette=XXX` where XXX is a monochrome 16bit or 8bit image containing the vignette as pixelwise attenuation factors. See TUM monoVO dataset for an example.

- `calib=XXX` where XXX is a geometric camera calibration file. See below.



##### Geometric Calibration File.


###### Calibration File for Pre-Rectified Images

    Pinhole fx fy cx cy 0
    in_width in_height
    "crop" / "full" / "none" / "fx fy cx cy 0"
    out_width out_height

###### Calibration File for FOV camera model:

    FOV fx fy cx cy omega
    in_width in_height
    "crop" / "full" / "fx fy cx cy 0"
    out_width out_height


###### Calibration File for Radio-Tangential camera model

    RadTan fx fy cx cy k1 k2 r1 r2
    in_width in_height
    "crop" / "full" / "fx fy cx cy 0"
    out_width out_height


###### Calibration File for Equidistant camera model

    EquiDistant fx fy cx cy k1 k2 k3 k4
    in_width in_height
    "crop" / "full" / "fx fy cx cy 0"
    out_width out_height


(note: for backwards-compatibility, "Pinhole", "FOV" and "RadTan" can be omitted). See the respective
`::distortCoordinates` implementation in  `Undistorter.cpp` for the exact corresponding projection function.
Furthermore, it should be straight-forward to implement other camera models.


**Explanation:**
Across all models `fx fy cx cy` denotes the focal length / principal point **relative to the image width / height**,
i.e., DSO computes the camera matrix `K` as

        K(0,0) = width * fx
        K(1,1) = height * fy
        K(0,2) = width * cx - 0.5
        K(1,2) = height * cy - 0.5
For backwards-compatibility, if the given `cx` and `cy` are larger than 1, DSO assumes all four parameters to directly be the entries of K,
and omits the above computation.


**That strange "0.5" offset:**
Internally, DSO uses the convention that the pixel at integer position (1,1) in the image, i.e. the pixel in the second row and second column,
contains the integral over the continuous image function from (0.5,0.5) to (1.5,1.5), i.e., approximates a "point-sample" of the
continuous image functions at (1.0, 1.0).
In turn, there seems to be no unifying convention across calibration toolboxes whether the pixel at integer position (1,1)
contains the integral over (0.5,0.5) to (1.5,1.5), or the integral over (1,1) to (2,2). The above conversion assumes that
the given calibration in the calibration file uses the latter convention, and thus applies the -0.5 correction.
Note that this also is taken into account when creating the scale-pyramid (see `globalCalib.cpp`).


**Rectification modes:**
For image rectification, DSO either supports rectification to a user-defined pinhole model (`fx fy cx cy 0`),
or has an option to automatically crop the image to the maximal rectangular, well-defined region (`crop`).
`full` will preserve the full original field of view and is mainly meant for debugging - it will create black
borders in undefined image regions, which DSO does NOT ignore (i.e., this option will generate additional
outliers along those borders, and corrupt the scale-pyramid).


#### 4.2 Commandline Options
The option `preset` must be specified first. Following options overwrite the preset partially.

- `preset=X`: Specify the overall preset. Needs to be the first argument and is required.
    - `preset=dataset` use for evaluating the tracking performance on a dataset offline. This will not be real-time because the dense tracking uses a CPU implementation that is more deterministic and because images are not preloaded to account for systems with less RAM.
    - `preset=gui` use for qualitatively evaluating the tracking/reconstruction performance on a dataset offline. This will not be real-time due to the reasons mentioned in `dataset` and additionally due to the gui.
    - `preset=runtime` use for evaluating the runtime performance on a dataset offline. This achieves ca. 21 FPS on euroc/V101.


For all options please `main_tandem_pangolin.cpp`. Some options are

- `mode=X`:
    -  `mode=0` use iff a photometric calibration exists (e.g. TUM monoVO dataset).
    -  `mode=1` use iff NO photometric calibration exists (e.g. ETH EuRoC MAV dataset).
    -  `mode=2` use iff images are not photometrically distorted (e.g. synthetic datasets).

- `result_folder=X`: Folder to which the results will be saved.

- `mvsnet_folder=X`: Folder that contains the exported MVSNet.

- `tracking=X`
    - `tracking=sparse` use sparse tracking like DSO
    - `tracking=dense:cpu` use dense tracking from TANDEM on the CPU. Not real-time but more deterministic
    - `tracking=dense:cuda` use dense tracking from TANDEM on the GPU. Real-time but less deterministic

- `tsdf_fusion=X`: Use TSDF fusion or not. TSDF fusion is required for dense tracking.

- `mesh_extraction_freq=X`: A mesh will be extracted and displayed when `mvs_frame_id % mesh_extraction_freq == 0`, where `mvs_frame_id` is the number of times the MVSNet was involved before. Using `mesh_extraction_freq >= 5` is recommended for performance and `mesh_extraction_freq = 0` will not extract a mesh. However, a mesh will always be saved after a successful run to `result_folder/mesh.obj`

- `nolog=1`: disable logging of eigenvalues etc. (good for performance)

- `reverse=1`: play sequence in reverse

- `nogui=1`: disable gui (good for performance)

- `nomt=1`: single-threaded execution

- `prefetch=1`: load into memory & rectify all images before running.

- `start=X`: start at frame X

- `end=X`: end at frame X

- `quiet=1`: disable most console output (good for performance)


### 5 General Notes for Good Results

#### Reconstruction
- The MVSNet implicitly relies on triangulation through the architecture of CVA-MVSNet, hence it is important to move the camera s.t. there is sufficient baseline for points that should be reconstructed well. Slow camera motion is helpful as it reduces drift and motion blur.

- The MVSNet implicitly relies on texture to reconstruct geometry. If a surface has close to no texture, the reconstruction will be less accurate. Some texture on the surface can be sufficient for the network to correctly estimate the whole surface if it is e.g. a wall.

#### Accurate Geometric Calibration
- Please have a look at Chapter 4.3 from the DSO paper, in particular Figure 20 (Geometric Noise). Direct approaches suffer a LOT from bad geometric calibrations: Geometric distortions of 1.5 pixel already reduce the accuracy by factor 10.

- **Do not use a rolling shutter camera**, the geometric distortions from a rolling shutter camera are huge. Even for high frame-rates (over 60fps).

- Note that the reprojection RMSE reported by most calibration tools is the reprojection RMSE on the "training data", i.e., overfitted to the the images you used for calibration. If it is low, that does not imply that your calibration is good, you may just have used insufficient images.

- Try different camera / distortion models, not all lenses can be modelled by all models.


#### Tracking: Photometric Calibration
Use a photometric calibration (e.g. using [https://github.com/tum-vision/mono_dataset_code](https://github.com/tum-vision/mono_dataset_code) ).

#### Translation vs. Rotation
DSO cannot do magic: if you rotate the camera too much without translation, it will fail. Since it is a pure visual odometry, it cannot recover by re-localizing, or track through strong rotations by using previously triangulated geometry.... everything that leaves the field of view is marginalized immediately.


#### Initialization
The current initializer is from DSO and it is not very good, i.e. it is very slow and occasionally fails.
Make sure, the initial camera motion is slow and "nice" (i.e., a lot of translation and 
little rotation) during initialization. Possibly replace by your own initializer.

### 6 License
TANDEM was developed at the Technical University of Munich and partially builds on DSO.
The open-source version is licensed under the GNU General Public License Version 3 (GPLv3).

### 7 Bibtex
Please consider citing DSO and TANDEM if you use this code. Please cite the EuRoC paper if you use their data.
```
@inproceedings{engel2016dso,
 author = {J. Engel and V. Koltun and D. Cremers},
 title = {Direct Sparse Odometry},
 booktitle = {arXiv:1607.02565},
 arxiv = {arXiv:1607.02565},
 year = {2016},
}
@article{Burri25012016,
  author = {Burri, Michael and Nikolic, Janosch and Gohl, Pascal and Schneider, Thomas and Rehder, Joern and Omari, Sammy and Achtelik, Markus W and Siegwart, Roland}, 
  title = {The EuRoC micro aerial vehicle datasets},
  year = {2016}, 
  journal = {The International Journal of Robotics Research} 
}
@inproceedings{tandem2021corl,
 author = {Lukas Koestler and Nan Yang and Niclas Zeller and Daniel Cremers},
 title = {{TANDEM}: Tracking and Dense Mapping in Real-time using Deep Multi-view Stereo},
 booktitle = {Conference on Robot Learning ({CoRL})},
 year = {2021}
}
```
