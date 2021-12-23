# TANDEM - DEMO
This README describes the setup of the TANDEM live demo using a Realsense D455, which has global shutter. TANDEM only uses the mono RGB image and ignores the depth and IMU data. It should be straightforward to adapt the code to another monocular, global shutter RGB camera. Make sure that the camera is connected to a USB3 port, trying different ports can help a lot. Also having a physically good USB3 connection is important.

**Please consider the section "General Notes for Good Results" from the main README**

## Build & Run

* The Realsense camera requires `librealsense2` for interfacing and we used version `2.49.0-0~realsense0.5306`. We used the method based on `librealsense2-dkms` (see their [github](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md#installing-the-packages)) for getting the kernel patches on the kernel `5.3.0-46-generic` with Ubuntu `18.04.5 LTS`.

* Follow the instructions in the README.md to run `cmake` in the build folder an then run `make tandem_demo -j` to build the executable.

* Calibrate your camera as explained below. **Warning**: The conversion mechanism for the Realsense parameters is a very coarse approximation and performance will be much degraded. This will be used if you omit `calib=X` from the command line. It is only there so that one can quickly check if the interface works.

* Run the demo from the base folder (**for good results please provide your calibration with `calib=X`**)
```
build/bin/tandem_demo \
    preset=demo \
    result_folder=results/demo \
    mvsnet_folder=exported/tandem_512x320 \
    mode=1 \
    demo_secs=60
```


## Calibration
A good geometric calibration is paramount to achieve good results. We use the excellent calibration software from [basalt](https://gitlab.com/VladyslavUsenko/basalt/-/blob/master/doc/Calibration.md).

* Build the recorder with `make realsense_calib_recorder -j`

* Make a directory for results `export TANDEM_CALIB_DIR=path/to/calibration dir && mkdir -p $TANDEM_CALIB_DIR`

* Record images: `build/bin/realsense_calib_recorder num_images=600 output_dir=$TANDEM_CALIB_DIR frame_skip=5`. If you have never recorded images for calibration please consider the [basalt documentation](https://gitlab.com/VladyslavUsenko/basalt/-/blob/master/doc/Calibration.md) or [kalibr's wiki](https://github.com/ethz-asl/kalibr/wiki). Images have to be **at least**: well light, taken from many angles, cover all portions of the image, taken under slow motion and thus have minimal rolling shutter. It is often good to place the calibration pattern, **which must be perfectly flat**, on the floor or a wall and move the camera.

* Setup a python environment with ROS packages to use the `scripts/calib_convert_to_rosbag.py` script. We used e.g. `pip install --extra-index-url https://rospypi.github.io/simple/ sensor_msgs` to install pure python version of ROS packages without setting up a full ROS environment, see their [github](https://github.com/rospypi/simple). [RoboStack](https://github.com/RoboStack) seems to be another alternative that we haven't tried.

* Convert the images to a rosbag used by basalt: `python scripts/calib_convert_to_rosbag.py $TANDEM_CALIB_DIR`

* Install `basalt_calibrate`: we used the `apt` [installation](https://gitlab.com/VladyslavUsenko/basalt#apt-installation-for-ubuntu-2004-and-1804-fast)

* Run the calibration: `basalt_calibrate --dataset-path $TANDEM_CALIB_DIR/calib.bag --dataset-type bag --aprilgrid /usr/etc/basalt/aprilgrid_6x6.json --result-path $TANDEM_CALIB_DIR --cam-types kb4` and follow the instructions [in the docs](https://gitlab.com/VladyslavUsenko/basalt/-/blob/master/doc/Calibration.md). We are done after the intrinsics have been optimized and saved (`save_calib` button).

* Convert to the `camera.txt` format needed by TANDEM: `python scripts/calib_convert_to_txt.py $TANDEM_CALIB_DIR`. This script also considers the down-scaling from `1280x800` to `512x300`. The demo is run in resolution `512x300` for better performance. Doing the calibration in resolution `1280x800` gives more accurate results.
