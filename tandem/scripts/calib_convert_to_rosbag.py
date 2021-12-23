import numpy as np
import rospy
import rosbag
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped
import cv2
import json
from cv_bridge import CvBridge
import sys


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Call like python calib_convert_to_rosbag TANDEM_CALIB_DIR"
    path = sys.argv[1]
    print("Path = ", path)
    bag = rosbag.Bag(path+"/calib.bag", "w")

    timestamps = np.loadtxt(path+"/timestamps_sec.txt")
    timestamps = timestamps - timestamps[0]
    num_images = timestamps.size
    print(f"Mean dt={int(1000*np.mean(np.diff(timestamps)))} ms")

    print("."*20)
    for i in range(num_images):
        img = cv2.imread(f"{path}/images/{i:06d}.png", cv2.IMREAD_GRAYSCALE)
        assert img is not None
        assert img.dtype == np.uint8
        stamp = rospy.Time(secs=int(timestamps[i]), nsecs=int(1e9*(timestamps[i]%1)))
        image_ros = Image()
        image_ros.header.stamp = stamp
        image_ros.height = img.shape[0]
        image_ros.width = img.shape[1]
        image_ros.encoding =  "mono8"
        image_ros.data = img.tobytes()
        bag.write("cam0/image_raw", image_ros, stamp)

        if ((i+1) % (num_images//20) == 0):
            print(".", end="")
    print()

    bag.close()
