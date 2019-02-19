import rospy
from xamla_vision.stereo_laser_line_client import StereoLaserLineClient, StereoLaserLineClientException


def stereo_laser_line_node():
    rospy.init_node('stereo_laser_line_node')
    print(rospy.get_param('calibration_file_path'))
    rospy.spin()


if __name__ == "main":
    stereo_laser_line_node()
