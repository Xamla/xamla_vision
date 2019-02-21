#!/usr/bin/env python3

import rospy
from xamla_vision.capture_client import GeniCamCaptureClient
from xamla_vision.stereo_laser_line_client import StereoLaserLineClient, StereoLaserLineClientException
from xamla_vision.srv import Scan, ScanRequest, ScanResponse
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32


class StereoLaserLineNode(object):
    def __init__(self):
        rospy.init_node('stereo_laser_line_node')
        calibration_file_path = rospy.get_param('~calibration_file_path')
        left_cam_serial = rospy.get_param('~left_cam_serial')
        right_cam_serial = rospy.get_param('~right_cam_serial')
        cam_with_laser_io = rospy.get_param('~cam_with_laser_io')
        laser_io_port = rospy.get_param('~laser_io_port')
        left_cam_client = GeniCamCaptureClient(serials=[left_cam_serial])
        right_cam_client = GeniCamCaptureClient(serials=[right_cam_serial])
        self.client = StereoLaserLineClient(calibration_file_path,
                                            left_cam_client,
                                            right_cam_client,
                                            camera_with_io=cam_with_laser_io,
                                            io_port=laser_io_port)

    def generate_point_cloud_callback(self, req: ScanRequest):
        print('call scan service')
        resp = ScanResponse()
        resp.error_code = 0
        resp.error_msg = 'success'
        resp.point_cloud = PointCloud()
        resp.point_cloud.header.frame_id = req.left_cam_frame_id
        resp.point_cloud.header.stamp = rospy.Time.now()
        resp.line_idx = []

        try:
            point_cloud, line_idx = self.client(None,
                                                [req.left_cam_exposure_time,
                                                 req.right_cam_exposure_time])
        except Exception as exc:
            if isinstance(exc, StereoLaserLineClientException):
                resp.error_code = exc.error_code
            else:
                resp.error_code = -10

            resp.error_msg = str(exc)

            return resp

        for p in point_cloud.T:
            point = Point32()
            point.x = p[0]
            point.y = p[1]
            point.z = p[2]
            resp.point_cloud.points.append(point)

        resp.line_idx = line_idx

        return resp


if __name__ == "__main__":
    node = StereoLaserLineNode()
    rospy.Service('~scan', Scan, node.generate_point_cloud_callback)
    rospy.spin()
