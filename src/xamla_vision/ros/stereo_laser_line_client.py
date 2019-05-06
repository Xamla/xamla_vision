import re
import rospy
import numpy as np
from xamla_vision.srv import Scan, ScanRequest, ScanResponse
from xamla_vision.stereo_laser_line_client import StereoLaserLineClientException
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32


def _point_cloud_to_numpy_array(point_cloud: PointCloud):

    number_of_points = len(point_cloud.points)
    points = np.zeros((3, number_of_points),
                      dtype=float)
    for i, p in enumerate(point_cloud.points):
        points[0, i] = p.x
        points[1, i] = p.y
        points[2, i] = p.z

    return points

# naming conflict: Also the implementation of the stereo laser line
# triangulation is named StereoLaserLineClient, this must be resolved


class StereoLaserLineClient(object):
    def __init__(self, server_node_name: str):
        self._server_node_name = server_node_name

        if (re.sub('[^A-Za-z0-9]+', '', rospy.get_name()) == 'unnamed'):
            rospy.init_node('StereoLaserLineClient', anonymous=True)

        self._scan_service = self._server_node_name+'/scan'

        try:
            rospy.wait_for_service(self._scan_service, 3.0)

        except rospy.exceptions.ROSException as exc:
            raise RuntimeError('StereoLaserLine scan service: {} is not'
                               ' available'.format(self._scan_service)) from exc

    @property
    def server_node_name(self):
        return self._server_node_name

    def __call__(self,
                 left_cam_exposure_time: int,
                 right_cam_exposure_time: int,
                 left_cam_frame_id: str,
                 debug_mode: bool=False):
        """
        Request laser line scan

        Parameters
        ----------
        left_cam_exposure_time: int
            left camera exposure time for line laser scan in micro seconds
        right_cam_exposure_time: int
            right camera exposure time for line laser scan in micro seconds
        left_cam_frame_id: str
            frame id of left camera cooridnate system
        debug_mode: bool (default False)
            show diff images and 3d point cloud and stop execution until
            windows are closed

        Result
        ------
        point_cloud: np.ndarray
            point cloud as point in a shape np.array((3, number_of_points) dtype=float)
        line_idx: np.array
            left camera imager row indicies of point cloud points
        support: int
            number of found point cloud points

        Raises
        ------
        rospy.ServiceException
            If scan service call fails
        StereoLaserLineClientException
            If result of laser line scan is bad
        """

        scan_service_handle = rospy.ServiceProxy(self._scan_service,
                                                 Scan)

        req = ScanRequest()
        req.left_cam_exposure_time = left_cam_exposure_time
        req.right_cam_exposure_time = right_cam_exposure_time
        req.left_cam_frame_id = left_cam_frame_id
        req.debug_mode = debug_mode

        resp = scan_service_handle(req)

        if resp.success is True:

            point_cloud = _point_cloud_to_numpy_array(resp.point_cloud)
            line_idx = np.array(resp.line_idx)
            support = int(resp.support)

            return point_cloud, line_idx, support

        else:
            raise StereoLaserLineClientException(msg=resp.error_msg,
                                                 error_code=resp.error_code,
                                                 support=resp.support)
