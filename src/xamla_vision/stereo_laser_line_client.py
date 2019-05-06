#!/usr/bin/env python3

import pathlib
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import rospy
import torchfile
from xamla_motion.data_types import Pose
from xamla_motion.utility import ROSNodeSteward

from camera_aravis.srv import SetIO, SetIORequest
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

from .capture_client import GeniCamCaptureClient


class StereoLaserLineClientException(Exception):
    """
    StereoLaserLineClient specific exception
    """

    def __init__(self, msg,
                 error_code, support=None,
                 original_exception=None):
        super(StereoLaserLineClientException, self).__init__(msg)
        self.error_code = error_code
        self.support = support
        self.original_exception = original_exception


class StereoLaserLineClient(object):

    """
    Client to project a laser line into the scene and
    reconstruct a point cloud of that line from
    the stereo images

    Methods
    -------
    exposure_time_search
        Search an exposure time to avoid over-/under-saturation of pixels
    __call__
        Project a laser line, do a stereo image and reconstruct the point cloud
    """

    def __init__(self, stereo_calibration_file_path: str,
                 left_camera_client: GeniCamCaptureClient,
                 right_camera_client: GeniCamCaptureClient,
                 camera_with_io: str,
                 io_port: int,
                 max_depth: float = 0.55,
                 axis: int = 1):
        """
        Parameters
        ----------
        stereo_calibration_file_path : str
            Path to the camera calibration file
        left_camera_client: GeniCamCaptureClient,
            The left camera client
        right_camera_client: GeniCamCaptureClient,
            The right camera client
        camera_with_io: str
            The serial of the camera with which the
            laser is controlled
        io_port: int
            IO port of the camera which enables and
            disables the laser
        max_depth: float (default 0.55)
            Defines the maximal distance in z-axis
            between left camera image plane and 3d point
            which is allowed.
        axis: int (default 1)
            The axis of the laser in image coordinates:
            1 for columns (y-axis), 0 for rows (x-axis)
        """

        path = pathlib.Path(stereo_calibration_file_path)

        if not path.is_file():
            raise FileNotFoundError('calibration file not found at'
                                    'location {}'.format(path))

        suffix = path.suffix

        if suffix == '.xml':
            self._from_xml(path)
        elif suffix == '.t7':
            self._from_t7(path)
        else:
            raise TypeError('only stereo calibration files with '
                            ' xml or t7 format are supported')

        (self.R1,
         self.R2,
         self.P1,
         self.P2,
         self.Q,
         self.roi1,
         self.roi2) = cv2.stereoRectify(self.camera_matrix_left,
                                        self.distortion_parameter_left,
                                        self.camera_matrix_right,
                                        self.distortion_parameter_right,
                                        self.size,
                                        self.left2right_rotation,
                                        self.left2right_translation)

        self.left_camera_id = left_camera_client.serials[0]
        self.right_camera_id = right_camera_client.serials[0]
        self.cameras = {self.left_camera_id:
                        {'client': left_camera_client},
                        self.right_camera_id:
                        {'client': right_camera_client}}

        self.cameras[self.left_camera_id]['exposure_time'] = None
        self.cameras[self.right_camera_id]['exposure_time'] = None

        self.max_depth = max_depth

        if camera_with_io not in self.cameras.keys():
            raise ValueError('camera with io: {} is not one of defined'
                             ' cameras {}'.format(camera_with_io,
                                                  self.cameras.keys()))
        self.camera_with_io = camera_with_io
        self.io_port = io_port

        self.axis = axis

        self.ros_node_steward = ROSNodeSteward()

        self.io_service = rospy.ServiceProxy(left_camera_client.node_name+'/set_io',
                                             SetIO)

        self.ransac = linear_model.RANSACRegressor(residual_threshold=0.005)

    def _from_xml(self, path: pathlib.Path):
        fs = cv2.FileStorage(str(path),
                             cv2.FileStorage_READ)
        try:
            self.camera_matrix_left = fs.getNode('Left_Kc').mat()
            self.camera_matrix_right = fs.getNode('Right_Kc').mat()

            self.distortion_parameter_left = fs.getNode('Left_kc').mat()
            self.distortion_parameter_right = fs.getNode('Right_kc').mat()

            self.left2right_rotation = fs.getNode('R').mat().astype(np.float64)
            self.left2right_translation = fs.getNode(
                'T').mat().astype(np.float64)

            self.size = (int(fs.getNode('FrameWidth').real()),
                         int(fs.getNode('FrameHeight').real()))
        finally:
            fs.release()

    def _from_t7(self, path: pathlib.Path):
        fs = torchfile.load(str(path))

        self.camera_matrix_left = fs.camLeftMatrix
        self.camera_matrix_right = fs.camRightMatrix

        self.distortion_parameter_left = fs.camLeftDistCoeffs
        self.distortion_parameter_right = fs.camRightDistCoeffs

        self.left2right_rotation = fs.R
        self.left2right_translation = fs.T

        self.size = (int(fs.imWidth), int(fs.imHeight))

    def _compute_3d_points(self, left_points, right_points):

        points_3d = cv2.triangulatePoints(self.P1, self.P2,
                                          left_points.T,
                                          right_points.T)

        return points_3d/points_3d[3, :]

    def _capture(self, with_laser: bool = False):
        io_req = SetIORequest()
        io_req.serials = [self.camera_with_io]
        io_req.io_port = self.io_port
        io_req.value = with_laser

        self.io_service.call(io_req)

        result = {}
        for k, v in self.cameras.items():
            result.update(v['client'](float(v['exposure_time'])))

        if with_laser:
            io_req.value = False
            self.io_service.call(io_req)

        return result

    def exposure_time_search(self):
        """ Searches exposure time.

        Make sure that no pixel is saturated by keeping the pixel with
        the highest intensity between 220 and 250.
        """

        for k, v in self.cameras.items():
            if v['exposure_time'] is None:
                v['exposure_time'] = 3000

        for i in range(40):
            result = self._capture(with_laser=True)

            converged = [False]*len(self.cameras)
            for k, (serial, image) in enumerate(result.items()):
                max_value = image.max()
                if max_value < 100:
                    self.cameras[serial]['exposure_time'] += 1000
                elif max_value < 200:
                    self.cameras[serial]['exposure_time'] += 500
                elif max_value > 254:
                    self.cameras[serial]['exposure_time'] -= 300
                elif max_value < 220:
                    self.cameras[serial]['exposure_time'] += 80
                elif max_value > 250:
                    self.cameras[serial]['exposure_time'] -= 40
                else:
                    converged[k] = True

            if all(converged):
                print('found optimal exposure_times:'
                      ' {}'.format([s+' exposure time '+str(e['exposure_time'])
                                    for s, e in self.cameras.items()]))
                return {k: v['exposure_time'] for k, v in self.cameras.items()}

        raise StereoLaserLineClientException('no suitable exposure'
                                             ' time could be found',
                                             error_code=-1)

    def _undistort_points(self, point_dict):
        l = cv2.undistortPoints(np.array(point_dict[self.left_camera_id])[None, ...],
                                cameraMatrix=self.camera_matrix_left,
                                distCoeffs=self.distortion_parameter_left,
                                R=self.R1,
                                P=self.P1).squeeze()

        r = cv2.undistortPoints(np.array(point_dict[self.right_camera_id])[None, ...],
                                cameraMatrix=self.camera_matrix_right,
                                distCoeffs=self.distortion_parameter_right,
                                R=self.R2,
                                P=self.P2).squeeze()

        point_dict[self.left_camera_id] = l
        point_dict[self.right_camera_id] = r

    def _find_correspondences(self, point_dict, left_exists):
        left_points = point_dict[self.left_camera_id][:, self.axis]
        right_points = point_dict[self.right_camera_id][:, self.axis]

        diff = np.abs(left_points[:, None] - right_points)
        index = diff.argmin(axis=1)
        min_v = diff.min(axis=1)

        left_points = point_dict[self.left_camera_id]
        right_points = point_dict[self.right_camera_id]
        new_lp = []
        new_rp = []

        exists = []
        for i, idx in enumerate(index):
            if min_v[i] > 0.51:
                continue

            exists.append(left_exists[i])
            new_lp.append(left_points[i, :])
            new_rp.append(right_points[idx, :])

        point_dict[self.left_camera_id] = np.array(new_lp)
        point_dict[self.right_camera_id] = np.array(new_rp)

        return exists

    def _valid_check(self, left_exists: List[int], img_size: Tuple[int, int]):
        if len(left_exists) == 0:
            raise StereoLaserLineClientException('no valid laser'
                                                 ' line points found',
                                                 error_code=-2,
                                                 support=len(left_exists))
        # elif self.axis == 1 and len(left_exists) < img_size[0]/2:
        #     raise StereoLaserLineClientException('less valid points than half'
        #                                          ' of image height {} of '
        #                                          '{}'.format(len(left_exists),
        #                                                      img_size[0]/2),
        #                                          error_code=-3,
        #                                          support=len(left_exists))
        # elif self.axis == 0 and len(left_exists) < img_size[1]/2:
        #     raise StereoLaserLineClientException('less valid points than half'
        #                                          ' of image width {} of '
        #                                          '{}'.format(len(left_exists),
        #                                                      img_size[1]/2),
        #                                          error_code=-3,
        #                                          support=len(left_exists))

    def __call__(self, left_cam_pose: Union[None, Pose]=None,
                 exposure_times: Union[None, Tuple[int, int]]=None,
                 ransac_filter: bool=True, debug_mode: bool=False):
        """
        Get laser line points.

        This captures two stereo image pairs with and without laser line and
        triangulates laser line 3d points from it.

        Parameters
        ----------
        left_cam_pose : Union[None, Pose]
            When left_cam_pose is not None, the points are calculated in world view.
            Else, they are relative to the origin of the left camera.
        exposure_times: Union[None, Tuple[int, int]]=None
            When exposure_times is not None, the exposure is set to the values given
            in the Tuple.
            Else, they are calculated using exposure_time_search.
        ransac_filter: bool (default True)
            Indicate if a ransac filter is used to filter out outlier.
            If ransac_filter is True, the points are filtered using the ransac method
            to avoid points diverging too much in x-direction (from the view of the plane
            spanned by the two cameras).
            This assumes that when projecting the laser line onto the plane orthoganally,
            the resulting curve is a straight line in y-direction, so points diverging in
            x-direction must be outliers.
        debug_mode: bool (default False)
            if True show captured diff images of left and right camera and
            laser line 3d points

        Returns
        -------
        np.ndarray
            the point cloud
        np.ndarray
            image row or colum in which the 3d point was found (left camera image)
        """

        if exposure_times is not None:
            self.cameras[self.left_camera_id]['exposure_time'] = exposure_times[0]
            self.cameras[self.right_camera_id]['exposure_time'] = exposure_times[1]

        if (self.cameras[self.left_camera_id]['exposure_time'] is None or
                self.cameras[self.right_camera_id]['exposure_time'] is None):
            self.exposure_time_search()

        p_r = self._capture(with_laser=False)

        r = self._capture(with_laser=True)

        diff_images = {}
        laser_line_points = {k: [] for k in self.cameras.keys()}
        for serial, image in r.items():
            p_img = p_r[serial]

            # diff image with threshold
            diff_img = image.astype(float) - p_img.astype(float)

            diff_img[diff_img < 10] = 0.0

            diff_img = diff_img.astype(np.uint8)

            if debug_mode is True:
                diff_images[serial] = diff_img

            idx = np.argmax(diff_img, axis=self.axis)
            if self.axis == 0:
                max_values = diff_img[idx, np.arange(diff_img.shape[1])]
            else:
                max_values = diff_img[np.arange(diff_img.shape[0]), idx]

            start_idx = idx-20
            start_idx[start_idx < 0] = 0
            end_idx = idx+20

            if self.axis == 1:
                end_idx[end_idx >= self.size[0]] = self.size[0]-1
            else:
                end_idx[end_idx >= self.size[1]] = self.size[1]-1

            points = []

            for i, (start, end) in enumerate(zip(start_idx, end_idx)):
                if max_values[i] < 20:
                    continue  # in this image line is no laser line

                if self.axis == 0:
                    weights = diff_img[start:end+1, i]
                else:
                    weights = diff_img[i, start:end+1]

                mid_p = np.average(np.arange(start, end+1),
                                   weights=weights)

                if self.axis == 0:
                    points.append((i, mid_p))
                else:
                    points.append((mid_p, i))

            laser_line_points[serial] = points

        left_exists = [p[self.axis] for p in
                       laser_line_points[self.left_camera_id]]

        self._valid_check(left_exists, diff_img.shape)

        self._undistort_points(laser_line_points)
        left_exists = self._find_correspondences(laser_line_points,
                                                 left_exists)

        self._valid_check(left_exists, diff_img.shape)

        points_3d = self._compute_3d_points(laser_line_points[self.left_camera_id],
                                            laser_line_points[self.right_camera_id])

        X = np.array(left_exists)[:, np.newaxis]
        features = PolynomialFeatures(degree=5)
        X_feat = features.fit_transform(X)

        if ransac_filter is True:
            if self.axis == 1:
                self.ransac.fit(X_feat,
                                points_3d[0, :])
            else:
                self.ransac.fit(X_feat,
                                points_3d[1, :])
            left_exists = np.array(left_exists)[self.ransac.inlier_mask_]
            points_3d = points_3d[:, self.ransac.inlier_mask_]
        else:
            left_exists = np.array(left_exists)

        rot = np.eye(4)
        rot[:3, :3] = self.R1
        inv_rot = np.linalg.inv(rot)
        points_3d = np.matmul(inv_rot, points_3d)

        self._valid_check(left_exists, diff_img.shape)

        if np.any(points_3d[2, :] > self.max_depth):
            raise StereoLaserLineClientException('measured depth larger than '
                                                 'specified max depth {}'
                                                 ''.format(self.max_depth),
                                                 error_code=-4)

        result_points_3d = None

        if left_cam_pose is not None:
            m = left_cam_pose.transformation_matrix()
            result_points_3d = np.matmul(m, points_3d)[:3, :]
        else:
            result_points_3d = points_3d[:3, :]

        if debug_mode is True:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            fig_diff_images = plt.figure()
            ax = fig_diff_images.add_subplot(211)
            ax.imshow(
                diff_images[self.left_camera_id],
                cmap='gray', vmin=0, vmax=255
            )
            ax.set_title(self.left_camera_id)
            ax = fig_diff_images.add_subplot(212)
            ax.imshow(
                diff_images[self.right_camera_id],
                cmap='gray', vmin=0, vmax=255
            )
            ax.set_title(self.right_camera_id)

            fig_3d_points = plt.figure()
            ax = fig_3d_points.add_subplot(211, projection='3d')

            ax.scatter(
                result_points_3d[0, :],
                result_points_3d[1, :],
                result_points_3d[2, :],
                c='r', marker='o'
            )

            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')

            ax = fig_3d_points.add_subplot(212)
            ax.plot(
                result_points_3d[1, :],
                result_points_3d[2, :],
                'ro'
            )

            ax.set_xlabel('Y Label')
            ax.set_ylabel('Z Label')
            ax.set_title('3d laser line points')
            plt.show()

        return result_points_3d, left_exists
