import asyncio
from typing import Callable, Dict, List, Tuple

import actionlib
import numpy as np
import rospy
from xamla_motion.utility import ROSNodeSteward
from xamla_motion.data_types import Pose
from ximea_msgs.msg import CropBoxStatisticsAction, CropBoxStatisticsGoal, CropBox
from ximea_msgs.srv import (CropBoxStatisticsSetDefaultCrop, CropBoxStatisticsSetDefaultCropRequest,
                            SlstudioSetup, SlstudioSetupRequest)

# Add timer
# from laundrometer_python.utility import Timer

from xamla_motion.motion_service import generate_action_executor


class CropBoxStatisticsProperties(object):
    def __init__(self, default_crop_box: CropBox= None,
                 crop_box_statistics_action_path: str = None,
                 slstudio_service_topic: str = None,
                 crop_box_statistics_service_topic: str = None,
                 stereo_calibration_file_path: str = None,
                 camera_serials: List[str] = None)->None:
        self._crop_box_statistics_action_path = crop_box_statistics_action_path or '/ximea_mono/slstudio/cropBoxStatistics'
        self._slstudio_service_topic = slstudio_service_topic or '/ximea_mono/slstudio/setupSlstudio'
        self._crop_box_statistics_service_topic = crop_box_statistics_service_topic or '/ximea_mono/slstudio/setDefaultCropCropBoxStatistics'
        self._stereo_calibration_file_path = stereo_calibration_file_path or '/home/xamla/Rosvita.Control/projects/laundrometer/calibration/current/StereoCalibration.xml'
        self._camera_serials = camera_serials or [
            'CAMAU1639042', 'CAMAU1710001']
        self._default_crop_box = default_crop_box

    @property
    def crop_box_statistics_action_path(self):
        return self._crop_box_statistics_action_path

    @property
    def slstudio_service_topic(self):
        return self._slstudio_service_topic

    @property
    def crop_box_statistics_service_topic(self):
        return self._crop_box_statistics_service_topic

    @property
    def stereo_calibration_file_path(self):
        return self._stereo_calibration_file_path

    @property
    def camera_serials(self):
        return self._camera_serials

    @property
    def default_crop_box(self):
        return self._default_crop_box


class CropBoxStatisticsClient(object):
    def __init__(self, properties: CropBoxStatisticsProperties, logger=None, setup_slstudio=False):
        # initialization
        # needs to be performed only once
        self.logger = logger
        self._properties = properties
        self._ros_node_steward = ROSNodeSteward()
        rospy.wait_for_service(properties.slstudio_service_topic)
        rospy.wait_for_service(properties.crop_box_statistics_service_topic)
        if setup_slstudio:
            try:
                self.setup_slstudio_service = rospy.ServiceProxy(
                    properties.slstudio_service_topic, SlstudioSetup)
                request = SlstudioSetupRequest()
                request.stereo_configuration_file_path = properties.stereo_calibration_file_path
                request.shutter_speed_in_ms = 50
                request.serials = properties.camera_serials
                request.crop_parameter = []
                resp = self.setup_slstudio_service(request)
            except rospy.ServiceException as e:
                print("Service call failed: %s" % e)
                raise rospy.ServiceException('connection to setup_slstudio'
                                             ' server could not be established')

        if self._properties.default_crop_box:
            self.set_default_crop_box(self._properties.default_crop_box)

        # call height analysis action
        client = actionlib.SimpleActionClient(
            properties.crop_box_statistics_action_path, CropBoxStatisticsAction)
        if not client.wait_for_server(rospy.Duration(5)):
            raise rospy.ServiceException('connection to robot CropBoxStatisticsAction'
                                         ' server could not be established')
        self.client = generate_action_executor(client)

    @property
    def properties(self):
        return self._properties

    def set_default_crop_box(self, crop_box: CropBox):
        try:
            crop_box_statistics_set_default_crop = rospy.ServiceProxy(
                self._properties.crop_box_statistics_service_topic,
                CropBoxStatisticsSetDefaultCrop
            )

            request = CropBoxStatisticsSetDefaultCropRequest()
            request.crop_box = crop_box
            resp = crop_box_statistics_set_default_crop(request)
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)
            raise rospy.ServiceException('connection to CropBoxStatistics set default crop'
                                         ' server could not be established')

    async def __call__(self, crop_boxes: List[CropBox], transform: Pose, shutter_speed_in_ms: int=50):
        # TODO: Add timer 
        # with Timer('computeCropBoxStatistics', self.logger):
        goal = CropBoxStatisticsGoal()
        goal.crop_boxes = crop_boxes
        goal.cloud_transform = transform.to_pose_msg()
        goal.shutter_speed_in_ms = float(shutter_speed_in_ms)
        responds = await self.client(goal)

        result = responds.result()
        if result.success == False:
            raise rospy.TransportException('Failed to get images')

        return result.number_of_points
        # TODO: Add timer
