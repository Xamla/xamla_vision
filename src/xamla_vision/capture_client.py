from typing import Dict, List, Tuple

import rospy
from cv_bridge import CvBridge
import ximea_msgs.srv
import camera_aravis.srv

import numpy as np

from pathlib import Path
import cv2
import time


class XimeaCaptureClient(object):

    def __init__(self, serials: List[str], node_name: str='/ximea_mono'):
        self.bridge = CvBridge()
        self.serials = serials
        self.node_name = node_name

        send_command_service = self.node_name+'/send_command'
        try:
            rospy.wait_for_service(send_command_service, 3.0)

        except rospy.exceptions.ROSException as exc:
            raise RuntimeError('Capture client service: {} is not'
                               ' available'.format(send_command_service)) from exc

        self.send_command_proxy = rospy.ServiceProxy(send_command_service,
                                                     ximea_msgs.srv.SendCommand)

        capture_service = self.node_name+'/capture'
        try:
            rospy.wait_for_service(capture_service, 3.0)

        except rospy.exceptions.ROSException as exc:
            raise RuntimeError('Capture client service: {} is not'
                               ' available'.format(capture_service)) from exc

        self.capture_proxy = rospy.ServiceProxy(capture_service,
                                                ximea_msgs.srv.Capture)

    def __call__(self, exposure_time: float) -> Dict[str, np.ndarray]:
        exposure_req = ximea_msgs.srv.SendCommandRequest()
        exposure_req.command_name = 'setExposure'
        exposure_req.serials = self.serials
        exposure_req.value = exposure_time

        self.send_command_proxy(exposure_req)

        capture_req = ximea_msgs.srv.CaptureRequest()
        capture_req.serials = self.serials

        res = self.capture_proxy(capture_req)

        images = {}
        for i, img_msg in enumerate(res.images):
            images[res.serials[i]] = self.bridge.imgmsg_to_cv2(img_msg,
                                                               desired_encoding="passthrough")

        return images


class GeniCamCaptureClient(object):

    def __init__(self, serials: List[str], node_name: str='/camera_aravis_node'):
        self.bridge = CvBridge()
        self.serials = serials
        self.node_name = node_name

        send_command_service = self.node_name+'/send_command'
        try:
            rospy.wait_for_service(send_command_service, 3.0)

        except rospy.exceptions.ROSException as exc:
            raise RuntimeError('Capture client service: {} is not'
                               ' available'.format(send_command_service)) from exc

        self.send_command_proxy = rospy.ServiceProxy(send_command_service,
                                                     camera_aravis.srv.SendCommand)

        capture_service = self.node_name+'/capture'
        try:
            rospy.wait_for_service(capture_service, 3.0)

        except rospy.exceptions.ROSException as exc:
            raise RuntimeError('Capture client service: {} is not'
                               ' available'.format(capture_service)) from exc

        self.capture_proxy = rospy.ServiceProxy(capture_service,
                                                camera_aravis.srv.Capture)

    def __call__(self, exposure_time: float) -> List[np.ndarray]:
        exposure_req = camera_aravis.srv.SendCommandRequest()
        exposure_req.command_name = 'ExposureTime'
        exposure_req.serials = self.serials
        exposure_req.value = exposure_time

        self.send_command_proxy(exposure_req)

        capture_req = camera_aravis.srv.CaptureRequest()
        capture_req.serials = self.serials

        res = self.capture_proxy(capture_req)

        images = {}
        for i, img_msg in enumerate(res.images):
            images[res.serials[i]] = self.bridge.imgmsg_to_cv2(img_msg,
                                                               desired_encoding="passthrough")

        return images


class SimulatedCaptureClient(object):

    def __init__(self, picture_paths: Dict[str, List[str]], repeat: List[int]=None):
        self.picture_paths = {}
        self.pictures = {}
        for s, ps in picture_paths.items():
            self.picture_paths[s] = []
            self.pictures[s] = []
            for p in ps:
                path = Path(p)
                if not path.is_file():
                    raise RuntimeError('image at path: {} not '
                                       'exist'.format(path))
                self.picture_paths[s].append(path)
                self.pictures[s].append(cv2.imread(str(path),
                                                   cv2.IMREAD_GRAYSCALE))

        # self.repeat = None
        for s in self.pictures:
            if repeat is not None and len(self.pictures[s]) != len(repeat):
                raise RuntimeError('lenght of picture paths and repeat must'
                                   ' be equal')
        self.repeat = repeat

        self.index = 0
        self.count = 0

    def reset(self):
        self.index = 0
        self.count = 0

    def __call__(self, exposure_time: float) -> Dict[str, np.ndarray]:
        if self.repeat is not None:
            images = {}
            for s, p in self.pictures.items():
                images[s] = p[self.index]

            self.count += 1

            if self.count >= self.repeat[self.index]:
                self.count = 0
                self.index += 1
        else:
            images = {}
            for s, p in self.pictures.items():
                images[s] = p[self.index]

            self.index += 1

        for s, v in self.pictures.items():
            if self.index > len(v):
                raise RuntimeError('End of image list, please reset')

        time.sleep(0.05)

        self.images = images
        return images
