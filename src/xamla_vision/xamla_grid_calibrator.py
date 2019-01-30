import cv2 as cv
import os
import pickle
from xamla_motion.data_types import Pose
from pyquaternion import Quaternion
from scipy.linalg import sqrtm, inv


class XamlaGridCalibrator(object):
    def __init__(self, pattern_localizer, capture_client,
                 left_serial, right_serial, exposure_time, world_view_client,
                 world_view_folder, move_group, hand_eye, storage_folder,
                 flange_link_name, offset_pattern_screw_on_table,
                 pattern_point_to_pattern_edge):
        self.pattern_localizer = pattern_localizer
        self.capture_client = capture_client
        self.left_serial = left_serial
        self.right_serial = right_serial
        self.exposure_time = exposure_time
        self.world_view_client = world_view_client
        self.world_view_folder = world_view_folder
        self.move_group = move_group
        self.hand_eye = hand_eye
        self.storage_folder = storage_folder
        self.flange_link_name = flange_link_name
        self.offset_pattern_screw_on_table = offset_pattern_screw_on_table
        self.pattern_point_to_pattern_edge = pattern_point_to_pattern_edge

    def addOrUpdatePose(self, display_name, folder, pose):
        try:
            get_value = self.world_view_client.get_pose(display_name, folder)
            self.world_view_client.update_pose(display_name, folder, pose)
        except:
            self.world_view_client.add_pose(display_name, folder, pose)

    def runGridCalibration(self):
        completed = False
        counter = 0
        try:
            self.world_view_client.add_folder('debug', self.world_view_folder)
        except:
            print(
                'Could not create {}/debug in world view.'.format(self.world_view_folder))

        if not os.path.isdir(self.storage_folder):
            os.makedirs(self.storage_folder)

        while (not completed):
            command = input(
                'Move robot to measure point #{} (0 to end): '.format(counter))
            if command == '0':
                completed = True
            else:
                x_coordinate = input(
                    'Enter X coordinate (integer) of Xamla Grid Point: ')
                y_coordinate = input(
                    'Enter Y coordinate (integer) of Xamla Grid Point: ')

                images = self.capture_client(self.exposure_time)
                image_left = cv.cvtColor(
                    images[self.left_serial], cv.COLOR_GRAY2RGB)
                image_right = cv.cvtColor(
                    images[self.right_serial], cv.COLOR_GRAY2RGB)

                cv.imwrite(os.path.join(self.storage_folder,
                                        '{}_left.png'.format(counter)), image_left)
                cv.imwrite(os.path.join(self.storage_folder,
                                        '{}_right.png'.format(counter)), image_right)

                current_posture = self.move_group.get_current_joint_positions()
                current_pose = self.move_group.motion_service.query_pose(
                    self.move_group.name, current_posture, self.flange_link_name)
                with open(os.path.join(self.storage_folder, '{}_posture.p'.format(counter)), 'wb') as f:
                    pickle.dump(current_posture, f)
                with open(os.path.join(self.storage_folder, '{}_pose.p'.format(counter)), 'wb') as f:
                    pickle.dump(current_pose, f)

                camPoseFinal, circlesGridPointsLeft, circlesGridPointsRight, pointsInCamCoords = self.pattern_localizer.calcCamPoseViaPlaneFit(
                    image_left, image_right, "left", False)
                camera_pose = current_pose * self.hand_eye
                pattern_pose = camera_pose * camPoseFinal
                #screw_center_on_table = Pose(pattern_pose.translation + self.offset_pattern_screw_on_table.translation, pattern_pose.rotation)
                screw_center_on_table = pattern_pose * \
                    self.pattern_point_to_pattern_edge * self.offset_pattern_screw_on_table
                # rotate Z upwards
                screw_center_on_table = screw_center_on_table.rotate(
                    Quaternion([0, 1, 1, 0]))

                display_name = 'GridPoint{}_{}{}'.format(
                    '%03d' % counter, x_coordinate, y_coordinate)
                self.addOrUpdatePose(
                    display_name, self.world_view_folder, screw_center_on_table)
                self.addOrUpdatePose('Camera{}'.format(
                    '%03d' % counter), self.world_view_folder + '/debug', camera_pose)
                self.addOrUpdatePose('Pattern{}'.format(
                    '%03d' % counter), self.world_view_folder + '/debug', pattern_pose)

                counter += 1

    def runPatternCalibration(self):
        # determined by measuring using a ruler
        offset_to_edge = Pose([-0.0145, -0.0142, 0])
        edge_to_screw_center_table = Pose([0.045, 0.045, 0.02333])
        return offset_to_edge + edge_to_screw_center_table
