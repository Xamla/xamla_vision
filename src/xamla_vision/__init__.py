#!/usr/bin/env python3
name = "xamla_vision"

from .capture_client import (XimeaCaptureClient,
                             GeniCamCaptureClient,
                             SimulatedCaptureClient)

from .crop_box_statistics_client import (CropBoxStatisticsProperties,
                                         CropBoxStatisticsClient)

from .stereo_laser_line_client import StereoLaserLineClient

