#!/usr/bin/env python3
name = "xamla_vision"

from .capture_client import (XimeaCaptureClient,
                             GeniCamCaptureClient,
                             SimulatedCaptureClient)
                             
from .stereo_laser_line_client import StereoLaserLineClient

