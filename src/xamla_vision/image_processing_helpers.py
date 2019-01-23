import cv2
import numpy as np
from typing import List, Iterable
from xamla_motion.data_types import Pose

import xamla_vision.x3d as x3d


class ImageProcessingException(Exception):
    """
    Base banner loop exception to enable lib depended
    exception handling
    """

    def __init__(self, msg, original_exception=None):
        super(ImageProcessingException, self).__init__(msg)
        self.original_exception = original_exception


class DynamicRangeException(ImageProcessingException):
    """
    Base banner loop exception to enable lib depended
    exception handling
    """

    def __init__(self, msg, original_exception=None):
        super(DynamicRangeException, self).__init__(msg)
        self.original_exception = original_exception


def dynamic_range(image: np.ndarray, ex_lo_frac: float = 0.05,
                  ex_hi_frac: float = 0.05, mask: (None or np.ndarray) = None):
    """
    Compute the dynamic range in a 8 bit grayscale image

    Range is computed between the lowest value of the upper
    procentage of the image defined by ex_hi_frac and highest
    value of the lower procentage of the image defined by
    ex_lo_frac

    Parameters
    ----------
    image : np.ndarray
        8 bit grayscale image to process
    ex_lo_frac : float (default 0.05)
        defines the lower prozentage of the image
        which is considered
    ex_hi_frac : float (default 0.05)
        defines the upper prozentage of the image
        which is considered
    mask : None or np.ndarray (default None)
        if defined only pixel of mask are considered
        for the dynamic range computation
    """
    hist = cv2.calcHist(images=[image],
                        channels=[0],
                        mask=mask,
                        histSize=[256],
                        ranges=[0, 256])

    cumsum = hist.squeeze().cumsum()
    total = cumsum[-1]

    lo_count = ex_lo_frac * total
    lo = next(i for i, value in enumerate(cumsum) if value > lo_count)

    hi_count = ex_hi_frac * total
    hi = next(i for i in range(len(cumsum)-1, -1, -1)
              if (total-cumsum[i]) > hi_count)

    return hi-lo, lo, hi


def check_presents_of_fabric(image: np.ndarray, roi: (None, Iterable[int]),
                             min_dr: int = 45):
    """
    Check dynamic range in a specific Region of Interest
    TODO: Rename this to something unspecific to fabric.

    Parameters
    ----------
    image : np.ndarray
        8 bit grayscale image to process
    roi : None or Iterable[int]
        defines a rectangular region of interset by
        four index values [left, top, right, bottom]
        if not defined complete image is the roi
    min_dr : int (default 45)
        minimal dynamic range of image in 8 bit
        grayscale image [0, 255]
    """

    if not roi:
        roi = [0, 0, image.shape[1], image.shape[0]]

    roi_image = image[roi[1]:roi[3], roi[0]:roi[2]]

    dr, _, _ = dynamic_range(roi_image, 0.05, 0.05)

    if dr < min_dr:
        err = 'dynamic range of input image is not sufficient' \
            ' (actual {}, required {})'.format(dr, min_dr)
        raise DynamicRangeException(err)


def find_contours(image: np.ndarray, roi: (None, Iterable[int]),
                  min_dr: int = 45):
    """
    Find contours in a 8bit grayscale image

    Parameters
    ----------
    image : np.ndarray
        8 bit grayscale image to process
    roi : None or Iterable[int]
        defines a rectangular region of interset by
        four index values [left, top, right, bottom]
        if not defined complete image is the roi
    min_dr : int (default 45)
        minimal dynamic range of image in 8 bit
        grayscale image [0, 255]
    """
    if not roi:
        roi = [0, 0, image.shape[1], image.shape[0]]

    check_presents_of_fabric(image, roi, min_dr)

    roi_image = image[roi[1]:roi[3], roi[0]:roi[2]]

    blur = cv2.GaussianBlur(src=roi_image, ksize=(7, 7), sigmaX=0)

    ret, threshold = cv2.threshold(src=blur, thresh=0, maxval=255,
                                   type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    _, contours, hierarchy = cv2.findContours(image=threshold,
                                              mode=cv2.RETR_EXTERNAL,
                                              method=cv2.CHAIN_APPROX_SIMPLE,
                                              offset=(roi[0], roi[1]))

    return contours, hierarchy


def largest_contour(contours: List):
    """
    Find the largest contour in a list of contours

    Parameters
    ----------
    Contours : List
        list of contours
    """

    max_area = 0
    largest = None

    for c in contours:
        area = cv2.contourArea(contour=c)
        if area > max_area:
            max_area = area
            largest = c

    return largest, max_area


def project_pixel_on_plane(pixel_point: np.ndarray, plane_params: np.ndarray,
                           camera_pose: Pose, camera_matrix: np.ndarray):
    """
    Find pose of specifc image pixel

    Assumption is that the image content lies on a plane

    Parameters
    ----------
    pixel_point : np.ndarray
        pixel point [x, y]
    plane_params : np.ndarray
        plane in general form ax + by + cz + d = 0
        as numpy array [a,b,c,d]
    camera_pose : xamla_motion.data_types.Pose
        pose of camera in world coordinates
    camera_matrix : np.ndarray
        3x3 camera matrix see opencv for more information
    """
    cam_pose = camera_pose.transformation_matrix()

    # create ray in camera space
    htp = np.array([pixel_point[0], pixel_point[1], 1])
    ray = np.matmul(np.linalg.inv(camera_matrix), htp)

    # transform form camera to world axis
    ray_world = np.matmul(cam_pose[0:3, 0:3], ray)
    eye = cam_pose[0:3, 3]
    at = eye + ray_world

    # compute intersection ray and plane
    t, hit = x3d.intersect_ray_plane(eye, at, plane_params)

    return hit


def compute_otsu_threshold_with_mask(image: np.ndarray, mask: np.ndarray):
    """
    Computes otsu threshold value with mask because opencv not provide it

    Parameters
    ----------
    image : np.ndarray
        unit8 grayscale image
    mask : np.ndarray
        uint8 mask zeros values are
        ignored 255 are in mask

    Returns
    -------
    otsu_treshold : int
        determine threshold value
    """

    hist = cv2.calcHist(images=[image],
                        channels=[0],
                        mask=mask,
                        histSize=[256],
                        ranges=[0, 256])

    hist_norm = hist.ravel()/hist.max()
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1
    for i in range(1, 256):
        p1, p2 = np.hsplit(hist_norm, [i])  # probabilities
        q1, q2 = Q[i], Q[255]-Q[i]  # cum sum of classes
        if q1 == 0:
            q1 = 1e-4
        if q2 == 0:
            q2 = 1e-4
        b1, b2 = np.hsplit(bins, [i])  # weights
        # finding means and variances
        m1, m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1, v2 = np.sum(((b1-m1)**2)*p1)/q1, np.sum(((b2-m2)**2)*p2)/q2
        # calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i

    return thresh
