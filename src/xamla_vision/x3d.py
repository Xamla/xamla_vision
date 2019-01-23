import numpy as np
from xamla_motion.data_types import Pose


def normalize(v: np.ndarray):
    return v/np.linalg.norm(v)


def intersect_ray_plane(eye: np.ndarray, at: np.ndarray,
                        plan_params: np.ndarray):

    n = plan_params[0:3]

    p0 = n * -plan_params[3]
    l = at - eye
    d = np.matmul(n, l)
    if np.abs(d) < 1e-6:
        raise RuntimeError('plane and ray must no be parallel')

    t = np.matmul((p0 - eye), n) / d
    return t, eye + l*t


def plane_parameters_from_pose(pose: Pose):
    """
    Compute plane parameters in general form ax + by + cz + d = 0

    Parameters
    ----------
    pose : xamla_motion.data_types.Pose
        pose which plane in extracted z is normal axis
    """

    pm = pose.transformation_matrix()

    z = normalize(pm[0:3, 2])
    d = np.matmul(z, pm[0:3, 3])
    return np.asarray([z[0], z[1], z[2], -d])
