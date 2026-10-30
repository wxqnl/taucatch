"""First-order covariance at a fixed-x crossing, from a causal visual forecast.

Returns uncertainty in (arrival time, crossing y, crossing z). Along-path depth
uncertainty mainly changes arrival time; velocity/cross-covariances still enter
both transverse errors. This is a local linearization, not calibration evidence.
"""
import numpy as np


def crossing_covariance(covariance, horizon, velocity):
    covariance=np.asarray(covariance,float);velocity=np.asarray(velocity,float)
    if covariance.shape!=(6,6) or velocity.shape!=(3,) or not np.isfinite(covariance).all() or not np.isfinite(velocity).all():
        raise ValueError('Invalid crossing covariance inputs')
    if horizon<0 or not np.isfinite(horizon) or abs(velocity[0])<1e-6:
        raise ValueError('Invalid crossing horizon/direction')
    if np.linalg.eigvalsh((covariance+covariance.T)/2).min() < -1e-8:
        raise ValueError('Non-positive-semidefinite state covariance')
    base=np.column_stack((np.eye(3),horizon*np.eye(3)))
    time_gradient=-base[0]/velocity[0]
    jacobian=np.stack((time_gradient,base[1]+velocity[1]*time_gradient,
        base[2]+velocity[2]*time_gradient))
    projected=jacobian@covariance@jacobian.T
    return dict(covariance=projected,jacobian=jacobian,
        sigma_time_s=float(np.sqrt(max(0,projected[0,0]))),
        sigma_plane_m=float(np.sqrt(max(0,(projected[1,1]+projected[2,2])/2))))
