import numpy as np
try:
    import open3d as o3d
except ModuleNotFoundError as e:
    print(e, 'install it before proceeding', sep=', ')
else:
	o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel(0))
from scipy import spatial


__all__ = ['estimate_normals', 'orient_normals', 'orient_normals_cvx']


def _weight(p, nbhd, kernel='linear', gamma=None):
    """Return scaling weights given distance between a targeted point
    and its surrounding local neighborhood.
    
    Parameters
    ----------
    p : numpy_ndarray
        Targeted point of shape (3, )
    nbhd : numpy.ndarray
        An array of shape (N, 3) representing the local neighborhood
    kernel : str, optional
        The weighting function. If not given, weights are set to unity
    gamma : float, optional
        A scaling factor for the weighting function. If not given, it
        is set to 1
        
    Returns
    -------
    numpy.ndarray
        Array with weights of (N, )
    """
    dist = np.linalg.norm(nbhd - p, axis=1)  # squared Euclidean distance
    if gamma is None:
        gamma = 1.
    if kernel == 'linear':
        w = np.maximum(1 - gamma * dist, 0)
    elif kernel == 'truncated':
        w = np.maximum(1 - gamma * dist ** 2, 0)
    elif kernel == 'inverse':
        w = 1 / (dist + 1e12) ** gamma
    elif kernel == 'gaussian':
        w = np.exp(-(gamma * dist) ** 2)
    elif kernel == 'multiquadric':
        w = np.sqrt(1 + (gamma * dist) ** 2)
    elif kernel == 'inverse_quadric':
        w = 1 / (1 + (gamma * dist) ** 2)
    elif kernel == 'inverse_multiquadric':
        w = 1 / np.sqrt(1 + (gamma * dist) ** 2 )
    elif kernel == 'thin_plate_spline':
        w = dist ** 2 * np.log(dist)
    elif kernel == 'rbf':
        w = np.exp(-dist ** 2 / (2 * gamma ** 2))
    elif kernel == 'cosine':
        w = (nbhd @ p) / np.linalg.norm(nbhd * p, axis=1)
    return w


def estimate_normals(xyz, k, **kwargs):
    """Return the unit normals by fitting local tangent plane at each
    point in the point cloud.
    
    Ref: Hoppe et al., in proceedings of SIGGRAPH 1992, pp. 71-78,
         doi: 10.1145/133994.134011
    
    Parameters
    ----------
    xyz : numpy.ndarray
        The point cloud of shape (N, 3), N is the number of points
    k : float
        The number of nearest neighbors of a local neighborhood around
        a current query point
    kwargs : dict, optional
        Additional keyword arguments for computing weights. For details
        see `_weight` function
    
    Returns
    -------
    numpy.ndarray
        The unit normals of shape (N, 3), where N is the number of
        points in the point cloud
    """
    # create a kd-tree for quick nearest-neighbor lookup
    tree = spatial.KDTree(xyz)
    n = np.empty_like(xyz)
    for i, p in enumerate(xyz):
        # extract the local neighborhood
        _, idx = tree.query([p], k=k, eps=0.1, workers=-1)
        nbhd = xyz[idx.flatten()]
        
        # compute the kernel function and create the weights matrix
        if 'kernel' in kwargs:
            w = _weight(p, nbhd, **kwargs)
        else:
            w = np.ones((nbhd.shape[0], ))
        W = np.diag(w)
        
        # extract an eigenvector with smallest associeted eigenvalue
        X = nbhd.copy()
        X = X - np.average(X, weights=w, axis=0)
        C = (X.T @ (W @ X)) / np.sum(w)
        U, S, VT = np.linalg.svd(C)
        n[i, :] =  U[:, np.argmin(S)]
    return n


def orient_normals(points, normals, k):
    """Orient the normals with respect to consistent tangent planes.
    
    Ref: Hoppe et al., in proceedings of SIGGRAPH 1992, pp. 71-78,
         doi: 10.1145/133994.134011
    
    Parameters
    ----------
    points : numpy.ndarray
        The point cloud of shape (N, 3), N is the number of points
    normals : numpy.ndarray
        Normals of shape (N, 3), where N is the number of points in the
        point cloud
    k : int
        Number of k nearest neighbors used in constructing the
        Riemannian graph used to propagate normal orientation
    
    Returns
    -------
    numpy.ndarray
        Oriented normals.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.orient_normals_consistent_tangent_plane(k)
    return np.asarray(pcd.normals)


def orient_normals_cvx(xyz, n):
    """Orient the normals in the outward direction.

    Note: Only for convex-only shapes.
    
    Parameters
    ----------
    xyz : numpy.ndarray
        The point cloud of shape (N, 3), N is the number of points.
    n : numpy.ndarray
        The unit normals of shape (N, 3), where N is the number of
        points in the point cloud.
    
    Returns
    -------
    numpy.ndarray
        Oriented normals.
    """
    center = np.mean(xyz, axis=0)
    for i in range(xyz.shape[0]):
        pi = xyz[i, :] - center
        ni = n[i]
        angle = np.arccos(np.clip(np.dot(ni, pi), -1.0, 1.0))
        if (angle > np.pi/2) or (angle < -np.pi/2):
            n[i] = -ni
    return n
