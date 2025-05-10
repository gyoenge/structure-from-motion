import numpy as np

def triangulate_points(EM0, EM1, inlier_p1, inlier_p2, camera_intrinsic):
    """
    Step4: Computes 3D points via linear triangulation using the given camera poses (EM0, EM1)
    and the corresponding inlier keypoint coordinates from two images.
    
    Allow functions:
        numpy
        
    Deny functions:
        cv2

    Parameters:
        EM0             : Pose of the first camera ([I|0], 3x4 numpy array).
        EM1             : Pose of the second camera (3x4 numpy array).
        inlier_p1       : Inlier keypoints from the first image (N x 2 numpy array, [x, y]).
        inlier_p2       : Inlier keypoints from the second image (N x 2 numpy array, [x, y]).
        camera_intrinsic: Camera intrinsic matrix (3x3 numpy array).
        
    Returns:
        points_3d (np.ndarray): (N x 3) numpy array where each row is the triangulated 3D coordinate (X, Y, Z).
        inlier_idx (np.ndarray): (N,) numpy array containing the indices of the inlier points used.
    """
    #TODO: Fill this functions

    # (1) normalize with carmera intrinsic 
    # inlier_p1 = (np.linalg.inv(camera_intrinsic) @ np.hstack([inlier_p1, np.ones((len(inlier_p1),1))]).T).T
    # inlier_p2 = (np.linalg.inv(camera_intrinsic) @ np.hstack([inlier_p2, np.ones((len(inlier_p2),1))]).T).T
    # (2) convert to extrinsic matrix 
    EM0 = camera_intrinsic @ EM0
    EM1 = camera_intrinsic @ EM1

    # triangulation 
    points_3d = []
    for i, (pt1, pt2) in enumerate(zip(inlier_p1, inlier_p2)):
        # solve SVD 
        A = np.stack([
            pt1[0] * EM0[2] - EM0[0],
            pt1[1] * EM0[2] - EM0[1],
            pt2[0] * EM1[2] - EM1[0],
            pt2[1] * EM1[2] - EM1[1]
        ])
        _, _, VT = np.linalg.svd(A)
        X = VT[-1] 

        # if np.abs(X[3]) > 1e-8:
        #     X = X / X[3]

        # normalize (homogeneous)
        X = X / X[3]
        
        points_3d.append(X[:3])
    points_3d = np.stack(points_3d)
    inlier_idx = np.arange(len(inlier_p1))

    ## test 
    # print(np.min(points_3d[:,0]), np.max(points_3d[:,0]))
    # print(np.min(points_3d[:,1]), np.max(points_3d[:,1]))
    # print(np.min(points_3d[:,2]), np.max(points_3d[:,2]))

    return points_3d, inlier_idx
