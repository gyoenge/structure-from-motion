import numpy as np

def essential_matrix_decomposition(E, inlier_p1, inlier_p2, camera_intrinsic):
    """
    Step3: Decomposes the Essential Matrix and performs triangulation to compute the poses of the two cameras.
    The function returns the pose of the first camera (P0, which is [I | 0]) and the selected pose for
    the second camera (P1) based on the cheirality condition.

    Allow functions:
        numpy
        
    Deny functions:
        cv2

    Parameters:
        E (np.ndarray): Essential Matrix (3x3 numpy array).
        inlier_p1 (np.ndarray): Inlier keypoint coordinates from the first image (N x 2 numpy array).
        inlier_p2 (np.ndarray): Inlier keypoint coordinates from the second image (N x 2 numpy array).
        camera_intrinsic (np.ndarray): Camera intrinsic matrix (3x3 numpy array).

    Returns:
        P0 (np.ndarray): Pose of the first camera ([I | 0], 3x4 numpy array).
        P1 (np.ndarray): Pose of the selected second camera (3x4 numpy array).
    """
    #TODO: Fill this functions

    # P0 = [I | 0]
    P0 = np.hstack([np.eye(3), np.zeros([3,1])]) 

    # Essential matrix decomposition
    U, _, VT = np.linalg.svd(E)
    # U = U / np.linalg.norm(U, axis=0, keepdims=True)
    # VT = VT / np.linalg.norm(VT, axis=1, keepdims=True)
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])

    # Enforce det(R)=+1
    if np.linalg.det(U @ VT) < 0:
        VT = -VT
    # if np.linalg.det(U) < 0:
    #     U *= -1
    # if np.linalg.det(VT) < 0:
    #     VT *= -1

    # 4 candidates of P1 
    UWVT = U @ W @ VT
    UWTVT = U @ W.T @ VT
    u = (U @ np.array([0,0,1]).T).reshape(3, 1)
    P1s = [
        np.hstack([UWVT, u]),
        np.hstack([UWVT, -u]),
        np.hstack([UWTVT, u]),
        np.hstack([UWTVT, -u])
    ]

    # Select P1 
    inlier_p1 = (np.linalg.inv(camera_intrinsic) @ np.hstack([inlier_p1, np.ones((len(inlier_p1),1))]).T).T
    inlier_p2 = (np.linalg.inv(camera_intrinsic) @ np.hstack([inlier_p2, np.ones((len(inlier_p2),1))]).T).T
    best_poszcount = 0 
    for _P1 in P1s: 
        # check rotation matrix determinant 
        # print(f"R det: {np.linalg.det(_P1[:,:3])}")
        if np.linalg.det(_P1[:,:3]) < 0 :
            continue

        # compute triangulation
        X_list = []
        for x0, x1 in zip(inlier_p1, inlier_p2):
            A = np.stack([
                x0[0] * P0[2] - P0[0],
                x0[1] * P0[2] - P0[1],
                x1[0] * _P1[2] - _P1[0],
                x1[1] * _P1[2] - _P1[1]
            ])
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1]
            X_list.append(X)
        X_all = np.stack(X_list)  # (N, 4)

        # check positive z count

        # (1)
        # X_all = X_all.reshape(-1, 4, 1)  # (N, 4, 1)
        # PX_all = _P1 @ X_all # (N, 3, 1)
        # PX_all = PX_all.reshape(-1, 3) # (N, 3)
        # w_all = PX_all[:,2] 
        # depths = (np.sign(np.linalg.det(_P1[:,:3])) * w_all) / \
        #         (np.linalg.norm(_P1[:,3]) * np.linalg.norm(_P1[:,:3][2]))
        
        # 
        # poszcount = np.sum((depths > 0) & (X_all[:,2].squeeze() > 0))  # 각각 (N, )

        # (2) 
        X_all /= X_all[:, 3].reshape(-1, 1)  # (N, 4), normalize
        z1 = X_all[:, 2]  # Z from camera1 (P0)
        poszcount = np.sum(z1 > 0)

        if poszcount > best_poszcount:
            best_poszcount = poszcount
            P1 = _P1

        ## check 
        # print(f"inlier len : {len(inlier_p1)}")
        # print(f"depths.shape : {depths.shape}")
        # print(f"X_all[:,2].squeeze().shape : {X_all[:,2].squeeze().shape}")
        # print(f"_P1 : {_P1}")
        # print(f"poszcount : {poszcount}")
    
    return P0, P1
