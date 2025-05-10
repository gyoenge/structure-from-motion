import numpy as np
from tqdm import tqdm
import random
import matlab.engine

def three_point_algorithm(matches, next_matches, inlier_idx, initial_point, mid_image_kp, add_image_kp, 
                          camera_intrinsic, eng, threepoint_threshold=1e-4, threepoint_max_iter=1000,):
    """
    Estimate the projection matrix of a third image via P3P and RANSAC.

    Parameters:
        matches (list[cv2.DMatch]): Matches between image1 and image2.
        next_matches (list[cv2.DMatch]): Matches between image2 and image3.
        inlier_idx (np.ndarray): Indices of inliers from calculate_inlier_points.
        initial_point (np.ndarray): 3D points from stereo reconstruction (Nx3).
        ## added ##
        mid_image_kp (list[cv2.KeyPoint]): Keypoints from image2.
        ###########
        add_image_kp (list[cv2.KeyPoint]): Keypoints from image3.
        camera_intrinsic (np.ndarray): Intrinsic camera matrix (3x3).
        eng (matlab.engine): MATLAB engine instance.
        threepoint_threshold (float): Reprojection error threshold.
        threepoint_max_iter (int): Maximum RANSAC iterations.

    Returns:
        best_P (np.ndarray): Best estimated projection matrix (3x4).
    """
    # TODO: Fill this function
    
    # match 3 images 
    matched_3D = []
    matched_kp3 = [] 
    for idx_3D, idx_first_matches in enumerate(inlier_idx): 
        idx_kp2 = matches[idx_first_matches].trainIdx
        idx_kp3 = None 
        # min_dist = 3.0
        for next_match in next_matches:
            # (1) point position based
            # pt_prev = mid_image_kp[idx_kp2].pt
            # pt_next = mid_image_kp[next_match.queryIdx].pt
            # dist = np.linalg.norm(np.array(pt_next) - np.array(pt_prev))
            # if dist < min_dist:
            #     # print(f"{np.array(pt_next).astype(int)}, {np.array(pt_prev).astype(int)}, {dist}")
            #     idx_kp3 = next_match.trainIdx
            #     min_dist = dist 
            # (2) point index based 
            if mid_image_kp[idx_kp2] == mid_image_kp[next_match.queryIdx]:
                # print(mid_image_kp[idx_kp2].pt, mid_image_kp[next_match.queryIdx].pt)
                idx_kp3 = next_match.trainIdx
        if idx_kp3 is None: 
            continue
        matched_3D.append(int(idx_3D))
        matched_kp3.append(idx_kp3) 
    # matched points 
    points2D = np.array([add_image_kp[idx].pt for idx in matched_kp3], dtype=np.float64)  # (N, 2)
    points3D = initial_point[np.array(matched_3D)]  # (N, 3)

    if len(matched_kp3) < 3: 
        raise ValueError(f"Cannot run three point algorithm.. (matched count: {len(matched_kp3)})")
    
    ## opencv test
    # import cv2 
    # success, rvec, tvec = cv2.solvePnP(points3D, points2D, camera_intrinsic, None)
    # R, _ = cv2.Rodrigues(rvec)
    # P3 = np.hstack((R, tvec))
    # print("cv2 result : \n", P3) 

    # RANSAC 
    best_P = None
    max_inliers = 0
    for _ in tqdm(range(threepoint_max_iter), desc="Three point algorithm"):
        # sample 3 points
        sampled_idx = random.sample(range(len(points3D)), 3)
        pts3D_sample = points3D[sampled_idx, :]  # (3, 3)
        pts2D_sample = points2D[sampled_idx, :]  # (3, 2)
        pts2D_sample_norm = (np.linalg.inv(camera_intrinsic) @ np.hstack([pts2D_sample, np.ones((3,1))]).T).T[:, :2]

        # sovle P3P algorithm 
        input_matrix = np.hstack([pts2D_sample_norm, np.ones((3,1)), pts3D_sample])
        poses = eng.PerspectiveThreePoint(
            matlab.double(input_matrix.tolist())
        )
        poses = np.array(poses)
        
        if poses.ndim == 0 or poses.shape[0] == 0:
            continue
    
        # RANSAC evaluation 
        n_poses = poses.shape[0]//4
        poses = [poses[4*i:4*(i+1)] for i in range(n_poses)]
        # print(f"pose candidate number : {n_poses}")
        for pose in poses:            
            # check R determinant  
            if np.linalg.det(pose[:3, :3]) < 0:
                pose[:3, :3] = -pose[:3, :3]
            
            # print(pose[:3,:])

            # calculate reprojection error
            projected = (camera_intrinsic @ pose[:3,:] @ np.hstack((points3D, np.ones((len(points3D), 1)))).T).T # (N, 3)
            # print(f"z val negative : {np.where(projected[:,2]<0)[0].shape[0]}")
            projected = projected[:,:2] / projected[:,2:]
            errors = np.linalg.norm(projected - points2D, axis=1) / np.linalg.norm(points2D, axis=1)
            # print(f"projected[0]: {projected[0]}, points2D_normalized[0]: {points2D[0]}, \nerrors[0]: {errors[0]}")

            # find inliers 
            inliers = errors < threepoint_threshold
            inlier_count = np.sum(inliers)

            # update best estimation 
            if inlier_count > max_inliers:
                max_inliers = inlier_count
                best_P = pose[:3,:]

                ## test
                # print("## updated : ")
                # print(f"max_liniers : {max_inliers}")
                # print(f"best_P : {best_P}")
                # print(f"{np.where(inliers)[0].shape}")
                # print(f"{np.where(inliers)[0][0]}")
                # print(f"{points2D[(np.where(inliers)[0][0])]}")
    
    ## test
    # print(f"poses : {poses}")
    # print(f"pose : {best_P[:3,:]}")
    print(f"three point matched count : {len(matched_kp3)}") 
    print(f"three point inlier count : {max_inliers}") 
    print(f"det(best_P[:,:3]) : {np.linalg.det(best_P[:,:3])}")

    return best_P, matched_3D, matched_kp3 ## added matched_3D, matched_kp3 


def calculate_inlier_points(EM1, EM2, kp1, kp2, matches, camera_intrinsic, threshold=1e-2):
    """
    Identify inlier keypoint pairs between two images given their essential matrices.

    Parameters:
        EM1 (np.ndarray): Projection matrix of image 1 (3x4).
        EM2 (np.ndarray): Projection matrix of image 2 (3x4).
        kp1 (list[cv2.KeyPoint]): Keypoints from image 1.
        kp2 (list[cv2.KeyPoint]): Keypoints from image 2.
        matches (list[cv2.DMatch]): Matches between descriptors of image1 and image2.
        camera_intrinsic (np.ndarray): Intrinsic camera matrix (3x3).
        threshold (float): Epipolar error threshold for inlier selection.

    Returns:
        inlier_p1 (np.ndarray): Array of inlier points from image 1 (Nx2).
        inlier_p2 (np.ndarray): Array of inlier points from image 2 (Nx2).
        inlier_idx (np.ndarray): Indices of inlier matches.
    """
    # TODO: Fill this function
    E = convert_camera_matrix(EM1, EM2)

    # convert into numpy matrices 
    matches = np.array([[m.queryIdx, m.trainIdx, m.distance] for m in matches], dtype=np.float32)
    kp1 = np.array([kp.pt for kp in kp1], dtype=np.float32)
    kp2 = np.array([kp.pt for kp in kp2], dtype=np.float32)
    # normalize with carmera intrinsic 
    kp1_normalized = (np.linalg.inv(camera_intrinsic) @ np.hstack([kp1, np.ones((len(kp1),1))]).T).T
    kp2_normalized = (np.linalg.inv(camera_intrinsic) @ np.hstack([kp2, np.ones((len(kp2),1))]).T).T
    # matched all 
    kp1_matched = kp1_normalized[matches[:,0].astype(int)]
    kp2_matched = kp2_normalized[matches[:,1].astype(int)]

    # calculate error  
    errors = np.diagonal(kp2_matched @ E @ kp1_matched.T)
    ## check 
    # print(f"np.max(errors): {np.max(errors)}")
    # if len(np.where(errors>0)[0])!=0: 
    #     print(f"np.min(errors): {np.min(errors[errors>0])}")

    # find inliers 
    inliers = (errors < threshold) & (errors > 0)
    inlier_idx = np.where(inliers)[0]
    inlier_matches = matches[inlier_idx]
    inlier_p1 = kp1[inlier_matches[:,0].astype(int)]
    inlier_p2 = kp2[inlier_matches[:,1].astype(int)]

    print(f"growing step inlier count : {len(inlier_idx)}")
    
    return inlier_p1, inlier_p2, inlier_idx


def convert_camera_matrix(Rt0, Rt1):
    """
    Convert two camera projection matrices to the essential matrix E.

    Parameters:
        Rt0 (np.ndarray): Projection matrix of camera 0 (3x4).
        Rt1 (np.ndarray): Projection matrix of camera 1 (3x4).

    Returns:
        E (np.ndarray): Essential matrix (3x3).
    """
    R0, t0 = Rt0[:, :3], Rt0[:, 3]
    R1, t1 = Rt1[:, :3], Rt1[:, 3]
    R = R1 @ R0.T
    t = t1 - R @ t0
    t_cross = np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])
    E = t_cross @ R
    return E