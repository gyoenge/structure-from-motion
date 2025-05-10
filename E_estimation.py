import numpy as np
from tqdm import tqdm
import matlab.engine

def essential_matrix_estimation(kp1, kp2, matches, camera_intrinsic, eng, 
                                max_iter=5000, threshold=1e-5):
    """
    Step2: Estimates the Essential Matrix using the 5-Point Algorithm with RANSAC.
    It takes the camera intrinsic matrix, keypoints from two images, their matches,
    and a MATLAB engine instance (which must already be started). The function uses
    a RANSAC loop to find the best Essential Matrix candidate that fits the normalized
    matched keypoints.

    Allow functions:
        numpy
        tqdm (for progress tracking)
        eng.calibrated_fivepoint() (please read ./Step2/calibrated_fivepoint.m)

    Deny functions:
        cv2

    Parameters:
        kp1 (list): List of cv2.KeyPoint objects from the first image.
        kp2 (list): List of cv2.KeyPoint objects from the second image.
        matches (list): List of cv2.DMatch objects representing the matches between the images.
        camera_intrinsic (np.ndarray): Camera intrinsic matrix (3x3).
        eng: MATLAB engine object (already started).
        max_iter (int): Maximum number of RANSAC iterations (default 5000).
        threshold (float): Inlier threshold for error (default 1e-5).

    Returns:
        E_est (np.ndarray): The estimated Essential Matrix (3x3).
        inlier_p1 (np.ndarray): Inlier keypoint coordinates from the first image (N x 2).
        inlier_p2 (np.ndarray): Inlier keypoint coordinates from the second image (N x 2).
        best_inlier_idx (np.ndarray): Inlier matching index (N, )
    """
    # TODO: Fill this function

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
    
    # RANSAC 
    best_inlier_count = 0
    for i in tqdm(range(max_iter), desc="Essential matrix estimation"):
        # sample 5 points 
        sampled_idx = np.random.choice(kp1_matched.shape[0], size=5, replace=False)
        kp1_sampled = kp1_matched[sampled_idx].T
        kp2_sampled = kp2_matched[sampled_idx].T

        # solve 5-point algorithm 
        evec = eng.calibrated_fivepoint(
            matlab.double(kp1_sampled.tolist()), 
            matlab.double(kp2_sampled.tolist())
        )

        if len(evec[0]) == 0:
            continue
        
        # RANSAC evaluation 
        Es = [np.array(evec)[:, i].reshape(3, 3) for i in range(len(evec[0]))]
        for E in Es:
            # calculate error  
            errors = np.diagonal(kp2_matched @ E @ kp1_matched.T)

            # find inliers 
            inliers = (errors < threshold) & (errors > 0)
            inlier_count = np.sum(inliers)

            # update best estimation 
            if inlier_count > best_inlier_count:
                E_est = E
                best_inlier_idx = np.where(inliers)[0]
                best_inlier_count = inlier_count

        # early stop
        if best_inlier_count > len(matches) * 0.95 :
            break
    
    if i+1 < max_iter: 
        print(f"early stopped with {i+1} iterations...")
    
    # save inlier points 
    inlier_matches = matches[best_inlier_idx]
    inlier_p1 = kp1[inlier_matches[:,0].astype(int)]
    inlier_p2 = kp2[inlier_matches[:,1].astype(int)]

    return E_est, inlier_p1, inlier_p2, best_inlier_idx
