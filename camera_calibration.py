import cv2
import numpy as np
import os
from tqdm import tqdm

def camera_calibaration(images_files, checker_files, checkerboard):
    """
        Step7: Accepts a list of image filenames, a directory path containing the chessboard images, 
        and a tuple defining the chessboard dimensions (number of inner corners per row and column).
        The function performs camera calibration by:
            - Creating the 3D object points for the chessboard pattern.
            - Iterating through each image to convert it to grayscale.
            - Detecting the chessboard corners using cv2.findChessboardCorners.
            - Refining corner positions with cv2.cornerSubPix.
            - Collecting corresponding object points and image points.
            - Computing the camera intrinsic matrix using cv2.calibrateCamera.

        Allow functions:
            numpy
            cv2.imread()
            cv2.cvtColor()
            cv2.findChessboardCorners()
            cv2.cornerSubPix()
            cv2.calibrateCamera()
            tqdm (for progress tracking)

        Parameters:
            images_files (list[str]): List of calibration image filenames.
            checker_files (str): Directory path where the chessboard images are stored.
            checkerboard (tuple[int, int]): Dimensions of the chessboard (number of inner corners per row and column).

        Output:
            camera_matrix (numpy.ndarray): The intrinsic camera matrix computed from the calibration process (3 * 3).
    """
    #TODO: Fill this functions
    # prepare object points : (0,0,0), (1,0,0), ..., (cols-1,rows-1,0)
    objp = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
    square_size = 30 # mm size of checkerboard square 
    objp[0,:,:2] = square_size * np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)

    objpoints = []  # 3D points (real world)    
    imgpoints = []  # 2D points (image)
    for image_file in tqdm(images_files, desc="Camera calibration"):
        # load checkerboard image
        img_path = os.path.join(checker_files, image_file)
        img = cv2.imread(img_path)
        if img is None:
            continue 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard, \
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret:
            # refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_subpix = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners_subpix)

            # check 
    #         img = cv2.drawChessboardCorners(img, checkerboard, corners_subpix, ret)
    #         cv2.imshow('img',img)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # calibrate camera
    ret, camera_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    return camera_matrix