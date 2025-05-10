import cv2
import glob
import numpy as np
import os
from tqdm import tqdm

def matching_two_image(image1_path, image2_path, threshold_knn=0.75):
    """
    TODO:
    Step1: Accepts two image file paths and performs SIFT-based matching between them.
    It detects keypoints and computes descriptors using the SIFT algorithm,
    then matches descriptors using BFMatcher with k-NN matching.
    Finally, it applies Lowe's ratio test to filter out unreliable matches.
    
    Allow functions:
        cv2.cvtColor()
        cv2.SIFT_create()
        cv2.SIFT_create().*
        cv2.BFMatcher()
        cv2.BFMatcher().*
        cv2.drawMatchesKnn()
        
    Parameters:
        image1_path (str): File path for the first image.
        image2_path (str): File path for the second image.
        threshold_knn (float): Lowe's ratio test threshold (default is 0.75).
        
    Output:
        img1, img2 (numpy.ndarray): The original images.
        kp1, kp2 (list[cv2.KeyPoint]): Lists of keypoints detected in each image.
        des1, des2 (numpy.ndarray): SIFT descriptors for each image.
        matches (list[cv2.DMatch]): The matching results after applying Lowe's ratio test.
    """
    #TODO: Fill this functions

    # load images 
    img1 = cv2.imread(image1_path, cv2.IMREAD_COLOR)
    img2 = cv2.imread(image2_path, cv2.IMREAD_COLOR)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # feature extract
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # feature matching
    bfmatcher = cv2.BFMatcher_create(normType=cv2.NORM_L2, crossCheck=False)
    matches_candidate = bfmatcher.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    matches = []
    for m1, m2 in matches_candidate:
        if (m1.distance / m2.distance) < threshold_knn:
            matches.append(m1)

    ## test
    # matches_for_draw = [[m] for m in matches]
    # vis_result = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches_for_draw, None, \
    #                                 flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow("matches", vis_result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return img1, img2, kp1, kp2, des1, des2, matches
