import cv2
import numpy as np


def draw_matches(img1, img2, kp1, kp2, matches, inlier_idx):
    matches_inlier = [matches[i] for i in inlier_idx.astype(int)]
    matches_for_draw = [[m] for m in matches_inlier]
    matching_result_img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches_for_draw, None, flags=2)
    matching_result_img = cv2.resize(matching_result_img, (3200, 1200), interpolation=cv2.INTER_AREA)
    return matching_result_img

def draw_keypoints(img, points, colors, radius=15):
    img_copy = img.copy()
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
    for pt, color in zip(points, colors):
        pt = tuple(map(int, pt))
        cv2.circle(img_copy, pt, radius, color, -1)
    return img_copy

def visualize_three_image_matches(img1, img2, img3,
                            keypoints1, keypoints2, keypoints3,
                            matches, next_matches,
                            inlier_idx, matched_3D, matched_kp3):
    h = max(img1.shape[0], img2.shape[0], img3.shape[0])
    w1, w2, w3 = img1.shape[1], img2.shape[1], img3.shape[1]
    vis_image = np.zeros((h, w1 + w2 + w3, 3), dtype=np.uint8)

    vis_image[:img1.shape[0], :w1] = img1
    vis_image[:img2.shape[0], w1:w1 + w2] = img2
    vis_image[:img3.shape[0], w1 + w2:] = img3

    for i, idx3 in enumerate(matched_kp3):
        idx2 = matches[inlier_idx[matched_3D[i]]].trainIdx
        idx1 = matches[inlier_idx[matched_3D[i]]].queryIdx

        pt1 = tuple(map(int, keypoints1[idx1].pt))
        pt2 = tuple(map(int, keypoints2[idx2].pt))
        pt3 = tuple(map(int, keypoints3[idx3].pt))

        pt1 = (pt1[0], pt1[1])
        pt2 = (pt2[0] + w1, pt2[1])
        pt3 = (pt3[0] + w1 + w2, pt3[1])

        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.circle(vis_image, pt1, 15, color, -1)
        cv2.circle(vis_image, pt2, 15, color, -1)
        cv2.circle(vis_image, pt3, 15, color, -1)
        cv2.line(vis_image, pt1, pt2, color, 1)
        cv2.line(vis_image, pt2, pt3, color, 1)

    return vis_image


def visualize_three_image_matches_withkpnp(img1, img2, img3,
                            keypoint1, keypoint2, keypoint3):
    h = max(img1.shape[0], img2.shape[0], img3.shape[0])
    w1, w2, w3 = img1.shape[1], img2.shape[1], img3.shape[1]
    vis_image = np.zeros((h, w1 + w2 + w3, 3), dtype=np.uint8)

    vis_image[:img1.shape[0], :w1] = img1
    vis_image[:img2.shape[0], w1:w1 + w2] = img2
    vis_image[:img3.shape[0], w1 + w2:] = img3

    pt1 = tuple(map(int, keypoint1))
    pt2 = tuple(map(int, keypoint2))
    pt3 = tuple(map(int, keypoint3))

    pt1 = (pt1[0], pt1[1])
    pt2 = (pt2[0] + w1, pt2[1])
    pt3 = (pt3[0] + w1 + w2, pt3[1])

    color = tuple(np.random.randint(0, 255, 3).tolist())
    cv2.circle(vis_image, pt1, 15, color, -1)
    cv2.circle(vis_image, pt2, 15, color, -1)
    cv2.circle(vis_image, pt3, 15, color, -1)
    cv2.line(vis_image, pt1, pt2, color, 1)
    cv2.line(vis_image, pt2, pt3, color, 1)

    return vis_image