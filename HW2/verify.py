import cv2
import numpy as np
import random
import math
import sys

# read the image file & output the color & gray image


def read_img(path):
    # opencv read image in BGR color space
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray

# the dtype of img must be "uint8" to avoid the error of SIFT detector


def img_to_gray(img):
    if img.dtype != "uint8":
        print("The input image dtype is not uint8 , image type is : ", img.dtype)
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

# create a window to show the image
# It will show all the windows after you call im_show()
# Remember to call im_show() in the end of main


def creat_im_window(window_name, img):
    cv2.imshow(window_name, img)

# show the all window you call before im_show()
# and press any key to close all windows


def im_show():
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def SIFT(img_gray):
    print("Computing SIFT...")
    SIFT_Detector = cv2.SIFT_create()
    kp, des = SIFT_Detector.detectAndCompute(img_gray, None)
    return kp, des


def KNN(kps_l, kps_r, features_l, features_r):
    print("Finding good matches...")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    features_l = features_l.astype('uint8')
    features_r = features_r.astype('uint8')
    matches = bf.match(features_l, features_r)

    matches = sorted(matches, key=lambda x: x.distance)

    return matches


def drawMatches(img1, kp1, img2, kp2, matches_pos):

    vis = cv2.drawMatches(img1, kp1, img2, kp2, matches_pos,
                          None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    creat_im_window("img with matching points", vis)
    im_show()
    cv2.imwrite("match_verify.jpg", vis)
    return vis


def find_homography(kp1, kp2, matches_pos):
    dstPoints = []
    srcPoints = []

    for match in matches_pos:
        dstPoints.append(kp1[match.queryIdx].pt)
        srcPoints.append(kp2[match.trainIdx].pt)
    dstPoints = np.array(dstPoints)
    srcPoints = np.array(srcPoints)

    H_verify = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
    H_verify = np.array(H_verify[0])

    return H_verify


def Convert_xy(x, y):
    global center, f

    xt = (f * np.tan((x - center[0]) / f)) + center[0]
    yt = ((y - center[1]) / np.cos((x - center[0]) / f)) + center[1]

    return xt, yt


def ProjectOntoCylinder(InitialImage):
    global w, h, center, f
    h, w = InitialImage.shape[:2]
    center = [w // 2, h // 2]
    f = 1100       # 1100 field; 1000 Sun; 1500 Rainier; 1050 Helens

    # Creating a blank transformed image
    TransformedImage = np.zeros(InitialImage.shape, dtype=np.uint8)

    # Storing all coordinates of the transformed image in 2 arrays (x and y coordinates)
    AllCoordinates_of_ti = np.array(
        [np.array([i, j]) for i in range(w) for j in range(h)])
    ti_x = AllCoordinates_of_ti[:, 0]
    ti_y = AllCoordinates_of_ti[:, 1]

    # Finding corresponding coordinates of the transformed image in the initial image
    ii_x, ii_y = Convert_xy(ti_x, ti_y)

    # Rounding off the coordinate values to get exact pixel values (top-left corner)
    ii_tl_x = ii_x.astype(int)
    ii_tl_y = ii_y.astype(int)

    # Finding transformed image points whose corresponding
    # initial image points lies inside the initial image
    GoodIndices = (ii_tl_x >= 0) * (ii_tl_x <= (w-2)) * \
                  (ii_tl_y >= 0) * (ii_tl_y <= (h-2))

    # Removing all the outside points from everywhere
    ti_x = ti_x[GoodIndices]
    ti_y = ti_y[GoodIndices]

    ii_x = ii_x[GoodIndices]
    ii_y = ii_y[GoodIndices]

    ii_tl_x = ii_tl_x[GoodIndices]
    ii_tl_y = ii_tl_y[GoodIndices]

    # Bilinear interpolation
    dx = ii_x - ii_tl_x
    dy = ii_y - ii_tl_y

    weight_tl = (1.0 - dx) * (1.0 - dy)
    weight_tr = (dx) * (1.0 - dy)
    weight_bl = (1.0 - dx) * (dy)
    weight_br = (dx) * (dy)

    TransformedImage[ti_y, ti_x, :] = (weight_tl[:, None] * InitialImage[ii_tl_y,     ii_tl_x, :]) + \
                                      (weight_tr[:, None] * InitialImage[ii_tl_y,     ii_tl_x + 1, :]) + \
                                      (weight_bl[:, None] * InitialImage[ii_tl_y + 1, ii_tl_x, :]) + \
                                      (weight_br[:, None] *
                                       InitialImage[ii_tl_y + 1, ii_tl_x + 1, :])

    # Getting x coorinate to remove black region from right and left in the transformed image
    min_x = min(ti_x)

    # Cropping out the black region from both sides (using symmetricity)
    TransformedImage = TransformedImage[:, min_x: -min_x, :]

    return TransformedImage, ti_x-min_x, ti_y


def blending(img1, img2):
    (hl, wl) = img1.shape[:2]
    (hr, wr) = img2.shape[:2]
    img_left_mask = np.zeros((hr, wr), dtype="int")
    img_right_mask = np.zeros((hr, wr), dtype="int")
    constant_width = 3  # constant width

    for i in range(hl):
        for j in range(wl):
            if np.count_nonzero(img1[i, j]) > 0:
                img_left_mask[i, j] = 1
    for i in range(hr):
        for j in range(wr):
            if np.count_nonzero(img2[i, j]) > 0:
                img_right_mask[i, j] = 1

    overlap_mask = np.zeros((hr, wr), dtype="int")
    for i in range(hr):
        for j in range(wr):
            if (np.count_nonzero(img_left_mask[i, j]) > 0 and np.count_nonzero(img_right_mask[i, j]) > 0):
                overlap_mask[i, j] = 1

    alpha_mask = np.zeros((hr, wr))  # alpha value depend on left image
    for i in range(hr):
        minIdx = maxIdx = -1
        for j in range(wr):
            if (overlap_mask[i, j] == 1 and minIdx == -1):
                minIdx = j
            if (overlap_mask[i, j] == 1):
                maxIdx = j

        if (minIdx == maxIdx):
            continue

        decrease_step = 1 / (maxIdx - minIdx)

        middleIdx = int((maxIdx + minIdx) / 2)

        # left
        for j in range(minIdx, middleIdx + 1):
            if (j >= middleIdx - constant_width):
                alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))
            else:
                alpha_mask[i, j] = 1
        for j in range(middleIdx + 1, maxIdx + 1):
            if (j <= middleIdx + constant_width):
                alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))
            else:
                alpha_mask[i, j] = 0

    linearBlendingWithConstantWidth_img = np.copy(img2)
    linearBlendingWithConstantWidth_img[:hl, :wl] = np.copy(img1)
    for i in range(hr):
        for j in range(wr):
            if (np.count_nonzero(overlap_mask[i, j]) > 0):
                linearBlendingWithConstantWidth_img[i, j] = alpha_mask[i, j] * img1[i, j] + (
                    1 - alpha_mask[i, j]) * img2[i, j]

    return linearBlendingWithConstantWidth_img


def warp(img1, img2, H):
    img_left, img_right = img1, img2
    (hl, wl) = img_left.shape[:2]
    (hr, wr) = img_right.shape[:2]
    # create the (stitch)big image accroding the imgs height and width
    stitch_img = np.zeros((max(hl, hr), wl + wr, 3), dtype="int")

    # Transform Right image(the coordination of right image) to destination iamge(the coordination of left image) with HomoMat
    inv_H = np.linalg.inv(H)
    for i in range(stitch_img.shape[0]):
        for j in range(stitch_img.shape[1]):
            coor = np.array([j, i, 1])
            img_right_coor = inv_H @ coor  # the coordination of right image
            img_right_coor /= img_right_coor[2]

            # you can try like nearest neighbors or interpolation
            y, x = int(round(img_right_coor[0])), int(
                round(img_right_coor[1]))  # y for width, x for height

            # if the computed coordination not in the (hegiht, width) of right image, it's not need to be process
            if (x < 0 or x >= hr or y < 0 or y >= wr):
                continue
            # else we need the tranform for this pixel
            stitch_img[i, j] = img_right[x, y]

    return stitch_img


def stitching(img1, img2, kp1, kp2, matches_pos):
    print("stitching...")
    H = find_homography(kp1, kp2, matches_pos)
    img1_size = img1.shape  # (h, w)
    img2_size = img2.shape  # (h, w)
    corners_l = [[0, 0, 1], [0, img1_size[0]-1, 1], 
                [img1_size[1]-1, 0, 1], [img1_size[1]-1, img1_size[0]-1, 1]]
    corners_l = np.array(corners_l)
    corners_r = []
    for cor_l in corners_l:
        cor_r = H @ cor_l.T
        corners_r.append(cor_r)
    corners_r = np.array(corners_r).astype('int8')

    x1_r = min(min(corners_r[:, 0]), 0)
    y1_r = min(min(corners_r[:, 1]), 0)
    output_size = (img2_size[1]+abs(int(x1_r)), img2_size[0]+np.abs(int(y1_r)))

    A = [[1, 0, np.float64(-x1_r)],  # affine translation matrix
         [0, 1, np.float64(-y1_r)],
         [0, 0, 1]]

    A = np.array(A).astype('float64')
    H = A @ H

    warp_src = cv2.warpPerspective(src=img1, M=H, dsize=output_size)
    warp_dst = cv2.warpPerspective(src=img2, M=A, dsize=output_size)

    warp_src_cylinProject, _, _ = ProjectOntoCylinder(warp_src)
    warp_dst_cylinProject, _, _ = ProjectOntoCylinder(warp_dst)
    blended_image = blending(warp_src_cylinProject, warp_dst_cylinProject)

    creat_im_window("warp_src_verify", warp_src_cylinProject)
    creat_im_window("warp_dst_verify", warp_dst_cylinProject)
    creat_im_window("blended image", blended_image)
    im_show()
    cv2.imwrite("warp_src_verify.jpg", warp_src_cylinProject)
    cv2.imwrite("warp_dst_verify.jpg", warp_dst_cylinProject)
    cv2.imwrite("blended_image.jpg", blended_image)

    return blended_image


if __name__ == '__main__':
    base_img_path = "Photos/Base/Base"
    images = []
    images_gray = []
    for i in range(1, 4):
        img, img_gray = read_img(base_img_path+str(i)+".jpg")
        images.append(img)
        images_gray.append(img_gray)

    # kp_list = []
    # des_list = []
    # for i in range(len(images)):
    #     kp, des = SIFT(images_gray[i])
    #     kp_list.append(kp)
    #     des_list.append(des)

    # goodMatch_pos_list = []
    # for i in range(len(images)-1):
    #     goodMatch_pos = KNN(kp_list[i], kp_list[i+1], des_list[i], des_list[i+1])
    #     goodMatch_pos_list.append(goodMatch_pos)

    for i in range(len(images)-1):
        if i==0:
            kp1, des1 = SIFT(images_gray[i])
            kp2, des2 = SIFT(images_gray[i+1])
            goodMatch_pos = KNN(kp1, kp2, des1, des2)
            result_imgL_to_imgR = stitching(images[i], images[i+1], kp1, kp2, goodMatch_pos)
        else:
            kp1, des1 = SIFT(img_to_gray(result_imgL_to_imgR))
            kp2, des2 = SIFT(images_gray[i+1])
            goodMatch_pos = KNN(kp1, kp2, des1, des2)
            drawMatches(result_imgL_to_imgR, kp1, images[i+1], kp2, goodMatch_pos)
            result_imgL_to_imgR = stitching(result_imgL_to_imgR, images[i+1], kp1, kp2, goodMatch_pos)

    # result_img1_to_img2 = stitching(images[0], images[1], kp_list[0], kp_list[1], goodMatch_pos_list[0])
    # result_img12_to_img3 = stitching(result_img1_to_img2, images[2], kp_list[1], kp_list[2], goodMatch_pos_list[1])
    creat_im_window("result", result_imgL_to_imgR)
    im_show()