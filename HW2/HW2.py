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


def KNN(kps_l, kps_r, features_l, features_r, ratio):
    print("Finding good matches...")
    Match_idxAndDist = []
    for i in range(len(features_l)):
        min_idxDis = [-1, np.inf]
        secMin_idxDis = [-1, np.inf]
        for j in range(len(features_r)):
            dist = np.linalg.norm(features_l[i] - features_r[j])
            if (min_idxDis[1] > dist):
                secMin_idxDis = np.copy(min_idxDis)
                min_idxDis = [j, dist]
            elif (secMin_idxDis[1] > dist and secMin_idxDis[1] != min_idxDis[1]):
                secMin_idxDis = [j, dist]

        Match_idxAndDist.append(
            [min_idxDis[0], min_idxDis[1], secMin_idxDis[0], secMin_idxDis[1]])

    goodMatches = []
    for i in range(len(Match_idxAndDist)):
        if (Match_idxAndDist[i][1] <= Match_idxAndDist[i][3]*ratio):
            # store the index of a good match
            goodMatches.append((i, Match_idxAndDist[i][0]))

    print("Match:", len(Match_idxAndDist))
    goodMatches_pos = []
    for (idx, correspondingidx) in goodMatches:
        # psA means the coordinate of the goodMatch from image1
        # psB means the coordinate of the goodMatch from image2
        psA = (int(kps_l[idx].pt[0]), int(kps_l[idx].pt[1]))
        psB = (int(kps_r[correspondingidx].pt[0]),
               int(kps_r[correspondingidx].pt[1]))
        goodMatches_pos.append([psA, psB])
    # print("goodMatchpos:", goodMatches_pos)

    return goodMatches_pos


def drawMatches(img1, img2, matches_pos):
    img_left, img_right = img1, img2
    (hl, wl) = img_left.shape[:2]
    (hr, wr) = img_right.shape[:2]
    vis = np.zeros((max(hl, hr), wl+wr, 3), dtype='uint8')
    vis[0:hl, 0:wl] = img_left
    vis[0:hr, wl:] = img_right

    for (img_left_pos, img_right_pos) in matches_pos:
        pos_l = img_left_pos
        pos_r = img_right_pos[0]+wl, img_right_pos[1]
        cv2.circle(vis, pos_l, 3, (0, 0, 255), 1)
        cv2.circle(vis, pos_r, 3, (0, 255, 0), 1)
        cv2.line(vis, pos_l, pos_r, (255, 0, 0), 1)

    creat_im_window("img with matching points", vis)
    im_show()
    cv2.imwrite("match.jpg", vis)
    return vis


def find_homography(P, m):
    A = []
    for r in range(len(P)):
        A.append([-P[r, 0], -P[r, 1], -1,
                  0, 0, 0,
                  P[r, 0]*m[r, 0], P[r, 1]*m[r, 0], m[r, 0]])
        A.append([0, 0, 0,
                  -P[r, 0], -P[r, 1], -1,
                  P[r, 0]*m[r, 1], P[r, 1]*m[r, 1], m[r, 1]])

    u, s, vt = np.linalg.svd(A)
    H = np.reshape(vt[8], (3, 3))
    H = (1/H.item(8))*H

    return H

def RANSAC(matches_pos):
    dstPoints = []
    srcPoints = []
    for dstPt, srcPt in matches_pos:
        dstPoints.append(list(dstPt))
        srcPoints.append(list(srcPt))
    dstPoints = np.array(dstPoints)
    srcPoints = np.array(srcPoints)

    numSample = len(matches_pos)
    numRandomSample = 4
    thresh = 4.0
    iters = 8000
    Maxinliners = 0
    Best_H = None

    for i in range(iters):
        sampleIdx = random.sample(range(numSample), numRandomSample)
        H_tmp = find_homography()
        for s in sampleIdx:
            dist_esti = np.linalg.norm(dstPoints[s], srcPoints[s])




if __name__ == '__main__':
    base_img_path = "Photos/Base/Base"
    images = []
    images_gray = []
    for i in range(1, 4):
        img, img_gray = read_img(base_img_path+str(i)+".jpg")
        images.append(img)
        images_gray.append(img_gray)

    kp_list = []
    des_list = []
    for i in range(len(images)):
        kp, des = SIFT(images_gray[i])
        kp_list.append(kp)
        des_list.append(des)

    goodMatch_pos_list = []
    for i in range(len(images)-1):
        goodMatch_pos = KNN(kp_list[i], kp_list[i+1],
                            des_list[i], des_list[i+1], 0.75)
        goodMatch_pos_list.append(goodMatch_pos)

    # for i in range(len(images)-1):
    #     drawMatches(images[i], images[i+1], goodMatch_positions[i])
