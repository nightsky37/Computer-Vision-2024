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
        # psA means the coordinate of the goodMatch from image1 (dst)
        # psB means the coordinate of the goodMatch from image2 (src)
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
    # P: points in source plane(right), m: points in target plane(left)
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
    print("finding homography...")
    dstPoints = []
    srcPoints = []
    for dstPt, srcPt in matches_pos:
        dstPoints.append(list(dstPt))
        srcPoints.append(list(srcPt))
    dstPoints = np.array(dstPoints)
    srcPoints = np.array(srcPoints)

    numSample = len(matches_pos)
    numRandomSample = 4
    thresh = 5.0
    iters = 8000
    Maxinliners = 0
    Best_H = None
    verify = False

    for i in range(iters):
        SubsampleIdx = random.sample(range(numSample), numRandomSample)
        H_tmp = find_homography(srcPoints[SubsampleIdx], dstPoints[SubsampleIdx])        
        numInliners = 0
        for j in range(numSample):

            if j not in SubsampleIdx: # cuz we're gonna calculate the the p' using H obtained by random sample
                concatCoor = np.hstack((srcPoints[j], [1])) # add z-axis as 1 -> [xi, yi, 1]
                dstCoor = H_tmp @ concatCoor.T # H dot [xi, yi, 1].T
                if dstCoor[2] <= 1e-8:
                    continue
                dstCoor = dstCoor/dstCoor[2]
                if(np.linalg.norm(dstCoor[:2] - dstPoints[j])<thresh):
                    numInliners += 1
            if(numInliners > Maxinliners):
                Maxinliners = numInliners
                Best_H = H_tmp

        if verify:
            H_verify = cv2.findHomography(srcPoints[SubsampleIdx], dstPoints[SubsampleIdx], cv2.RANSAC, 5.0)
            if i<10:
                print("H_estimated:", Best_H)
                print("H_verify:", H_verify)

    # print("Maximum Inlier:", Maxinliners)

    return Best_H 


def Convert_xy(x, y):
    global center, f

    xt = ( f * np.tan( (x - center[0]) / f ) ) + center[0]
    yt = ( (y - center[1]) / np.cos( (x - center[0]) / f ) ) + center[1]
    
    return xt, yt


def ProjectOntoCylinder(InitialImage):
    global w, h, center, f
    h, w = InitialImage.shape[:2]
    center = [w // 2, h // 2]
    f = 1100       # 1100 field; 1000 Sun; 1500 Rainier; 1050 Helens
    
    # Creating a blank transformed image
    TransformedImage = np.zeros(InitialImage.shape, dtype=np.uint8)
    
    # Storing all coordinates of the transformed image in 2 arrays (x and y coordinates)
    AllCoordinates_of_ti =  np.array([np.array([i, j]) for i in range(w) for j in range(h)])
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
    weight_tr = (dx)       * (1.0 - dy)
    weight_bl = (1.0 - dx) * (dy)
    weight_br = (dx)       * (dy)
    
    TransformedImage[ti_y, ti_x, :] = ( weight_tl[:, None] * InitialImage[ii_tl_y,     ii_tl_x,     :] ) + \
                                      ( weight_tr[:, None] * InitialImage[ii_tl_y,     ii_tl_x + 1, :] ) + \
                                      ( weight_bl[:, None] * InitialImage[ii_tl_y + 1, ii_tl_x,     :] ) + \
                                      ( weight_br[:, None] * InitialImage[ii_tl_y + 1, ii_tl_x + 1, :] )


    # Getting x coorinate to remove black region from right and left in the transformed image
    min_x = min(ti_x)

    # Cropping out the black region from both sides (using symmetricity)
    TransformedImage = TransformedImage[:, min_x : -min_x, :]

    return TransformedImage, ti_x-min_x, ti_y

def blending(img1, img2):

    (hl, wl) = img1.shape[:2]
    (hr, wr) = img2.shape[:2]
    img_left_mask = np.zeros((hr, wr), dtype="int")
    img_right_mask = np.zeros((hr, wr), dtype="int")
    constant_width = 3 # constant width
    
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
    
    alpha_mask = np.zeros((hr, wr)) # alpha value depend on left image
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
                linearBlendingWithConstantWidth_img[i, j] = alpha_mask[i, j] * img1[i, j] + (1 - alpha_mask[i, j]) * img2[i, j]
    
    return linearBlendingWithConstantWidth_img

def stitching(img1, img2, matches_pos):
    print("stitching...")
    H = RANSAC(matches_pos)
    H = np.linalg.inv(H)
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

    A = [[1, 0, -x1_r],  # affine translation matrix
         [0, 1, -y1_r],
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
    # for i in range(1):  #len(images)-1
    #     goodMatch_pos = KNN(kp_list[i], kp_list[i+1],
    #                         des_list[i], des_list[i+1], 0.75)
    #     goodMatch_pos_list.append(goodMatch_pos)

    for i in range(len(images)-1):
        if i==0:
            kp1, des1 = SIFT(images_gray[i])
            kp2, des2 = SIFT(images_gray[i+1])
            goodMatch_pos = KNN(kp1, kp2, des1, des2, 0.75)
            result_imgL_to_imgR = stitching(images[i], images[i+1], goodMatch_pos)
        else:
            kp1, des1 = SIFT(img_to_gray(result_imgL_to_imgR))
            kp2, des2 = SIFT(images_gray[i+1])
            goodMatch_pos = KNN(kp1, kp2, des1, des2, 0.75)
            drawMatches(result_imgL_to_imgR, images[i+1],  goodMatch_pos)
            result_imgL_to_imgR = stitching(result_imgL_to_imgR, images[i+1], goodMatch_pos)

    # result_img1_to_img2 = stitching(images[0], images[1], kp_list[0], kp_list[1], goodMatch_pos_list[0])
    # result_img12_to_img3 = stitching(result_img1_to_img2, images[2], kp_list[1], kp_list[2], goodMatch_pos_list[1])
    creat_im_window("result", result_imgL_to_imgR)
    im_show()
