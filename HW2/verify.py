import cv2
import numpy as np
import random
import math
import sys
from PIL import Image, ImageChops
from matplotlib import cm

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
    bf = cv2.BFMatcher()
    features_l = features_l.astype('uint8')
    features_r = features_r.astype('uint8')
    matches = bf.knnMatch(features_l, features_r, k=2)

    # matches = sorted(matches, key=lambda x: x.distance)
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    return good


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
        dstPoints.append(kp2[match.trainIdx].pt)
        srcPoints.append(kp1[match.queryIdx].pt)
    dstPoints = np.array(dstPoints)
    srcPoints = np.array(srcPoints)

    H_verify = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
    H_verify = np.array(H_verify[0])

    # np.save(f"H_base_veri_{i}.npy", H_verify)

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

def linearBlending(img1, img2):
        '''
        linear Blending(also known as Feathering)
        '''
        img_left, img_right = img1, img2
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]
        img_left_mask = np.zeros((hr, wr), dtype="int")
        img_right_mask = np.zeros((hr, wr), dtype="int")
        
        # find the left image and right image mask region(Those not zero pixels)
        for i in range(hl):
            for j in range(wl):
                if np.count_nonzero(img_left[i, j]) > 0:
                    img_left_mask[i, j] = 1
        for i in range(hr):
            for j in range(wr):
                if np.count_nonzero(img_right[i, j]) > 0:
                    img_right_mask[i, j] = 1
        
        # find the overlap mask(overlap region of two image)
        overlap_mask = np.zeros((hr, wr), dtype="int")
        for i in range(hr):
            for j in range(wr):
                if (np.count_nonzero(img_left_mask[i, j]) > 0 and np.count_nonzero(img_right_mask[i, j]) > 0):
                    overlap_mask[i, j] = 1
        
        # compute the alpha mask to linear blending the overlap region
        alpha_mask = np.zeros((hr, wr)) # alpha value depend on left image
        for i in range(hr): 
            minIdx = maxIdx = -1
            for j in range(wr):
                if (overlap_mask[i, j] == 1 and minIdx == -1):
                    minIdx = j
                if (overlap_mask[i, j] == 1):
                    maxIdx = j
            
            if (minIdx == maxIdx): # represent this row's pixels are all zero, or only one pixel not zero
                continue
                
            decrease_step = 1 / (maxIdx - minIdx)
            for j in range(minIdx, maxIdx + 1):
                alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))
        
        
        
        linearBlending_img = np.copy(img_right)
        linearBlending_img[:hl, :wl] = np.copy(img_left)
        # linear blending
        for i in range(hr):
            for j in range(wr):
                if ( np.count_nonzero(overlap_mask[i, j]) > 0):
                    linearBlending_img[i, j] = alpha_mask[i, j] * img_left[i, j] + (1 - alpha_mask[i, j]) * img_right[i, j]
        
        return linearBlending_img

def removeBlackBorder(img):
        '''
        Remove img's the black border 
        '''
        h, w = img.shape[:2]
        reduced_h, reduced_w = h, w
        # right to left
        for col in range(w - 1, -1, -1):
            all_black = True
            for i in range(h):
                if (np.count_nonzero(img[i, col]) > 0):
                    all_black = False
                    break
            if (all_black == True):
                reduced_w = reduced_w - 1
                
        # bottom to top 
        for row in range(h - 1, -1, -1):
            all_black = True
            for i in range(reduced_w):
                if (np.count_nonzero(img[row, i]) > 0):
                    all_black = False
                    break
            if (all_black == True):
                reduced_h = reduced_h - 1
        
        return img[:reduced_h, :reduced_w]
    
def trim(im):
    im = Image.fromarray(np.uint8(im))
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return np.array(im.crop(bbox))


def find_translation(img1, img2, H):
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1-1],[w1-1,h1-1],[w1-1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2-1],[w2-1,h2-1],[w2-1,0]]).reshape(-1,1,2)
    

    pts1_ = cv2.perspectiveTransform(pts1, H)
    pts = np.concatenate((pts2, pts1_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    output_size = (xmax-xmin, ymax-ymin)
    # output_size = (w1+abs(int(xmin)), h1+np.abs(int(ymin)))


    A = [[1, 0, t[0]],  # affine translation matrix
         [0, 1, t[1]],
         [0, 0, 1]]
    A = np.array(A).astype("float64")
    
    return A, output_size, t


def mix_and_match(leftImage, warpedImage):
    i1y, i1x = leftImage.shape[:2]
    i2y, i2x = warpedImage.shape[:2]
    print (leftImage[-1,-1])

    # t = time.time()
    black_l = np.where(leftImage == np.array([0,0,0]))
    black_wi = np.where(warpedImage == np.array([0,0,0]))
    # print (time.time() - t)
    print (black_l[-1])

    for i in range(0, i1x):
        for j in range(0, i1y):
            try:
                if(np.array_equal(leftImage[j,i],np.array([0,0,0])) and  np.array_equal(warpedImage[j,i],np.array([0,0,0]))):
                    # print "BLACK"
                    # instead of just putting it with black, 
                    # take average of all nearby values and avg it.
                    warpedImage[j,i] = [0, 0, 0]
                else:
                    if(np.array_equal(warpedImage[j,i],[0,0,0])):
                        # print "PIXEL"
                        warpedImage[j,i] = leftImage[j,i]
                    else:
                        if not np.array_equal(leftImage[j,i], [0,0,0]):
                            bw, gw, rw = warpedImage[j,i]
                            bl,gl,rl = leftImage[j,i]
                            # b = (bl+bw)/2
                            # g = (gl+gw)/2
                            # r = (rl+rw)/2
                            warpedImage[j, i] = [bl,gl,rl]
            except:
                pass
    # cv2.imshow("waRPED mix", warpedImage)
    # cv2.waitKey()
    return warpedImage


def warp(img1, img2, H, A, output_size, i):
    warp_src = cv2.warpPerspective(src=img1, M=A @ H, dsize=output_size)
    warp_dst = cv2.warpPerspective(src=img2, M=A, dsize=output_size)
    stitched_image = mix_and_match(warp_dst, warp_src)
    cv2.imwrite(f"warpped_dst{i}_verify.jpg", warp_dst)
    cv2.imwrite(f"warpped_src{i}_verify.jpg", warp_src)
    cv2.imwrite(f"stitched{i}_verify.jpg", stitched_image)

    return stitched_image

def stitching(imgs, homographies, translations, output_sizes, ts):
    print("stitching...")
    
    H_list = []
    A_list = []
    for i in range(len(homographies)):
        H = homographies[i]
        A = translations[i]
        for j in range(i+1, len(homographies)):
            H = H @ homographies[j]
            A = A @ translations[j]
        H_list.append(H)
        A_list.append(A)

    warpped_imgs = []
    for i in  range(len(images)-1):
        warp_src = cv2.warpPerspective(src=imgs[i], M=A_list[i] @ H_list[i], dsize=output_sizes[i])
        warpped_imgs.append(warp_src)
        cv2.imwrite(f"warp_src{i}_verify.jpg", warp_src)

    warp_dst = cv2.warpPerspective(src=imgs[-1], M=A_list[-1] , dsize=output_sizes[-1])
    cv2.imwrite("warp_dst_verify.jpg", warp_dst)
    warpped_imgs.append(warp_dst)

    # stitched_image = warp_dst
    stitched_images = []
    for i in range(len(warpped_imgs)-1, 0, -1):
        # stitched_image = warpped_imgs[i]
        # h1, w1 = stitched_image.shape[:2]
        # h2, w2 = warpped_imgs[i+1].shape[:2]
        # t = ts[i]
        # stitched_image[t[1]:h1+t[1], t[0]:w1+t[0]] = warpped_imgs[i+1][t[1]:h1+t[1], t[0]:w1+t[0]]

        stitched_image = mix_and_match(warpped_imgs[i-1], warpped_imgs[i])
        stitched_images.append(stitched_image)
        cv2.imwrite(f"blended_image_verify{i}.jpg", stitched_image)

    for i in range(len(stitched_images)-1):
        stitched_image = mix_and_match(stitched_images[i], stitched_images[i+1])

    # creat_im_window("warp_src_verify", warp_src)
    # creat_im_window("warp_dst_verify", warp_dst)
    # creat_im_window("stitched_image_verify", stitched_image)
    # im_show()
    cv2.imwrite("stitched_image_verify.jpg", stitched_image)

    return stitched_image


if __name__ == '__main__':
    base_img_path = "Photos/Base/Base"
    images = []
    images_gray = []
    for i in range(1, 4):
        img, img_gray = read_img(base_img_path+str(i)+".jpg")
        images.append(img)
        images_gray.append(img_gray)

    
    homographies = []
    translations = []
    output_sizes = []
    ts = []
    for i in range(len(images)-1):
        kp1, des1 = SIFT(images_gray[i])
        kp2, des2 = SIFT(images_gray[i+1])
        goodMatch_pos = KNN(kp1, kp2, des1, des2)
        H = find_homography(kp1, kp2, goodMatch_pos)
        homographies.append(H)
        A, output_size, t = find_translation(images[i], images[i+1], H)
        translations.append(A)
        output_sizes.append(output_size)
        ts.append(t)

    # kp1, des1 = SIFT(images_gray[1])
    # kp2, des2 = SIFT(images_gray[2])
    # goodMatch_pos = KNN(kp1, kp2, des1, des2)
    # H = find_homography(kp1, kp2, goodMatch_pos)
    # A, output_size, t = find_translation(images[1], images[2], H)
    # warp1 = warp(images[1],images[2], H, A, output_size, 1)

    # kp1, des1 = SIFT(images_gray[0])
    # kp2, des2 = SIFT(images_gray[1])
    # goodMatch_pos = KNN(kp1, kp2, des1, des2)
    # H = find_homography(kp1, kp2, goodMatch_pos)
    # A, output_size, t = find_translation(images[0], warp1, H)
    # warp0 = warp(images[0], images[1], H, A, output_size, 0)

    # result = mix_and_match(warp0, warp1)
    result = stitching(images, homographies, translations, output_sizes, ts)


    cv2.imwrite(f"result.jpg", result)
    creat_im_window("result", result)
    im_show()
