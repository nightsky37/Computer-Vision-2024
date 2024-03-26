import cv2
import numpy as np
import open3d as o3d
import matplotlib

matplotlib.use('tkagg')
import matplotlib.pyplot as plt


# visualizing the mask (size : "image width" * "image height")
def mask_visualization(M):
    mask = np.copy(np.reshape(M, (image_row, image_col)))
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')

# visualizing the unit normal vector in RGB color space
# N is the normal map which contains the "unit normal vector" of all pixels (size : "image width" * "image height" * 3)
def normal_visualization(N):
    # converting the array shape to (w*h) * 3 , every row is a normal vetor of one pixel
    N_map = np.copy(np.reshape(N, (image_row, image_col, 3)))
    # Rescale to [0,1] float number
    N_map = (N_map + 1.0) / 2.0
    plt.figure()
    plt.imshow(N_map)
    plt.title('Normal map')

# visualizing the depth on 2D image
# D is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def depth_visualization(D):
    D_map = np.copy(np.reshape(D, (image_row,image_col)))
    # D = np.uint8(D)
    plt.figure()
    plt.imshow(D_map)
    plt.colorbar(label='Distance to Camera')
    plt.title('Depth map')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')

# convert depth map to point cloud and save it to ply file
# Z is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def save_ply(Z,filepath):
    Z_map = np.reshape(Z, (image_row,image_col)).copy()
    data = np.zeros((image_row*image_col,3),dtype=np.float32)
    # let all point float on a base plane 
    baseline_val = np.min(Z_map)
    Z_map[np.where(Z_map == 0)] = baseline_val
    for i in range(image_row):
        for j in range(image_col):
            idx = i * image_col + j
            data[idx][0] = j
            data[idx][1] = i
            data[idx][2] = Z_map[image_row - 1 - i][j]
    # output to ply file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.io.write_point_cloud(filepath, pcd,write_ascii=True)

# show the result of saved ply file
def show_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])

# read the .bmp file
def read_bmp(filepath):
    global image_row
    global image_col
    image = cv2.imread(filepath)
    image_row , image_col = image.shape[:2]
    return image

def getUnitLight():
    light=[]
    with open(light_path, 'r', encoding='utf-8') as txt:
        for line in txt.readlines():
            value = line[7:-2].split(',')
            value = [float(i) for i in value]
            light.append(value)

    light = np.array(light)
    light_norm=[]
    for l in light:
        light_norm.append(l/np.linalg.norm(l))

    light_norm = np.array(light_norm)
    return light_norm

def getNormalMap():
    normalmap = np.zeros(images[0].shape)
    light = getUnitLight()
    for x in range(images[0].shape[0]) :
        for y in range(images[0].shape[1]):
            I = []
            for i in range(6):
                I.append(images[i][x][y])
            I = np.array(I)
            N_T = np.dot(np.dot(np.linalg.inv(np.dot(light.T, light)), light.T), I)        
            N = N_T.T
            
            N_gray = N[0]*0.0722+N[1]*0.7152+N[2]*0.2126 # Kd*N
            Nnorm = np.linalg.norm(N_gray)

            if Nnorm==0:
                continue
            normalmap[x][y] = N_gray/Nnorm                     
    normalmap = normalmap.astype(np.float32)    
    return normalmap

def getMask(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(image_gray, 1, 255, cv2.THRESH_BINARY)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

def getDepthMap():
    h = normalmap.shape[0]
    w = normalmap.shape[1]
    P = np.zeros((h, w, 2), dtype=np.float32)
    Q = np.zeros((h, w, 2), dtype=np.float32)
    tempZ = np.zeros((h, w, 2), dtype=np.float32)
    Z = np.zeros((h, w), dtype=np.float32)
    mask = getMask(images[0])


if __name__ == '__main__':
    images=[]
    image_path=f'./test/star/pic' # bunny, star, venus, noisy_venus
    light_path='./test/star/LightSource.txt'

    for i in range(1,7):
        img=read_bmp(image_path+f'{i}.bmp')
        images.append(img)

    images = np.array(images)
    normalmap=getNormalMap()
    mask = getMask(images[0])
    plt.figure()
    plt.imshow(mask)
    plt.title('mask')


    normal_visualization(normalmap)



    # depth_visualization(Z)
    # save_ply(Z,filepath)
    # show_ply(filepath)

    # showing the windows of all visualization function
    plt.show()