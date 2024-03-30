import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import scipy


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

def getNormalMap(mask):
    normalmap = np.zeros(images[0].shape)
    light = getUnitLight()
    for x in range(images[0].shape[0]) :
        for y in range(images[0].shape[1]):
            if mask[x][y]!=0:
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
    lower_bound = 30 if obj == 'noisy_venus' else 1
    _, mask = cv2.threshold(image_gray, lower_bound, 255, cv2.THRESH_BINARY)
    print("image:", image_gray.shape)
    return mask

def getDepthMap(mask, normalmap):
    img_h, img_w = mask.shape
    N = np.reshape(normalmap, (image_row, image_col, 3))
    print("N:", N.shape)
    # return the index of non-zero value in mask along axis 0 and axis 1 respectively
    obj_h, obj_w = np.where(mask!=0) 
    print("mask shape:", mask.shape)
    num_pixel = np.size(obj_h) 
    print("num of pixels:", num_pixel)
    # create a matrix storing the index of mask 
    fullobj = np.zeros((img_h, img_w)).astype('int')
    for i in range(num_pixel):
        fullobj[obj_h[i], obj_w[i]] = i
    
    M = scipy.sparse.lil_matrix((2*num_pixel, num_pixel))
    v = np.zeros((2*num_pixel, 1))

    for i in range(num_pixel):
        # pixel-wise calculation
        h = obj_h[i]
        w = obj_w[i]
        nx = N[h, w, 0]
        ny = N[h ,w ,1]
        nz = N[h, w, 2]

        # z(x+1, y)-z(x,y) = -nx/nz
        row_idx = i*2 
        if mask[h, w+1]:
            idx_horizontal = fullobj[h, w+1]   
            M[row_idx, i] = -1
            M[row_idx, idx_horizontal] = 1
            v[row_idx] = -nx/nz
        elif mask[h, w-1]:
            idx_horizontal = fullobj[h, w-1]
            M[row_idx, i] = 1
            M[row_idx, idx_horizontal] = -1
            v[row_idx] = -nx/nz

        # z(x, y+1)-z(x,y) = -ny/nz
        row_idx = i*2+1
        if  mask[h+1, w]:
            idx_vertical = fullobj[h+1, w]
            M[row_idx, i] = 1
            M[row_idx, idx_vertical] = -1
            v[row_idx] = -ny/nz
        elif mask[h-1, w]:
            idx_vertical = fullobj[h-1, w]
            M[row_idx, i] = -1
            M[row_idx, idx_vertical] = 1
            v[row_idx] = -ny/nz
    
    # Mz = v => M.T*M*z = M.T*V => z = (M.T*M)^-1*M.T*V
    Mt_M = M.T@M
    Mt_V = M.T@v
    z = scipy.sparse.linalg.spsolve(Mt_M, Mt_V)

    if obj == 'venus':
        z_std = np.std(z, ddof=1)
        z_mean = np.mean(z)
        z_normalize = (z-z_mean)/z_std

        outlier_ind = np.abs(z_normalize)>10
        z_min = np.min(z[~outlier_ind])
        z_max = np.max(z[~outlier_ind])

    Z = mask.astype('float')
    for i in range(num_pixel):
        h = obj_h[i]
        w = obj_w[i]
        if obj == 'venus':
            Z[h, w] = (z[i]-z_min)/(z_max-z_min)*255
        else:
            Z[h, w] = z[i]
    
    return Z

if __name__ == '__main__':
    images=[]
    obj = 'noisy_venus' # bunny, star, venus, noisy_venus
    image_path=f'./test/{obj}/pic' 
    light_path=f'./test/{obj}/LightSource.txt'
    filepath = f'./result/{obj}.ply'    

    for i in range(1,7):
        img=read_bmp(image_path+f'{i}.bmp')
        if obj=='noisy_venus':
            # img = cv2.blur(img, (5,5))
            img = cv2.GaussianBlur(img, (13, 13), 0)
        images.append(img)
    images = np.array(images)

    mask = getMask(images[0])
    normalmap=getNormalMap(mask)
    Z = getDepthMap(mask, normalmap)

    normal_visualization(normalmap)
    depth_visualization(Z)
    save_ply(Z,filepath)
    show_ply(filepath)

    # showing the windows of all visualization function
    plt.show()
