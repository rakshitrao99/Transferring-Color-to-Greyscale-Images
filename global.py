# important libraries
import numpy as np
from scipy.ndimage import generic_filter
import matplotlib.pyplot as plt
import cv2
import random

# loading source image
img_source1 = cv2.imread("source_ship.png")
img_source = np.copy(img_source1)
cv2.imshow('image',img_source1)
cv2.waitKey(0)
cv2.destroyAllWindows()
img_source=cv2.cvtColor(img_source, cv2.COLOR_BGR2RGB)

# loading target image
img_target1 = cv2.imread("target_ship.png")
cv2.imshow('image',img_target1)
cv2.waitKey(0)
cv2.destroyAllWindows()
img_target=cv2.cvtColor(img_target1, cv2.COLOR_BGR2LAB)
img_target=img_target[:,:,0]
img_target = img_target.astype(np.float64)


# to convert image into LAB
def lab_image(img_array):
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)

src_img = lab_image(img_source)



src_img=src_img.astype(np.float64)

# get L channel of image
def getLumin(image_array):
    return image_array[:,:,0]
    
Lumin_src = getLumin(src_img)

# remapping of source and target histogram
def remapped(source, target_image):
    mu_source = np.mean(source)
    mu_target = np.mean(target_image)
    std_source = np.std(source)
    if std_source==0:
        std_source=1
    std_target = np.std(target_image)
    source = (std_target / std_source) * (source - mu_source) + mu_target
    return source
    
source_remapped = remapped(Lumin_src, img_target)

# adjusting LAB values of source to non negative integer
mini = np.amin(source_remapped)
source_remapped=source_remapped-mini

maxi = np.amax(source_remapped)

if maxi==0:
    maxi=1
source_remapped=source_remapped*225/maxi


# to compute neighbourhood statics
def sd_neighbourhood(image, N):
    return generic_filter(image, np.std, size = N)
    

res_src = sd_neighbourhood(Lumin_src, 5)

res_tar = sd_neighbourhood(img_target, 5)

result = np.zeros((img_target.shape[0], img_target.shape[1], 3))
result[:,:,0] = img_target

# jitter sampling and store the coordinates of sample in array
def jitter_sampling(image, M, N):
    alpha_f = image.shape[0]/M
    alpha = int(alpha_f)
    beta_f = image.shape[1]/N
    beta = int(beta_f)
    arr = []
    arr2 = []
    arr_std = []
    for i in range(1, alpha+1):
        for j in range(1, beta+1):
            y = random.randint((j-1)*N,j*N-1)
            x = random.randint((i-1)*M,i*M-1)
            arr.append((x,y))
            arr2.append(image[x][y])
            arr_std.append(res_src[x][y])
    return np.asarray(arr), np.asarray(arr2), np.asarray(arr_std)


eta, k, l = jitter_sampling(Lumin_src, 20, 20)

# best matching algorithum
def best_match(target_image, target_image_std, samples_arr_lumin, samples_arr_index, samples_arr_std):
    weight_1 = 0.5
    weight_2 = 1-weight_1
    weighted_sum = (weight_1 * np.square(samples_arr_lumin-target_image))+ (weight_2 * np.square(samples_arr_std - target_image_std))
    min_index = np.argmin(weighted_sum)
    return samples_arr_index[min_index][:]
    
# loop for colouring each pixel of the image
for i in range(result.shape[0]):
    for j in range(result.shape[1]):
        [a, b] = best_match(result[i][j][0], res_tar[i][j], k, eta, l)
        alpha_channel = src_img[a][b][1]
        beta_channel = src_img[a][b][2]
        result[i][j][1] = alpha_channel
        result[i][j][2] = beta_channel

result1=result.astype('uint8')
result2=cv2.cvtColor(result1, cv2.COLOR_LAB2BGR)
cv2.imshow('image',result2)
cv2.waitKey(0)
cv2.destroyAllWindows()


    

