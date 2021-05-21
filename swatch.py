# important libraries
import numpy as np
from scipy.ndimage import generic_filter
import matplotlib.pyplot as plt
import cv2
import time
import random

# array for storing target swatches coordinates
swatch_position = []

# loading source images
img_source0 = cv2.imread("source_ship.png")
img_source1 = []
img_source2 = np.copy(img_source0)
src = np.copy(img_source0)
src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

# loading target images
img_target0 = cv2.imread("target_ship.png")
img_target1 = []
img_target2 = np.copy(img_target0)
target = np.copy(img_target0)
target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)


# for marking which pixels come under swatches
mark = np.zeros((img_target0.shape[0], img_target0.shape[1]))

rect_source = []
rect_target = []
cropping = False

# to create swatches on source image


def crop_image_source(event, x, y, flags, param):
    global rect_source, cropping
    if event == cv2.EVENT_LBUTTONDOWN:
        rect_source = [(x, y)]
        cropping = True
    elif event == cv2.EVENT_LBUTTONUP:
        rect_source.append((x, y))
        cropping = False
        cv2.rectangle(
            img_source1, rect_source[0], rect_source[1], (0, 255, 0), 2)
        cv2.imshow('image', img_source1)

# to create swatches on target image


def crop_image_target(event, x, y, flags, param):
    global rect_target, cropping
    if event == cv2.EVENT_LBUTTONDOWN:
        rect_target = [(x, y)]
        cropping = True
    elif event == cv2.EVENT_LBUTTONUP:
        rect_target.append((x, y))
        cropping = False
        cv2.rectangle(
            img_target1, rect_target[0], rect_target[1], (0, 255, 0), 2)
        cv2.imshow('image', img_target1)

# keep looping until the 'c' key is pressed


def part(image, clone):
    while True:
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            image = clone.copy()
        elif key == ord("c"):
            break

# function to colour individual swatches
# similar as that of global algorithum


def global1(img_target, img_source):
    def lab_image(img_array):
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)

    src_img = lab_image(img_source)

    src_img = src_img.astype(np.float64)

    def getLumin(image_array):
        return image_array[:, :, 0]

    Lumin_src = getLumin(src_img)

    def remapped(source, target_image):
        mu_source = np.mean(source)
        mu_target = np.mean(target_image)
        std_source = np.std(source)
        if std_source == 0:
            std_source = 1
        std_target = np.std(target_image)
        source = (std_target / std_source) * (source - mu_source) + mu_target
        return source

    source_remapped = remapped(Lumin_src, img_target)

    mini = np.amin(source_remapped)
    source_remapped = source_remapped-mini

    maxi = np.amax(source_remapped)
    if max == 0:
        maxi = 1
    source_remapped = source_remapped*225/maxi

    def sd_neighbourhood(image, N):
        return generic_filter(image, np.std, size=N)

    res_src = sd_neighbourhood(Lumin_src, 5)

    res_tar = sd_neighbourhood(img_target, 5)

    result = np.full(
        (img_target.shape[0], img_target.shape[1], 3), 128, dtype=np.float64)
    result[:, :, 0] = img_target

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
                y = random.randint((j-1)*N, j*N-1)
                x = random.randint((i-1)*M, i*M-1)
                arr.append((x, y))
                arr2.append(image[x][y])
                arr_std.append(res_src[x][y])
        return np.asarray(arr), np.asarray(arr2), np.asarray(arr_std)

    eta, k, l = jitter_sampling(Lumin_src, 5, 5)

    def best_match(target_image, target_image_std, samples_arr_lumin, samples_arr_index, samples_arr_std):
        weight_1 = 0.5
        weight_2 = 1-weight_1
        weighted_sum = (weight_1 * np.square(samples_arr_lumin-target_image)) + \
            (weight_2 * np.square(samples_arr_std - target_image_std))
        min_index = np.argmin(weighted_sum)
        return samples_arr_index[min_index][:]

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            [a, b] = best_match(result[i][j][0], res_tar[i][j], k, eta, l)
            alpha_channel = src_img[a][b][1]
            beta_channel = src_img[a][b][2]
            result[i][j][1] = alpha_channel
            result[i][j][2] = beta_channel
    result1 = result.astype('uint8')
    result2 = cv2.cvtColor(result1, cv2.COLOR_LAB2BGR)
    img_target2[rect_target[0][1]:rect_target[1][1],
                rect_target[0][0]:rect_target[1][0]] = np.copy(result2)
    cv2.imshow('image', img_target2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


user = 3
# loop to create user defined number of swatches
for p in range(0, user):

    img_source1 = np.copy(img_source0)
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", crop_image_source)

    part(img_source1, img_source2)

    img_source = np.copy(
        src[rect_source[0][1]:rect_source[1][1], rect_source[0][0]:rect_source[1][0]])

    img_target1 = np.copy(img_target0)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", crop_image_target)

    part(img_target1, img_target0)
    cv2.destroyAllWindows()

    img_target = np.copy(
        target[rect_target[0][1]:rect_target[1][1], rect_target[0][0]:rect_target[1][0], 0])
    img_target = img_target.astype(np.float64)
    mark[rect_target[0][1]:rect_target[1][1], rect_target[0][0]:rect_target[1]
         [0]] = np.ones((img_target.shape[0], img_target.shape[1]))
    swatch_position.append(
        (rect_target[0][1], rect_target[0][0], rect_target[1][1], rect_target[1][0]))

    global1(img_target, img_source)
    rect_target = []
    rect_source = []

d = 3

dicti = {}
value = []

# storing swatches in the dictionary
for k in range(0, len(swatch_position)):
    x1 = swatch_position[k][0]
    y1 = swatch_position[k][1]
    x2 = swatch_position[k][2]
    y2 = swatch_position[k][3]
    value = np.copy(img_target2[x1:x2, y1:y2])
    value = cv2.copyMakeBorder(value, 2*d, 2*d, 2*d, 2*d, cv2.BORDER_REPLICATE)
    value = cv2.cvtColor(value, cv2.COLOR_BGR2LAB)
    dicti[k] = value


target = np.copy(img_target2)
target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)


target = cv2.copyMakeBorder(target, 2*d, 2*d, 2*d, 2*d, cv2.BORDER_REPLICATE)

mark = cv2.copyMakeBorder(mark, 2*d, 2*d, 2*d, 2*d, cv2.BORDER_REPLICATE)


temp = []
swamp = []
each = []
put = []

start = time.time()

# loop to colour the rest of the target image
#  taking neighbour hood window of 7 by 7
for i in range(2*d, target.shape[0], 2*d):
    if i >= target.shape[0]-d:
        continue
    for j in range(2*d, target.shape[1], 2*d):
        if j >= target.shape[1]-d:
            continue
        if mark[i-d][j-d] == 1 and mark[i-d][j+d] == 1 and mark[i+d][j-d] == 1 and mark[i+d][j+d] == 1:
            continue
        temp = np.copy(target[i-d:i+d, j-d:j+d, 0])
        temp = temp.astype(np.float64)
        error = 100000000

        # to loop over each swatches of target
        for k in range(0, len(swatch_position)):
            swamp = dicti[k]

           # interating over a swa by considering a neighbourhood window  of 7 by 7
            for l in range(2*d, swamp.shape[0], 2*d):
                if l >= swamp.shape[0]-d:
                    continue
                for m in range(2*d, swamp.shape[1], 2*d):
                    if m >= swamp.shape[1]-d:
                        continue
                    each = np.copy(swamp[l-d:l+d, m-d:m+d, 0])
                    each = each.astype(np.float64)

                    # computing error for particular window
                    compute = np.sum(np.square(temp-each))
                    if error > compute:
                        put = swamp[l-d:l+d, m-d:m+d]
                        error = compute
        # assigning colour to grey pixel neighbourhood window
        # assigning colour of window having least error among all swatches

        target[i-d:i+d, j-d:j+d, 1] = put[:, :, 1]
        target[i-d:i+d, j-d:j+d, 2] = put[:, :, 2]

# computing time to colourize the target grey image after swatches colouring
end = time.time()
print(end - start)


target = target[2*d:target.shape[0]-2*d, 2*d:target.shape[1]-2*d]
target = cv2.cvtColor(target, cv2.COLOR_LAB2BGR)
cv2.imshow('image', target)
cv2.waitKey(0)
cv2.destroyAllWindows()
