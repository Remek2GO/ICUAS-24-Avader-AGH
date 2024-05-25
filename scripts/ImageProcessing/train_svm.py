# train svm classifier

# import files
import os
import sys
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
import imutils


# read files from directory
def read_files(directory):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            files.append(filename)
    return files


# split into three categories
def split_files(files):
    red_files = []
    yellow_files = []
    none_files = []
    for file in files:
        if "red" in file:
            red_files.append(file)
        elif "yellow" in file:
            yellow_files.append(file)
        elif "none" in file:
            none_files.append(file)
    return red_files, yellow_files, none_files


SIGMA = 17
SEARCH_REGION_SCALE = 1
LR = 0.125

def get_gauss_response():

    width = 20
    height = 20
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))

    center_x = width // 2
    center_y = height // 2
    dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (2 * SIGMA)
    response = np.exp(-dist)

    return response


def pre_process(img):

    height, width = img.shape
    img = img.astype(np.float32)

    #---- TODO (3)
    img = np.log(img + 1)
    img = (img - np.mean(img)) / (np.std(img) + 1e-5)
    #---- TODO(3)

    #2d Hanning window
    win_col = np.hanning(width)
    win_row = np.hanning(height)
    mask_col, mask_row = np.meshgrid(win_col, win_row)
    window = mask_col * mask_row
    img = img * window

    return img

def random_warp(img):

    #---TODO (4)
    a = -15
    b = 15
    r = a + (b - a) * np.random.uniform()

    height, width = img.shape
    img_rot = imutils.rotate_bound(img, r)
    img_resized = cv2.resize(img_rot, (width, height))
    #---TODO (4)

    return img_resized

def initialize(init_frame, init_gt):

    g = get_gauss_response()
    G = np.fft.fft2(g)
    Ai, Bi = pre_training(init_gt, init_frame, G)

    return Ai, Bi, G

NUM_PRETRAIN = 128
def pre_training(init_gt, init_frame, G):

    template = init_frame
    fi = pre_process(template)
    
    Ai = G * np.conjugate(np.fft.fft2(fi))                # (1a)
    Bi = np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))  # (1b)

    for _ in range(NUM_PRETRAIN):
        fi = pre_process(random_warp(template))

        Ai = Ai + G * np.conjugate(np.fft.fft2(fi))               # (1a)
        Bi = Bi + np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi)) # (1b)

    return Ai, Bi

def predict(frame, H):

    #----TODO (5)
    fi = frame
    fi = pre_process(fi)
    Gi = H * np.fft.fft2(fi)
    gi = np.real(np.fft.ifft2(Gi))
    # cv2.imshow('response', gi)   
    #----TODO (5)

    return gi

# read images and flatten them
def read_images_flatten(directory, files):
    images = []
    N = len(files)
    images = np.zeros((N, 11*400))

    pattern_red = cv2.imread("/root/sim_ws/src/icuas24_competition/scripts/ImageProcessing/pattern_red2.png")
    pattern_yellow = cv2.imread("/root/sim_ws/src/icuas24_competition/scripts/ImageProcessing/pattern_yellow2.png")
    pattern_red = cv2.cvtColor(pattern_red, cv2.COLOR_BGR2GRAY)
    pattern_yellow = cv2.cvtColor(pattern_yellow, cv2.COLOR_BGR2GRAY)
    
    it = 0
    for file in files:
        image = cv2.imread(directory + file)
        if image.shape[0] != 20 or image.shape[1] != 20:
            image = cv2.resize(image, (20, 20))


        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(2,2))
        # Apply CLAHE to the grayscale image
        cl1 = clahe.apply(gray)

        thresh = cv2.adaptiveThreshold(cl1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 0)

        luv = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
        l, u, v = cv2.split(luv)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l2, a, b = cv2.split(lab)
        canny = cv2.Canny(cl1, 50, 200)

        ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycbcr)

        init_img = gray

        position = [0,0, 20, 20]
        Ai, Bi, G = initialize(pattern_red, position)
        response_red = predict(init_img, Ai/Bi)
        # print(response_red.max())
        # cv2.imshow("response_red", response_red)

        Ai, Bi, G = initialize(pattern_yellow, position)
        response_yellow = predict(init_img, Ai/Bi)
        # print(response_yellow.max())
        # cv2.imshow("response_yellow", response_yellow)

        images[it,:] = np.concatenate((image.flatten(), u.flatten(),v.flatten(), a.flatten(), b.flatten(), cr.flatten(), cb.flatten(), response_red.flatten(),response_yellow.flatten()))
        # images[it,:] = np.concatenate((image.flatten(),response_red.flatten(),response_yellow.flatten()))
        # images[it,:] = np.concatenate((image.flatten(),response_red.flatten(),response_yellow.flatten()))

        # cv2.imshow("Image", image)
        # cv2.imshow("l", l)
        # cv2.imshow("u", u)
        # cv2.imshow("v", v)
        # cv2.imshow("l2", l2)
        # cv2.imshow("a", a)
        # cv2.imshow("b", b)

        # cv2.imshow("canny", canny)
        # cv2.imshow("thresh", thresh)
        # cv2.imshow("y", y)
        # cv2.imshow("cr", cr)
        # cv2.imshow("cb", cb)

        # cv2.imshow("cl1", cl1)
        # cv2.waitKey(0)

        it += 1
    return images

# read images and get hog features
def read_images_hog(directory, files):
    images = []
    winSize = (20,20)
    blockSize = (10,10)
    blockStride = (5,5)
    cellSize = (5,5)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64

    num_blocks_per_window = ((winSize[0] - blockSize[0]) / cellSize[0] + 1)**2
    num_cells_per_block = (blockSize[0] / cellSize[0])**2
    num_features = int(num_blocks_per_window * num_cells_per_block * nbins)

    N = len(files)
    images = np.zeros((N, num_features))

    # use HOG descriptor
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
    
    it = 0
    for file in files:
        image = cv2.imread(directory + file)
        if image.shape[0] != 20 or image.shape[1] != 20:
            image = cv2.resize(image, (20, 20))

        images[it,:] = np.array(hog.compute(image)).flatten()
        it += 1
    return images


# read images from directory
def read_images_from_directory(directory):
    files = read_files(directory)
    red_files, yellow_files, none_files = split_files(files)
    red_images = read_images_flatten(directory, red_files)
    yellow_images = read_images_flatten(directory, yellow_files)
    none_images = read_images_flatten(directory, none_files)
    return red_images, yellow_images, none_images

# read images from directory
def read_images_hog_from_directory(directory):
    files = read_files(directory)
    red_files, yellow_files, none_files = split_files(files)
    red_images = read_images_hog(directory, red_files)
    yellow_images = read_images_hog(directory, yellow_files)
    none_images = read_images_hog(directory, none_files)
    return red_images, yellow_images, none_images


# read images from directory
directory = "/root/sim_ws/src/icuas24_competition/scripts/svm_images/"
red_images_flatten, yellow_images_flatten, none_images_flatten = read_images_from_directory(directory)
red_images_hog, yellow_images_hog, none_images_hog = read_images_hog_from_directory(directory)


# create labels
red_labels = np.ones(len(red_images_flatten))
yellow_labels = np.ones(len(yellow_images_flatten)) * 2
none_labels = np.ones(len(none_images_flatten)) * 3

## train svm classifier from RGB images

# combine data
data = np.concatenate((red_images_flatten, yellow_images_flatten, none_images_flatten))
labels = np.concatenate((red_labels, yellow_labels, none_labels))

# split data
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train svm
clf = svm.SVC()
clf.fit(X_train, y_train)

svm_classifier = cv2.ml.SVM_create()
svm_classifier.setKernel(cv2.ml.SVM_LINEAR)
svm_classifier.setType(cv2.ml.SVM_C_SVC)

svm_classifier.train(np.float32(X_train), cv2.ml.ROW_SAMPLE, np.int32(y_train))

# predict using OpenCV's SVM
_, y_pred = svm_classifier.predict(np.float32(X_test))
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

# predict
# y_pred = clf.predict(X_test)

# # accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy: ", accuracy)

# # save model
# filename = "svm_model_rgb.sav"
# pickle.dump(clf, open(filename, "wb"))

# display test results
# print("Test results")
# print("True labels: ", y_test)
# print("Predicted labels: ", y_pred)

detect_diff = y_test - y_pred.reshape(-1)
print("Detection errors: ", np.where(detect_diff != 0)[0])

# display images
# for i in np.where(detect_diff != 0)[0]:
#     if y_test[i] == 1:
#         print("True label: red")
#     elif y_test[i] == 2:
#         print("True label: yellow")
#     elif y_test[i] == 3:
#         print("True label: none")

#     if y_pred[i] == 1:
#         print("Predicted label: red")
#     elif y_pred[i] == 2:
#         print("Predicted label: yellow")
#     elif y_pred[i] == 3:
#         print("Predicted label: none")

#     print("Image:")
#     if y_test[i] == 1:
#         cv2.imshow("Image", red_images_flatten[i][:3*400].reshape(20,20,3).astype(np.uint8))
#     elif y_test[i] == 2:
#         cv2.imshow("Image", yellow_images_flatten[i][:3*400].reshape(20,20,3).astype(np.uint8))
#     elif y_test[i] == 3:
#         cv2.imshow("Image", none_images_flatten[i][:3*400].reshape(20,20,3).astype(np.uint8))
#     cv2.waitKey(1)

# cv2.destroyAllWindows()

#save model
svm_classifier.save("svm_model_cv4.xml")




# ## train svm classifier from RGB images
# data = np.concatenate((red_images_hog, yellow_images_hog, none_images_hog))
# labels = np.concatenate((red_labels, yellow_labels, none_labels))

# # split data
# X_train, X_test, y_train, y_test = train_test_split(
#     data, labels, test_size=0.2, random_state=42
# )

# # scale data
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # train svm
# clf = svm.SVC()
# clf.fit(X_train, y_train)

# # predict
# y_pred = clf.predict(X_test)

# # accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy: ", accuracy)

# # save model
# filename = "svm_model_hog.sav"
# pickle.dump(clf, open(filename, "wb"))

# # display test results
# # print("Test results")
# # print("True labels: ", y_test)
# # print("Predicted labels: ", y_pred)


# detect_diff = y_test - y_pred
# print("Detection errors: ", np.where(detect_diff != 0)[0])

# # display images
# for i in np.where(detect_diff != 0)[0]:
#     if y_test[i] == 1:
#         print("True label: red")
#     elif y_test[i] == 2:
#         print("True label: yellow")
#     elif y_test[i] == 3:
#         print("True label: none")

#     if y_pred[i] == 1:
#         print("Predicted label: red")
#     elif y_pred[i] == 2:
#         print("Predicted label: yellow")
#     elif y_pred[i] == 3:
#         print("Predicted label: none")

#     print("Image:")
#     if y_test[i] == 1:
#         cv2.imshow("Image", red_images_flatten[i][:3*400].reshape(20,20,3).astype(np.uint8))
#     elif y_test[i] == 2:
#         cv2.imshow("Image", yellow_images_flatten[i][:3*400].reshape(20,20,3).astype(np.uint8))
#     elif y_test[i] == 3:
#         cv2.imshow("Image", none_images_flatten[i][:3*400].reshape(20,20,3).astype(np.uint8))
#     cv2.waitKey(1)

# cv2.destroyAllWindows()
