import numpy as np
import cv2
import imutils


SIGMA = 17
SEARCH_REGION_SCALE = 1
LR = 0.125
NUM_PRETRAIN = 128


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

    img = np.log(img + 1)
    img = (img - np.mean(img)) / (np.std(img) + 1e-5)

    # 2d Hanning window
    win_col = np.hanning(width)
    win_row = np.hanning(height)
    mask_col, mask_row = np.meshgrid(win_col, win_row)
    window = mask_col * mask_row
    img = img * window

    return img


def random_warp(img):

    a = -15
    b = 15
    r = a + (b - a) * np.random.uniform()

    height, width = img.shape
    img_rot = imutils.rotate_bound(img, r)
    img_resized = cv2.resize(img_rot, (width, height))

    return img_resized


def initialize(init_frame, init_gt):

    g = get_gauss_response()
    G = np.fft.fft2(g)
    Ai, Bi = pre_training(init_gt, init_frame, G)

    return Ai, Bi, G


def pre_training(init_gt, init_frame, G):

    template = init_frame
    fi = pre_process(template)

    Ai = G * np.conjugate(np.fft.fft2(fi))  # (1a)
    Bi = np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))  # (1b)

    for _ in range(NUM_PRETRAIN):
        fi = pre_process(random_warp(template))

        Ai = Ai + G * np.conjugate(np.fft.fft2(fi))  # (1a)
        Bi = Bi + np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))  # (1b)

    return Ai, Bi


def predict(frame, H):

    fi = frame
    fi = pre_process(fi)
    Gi = H * np.fft.fft2(fi)
    gi = np.real(np.fft.ifft2(Gi))

    return gi
