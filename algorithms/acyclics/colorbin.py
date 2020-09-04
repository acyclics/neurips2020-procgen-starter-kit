import colour
import numpy as np
import cv2
from skimage import color

rgb_colors = [
    [0, 0, 0], #black
    [255, 0, 0], #red
    [255, 120, 0], #orange
    [255, 255, 0], #yellow
    [125, 255, 0], #spring green
    [0, 255, 0], #green
    [0, 255, 125], #turquoise
    [0, 255, 255], #cyan
    [0, 125, 255], #ocean
    [0, 0, 255], #blue
    [125, 0, 255], #violet
    [255, 0, 255], #magenta
    [255, 0, 125], #raspberry
    [127, 127, 127], #gray
    [255, 255, 255], #white
]

N_RGBS = len(rgb_colors)

lab_colors = [cv2.cvtColor(np.reshape(np.concatenate([np.reshape(np.repeat(c[0], [64**2]), [64, 64, 1]),
                np.reshape(np.repeat(c[1], [64**2]), [64, 64, 1]),
                np.reshape(np.repeat(c[2], [64**2]), [64, 64, 1])], axis=-1).astype(np.float32) / 255, [64, 64, 3]),
                cv2.COLOR_RGB2Lab) for c in rgb_colors]


def assign_rgb(idx):
    return rgb_colors[idx]


def visualize_color_bin(rgb):
    lab = cv2.cvtColor(rgb.astype(np.float32)/255, cv2.COLOR_RGB2Lab)
    lab_delta = np.asarray([colour.delta_E(lc, lab) for lc in lab_colors])
    lab_delta = np.argmin(lab_delta, axis=0)
    for i in range(lab.shape[0]):
        for j in range(lab.shape[1]):
            rgb[i, j, :] = rgb_colors[lab_delta[i,j]]
    return rgb


def visualize_edged_color_bin(rgb):
    rgb_img = visualize_color_bin(rgb)
    edges = cv2.Laplacian(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY), cv2.CV_64F)
    hsv_obs = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    edges = np.where(edges != 0, 1, 0)
    edges = np.uint8(edges)
    edges = np.expand_dims(edges, axis=-1)
    hsv_obs[:, :, 1:3] = hsv_obs[:, :, 1:3] * edges
    rgb = cv2.cvtColor(hsv_obs, cv2.COLOR_HSV2RGB)
    rgb = np.uint8(rgb)

    lab = cv2.cvtColor(rgb.astype(np.float32)/255, cv2.COLOR_RGB2Lab)
    lab_delta = np.asarray([colour.delta_E(lc, lab) for lc in lab_colors])
    lab_delta = np.argmin(lab_delta, axis=0)
    
    for i in range(lab.shape[0]):
        for j in range(lab.shape[1]):
            rgb[i, j, :] = rgb_colors[lab_delta[i,j]]
            
    return rgb


def color_bin(rgb):

    rgb_img = visualize_color_bin(rgb)
    edges = cv2.Laplacian(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY), cv2.CV_64F)
    hsv_obs = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    edges = np.where(edges != 0, 1, 0)
    edges = np.uint8(edges)
    edges = np.expand_dims(edges, axis=-1)
    hsv_obs[:, :, 1:3] = hsv_obs[:, :, 1:3] * edges
    rgb = cv2.cvtColor(hsv_obs, cv2.COLOR_HSV2RGB)
    rgb = np.uint8(rgb)

    lab = cv2.cvtColor(rgb.astype(np.float32)/255, cv2.COLOR_RGB2Lab)
    lab_delta = np.asarray([colour.delta_E(lc, lab) for lc in lab_colors])
    lab_delta = np.argmin(lab_delta, axis=0)
    one_hot_rgb = np.eye(N_RGBS)[lab_delta]

    return one_hot_rgb
