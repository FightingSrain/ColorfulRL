import numpy as np
import sys
import cv2
import torch
from skimage.measure import compare_ssim
from utils import gauss_filter, sigmoid


class Env():
    def __init__(self, config):
        self.image = None
        self.previous_image = None

        self.num_actions = config.num_actions
        self.actions = config.actions
        self.batch_size = config.batch_size

    def reset(self, ori_image, image):
        self.ori_image = ori_image
        self.image = image
        self.previous_image = None

    def step(self, act, p, dis=None):
        self.previous_image = self.image.copy()
        canvas = [np.zeros(self.image.shape, self.image.dtype) for _ in range(self.num_actions + 1)]
        b, c, h, w = self.image.shape

        k1 = 0.2
        k2 = 0.15
        for i in range(b):
            if np.sum(act[i] == self.actions['red']) > 0:
                i1 = np.expand_dims(self.image[i, 0, :, :], 0)
                i2 = np.expand_dims(self.image[i, 1, :, :], 0)
                i3 = np.expand_dims(self.image[i, 2, :, :], 0)
                canvas[self.actions['red']][i] = np.concatenate((i1, i2 + k1 + np.tanh(p[i][0]) * k2, i3), 0)

            if np.sum(act[i] == self.actions['green']) > 0:
                i1 = np.expand_dims(self.image[i, 0, :, :], 0)
                i2 = np.expand_dims(self.image[i, 1, :, :], 0)
                i3 = np.expand_dims(self.image[i, 2, :, :], 0)
                canvas[self.actions['green']][i] = np.concatenate((i1, i2 - k1 - np.tanh(p[i][1]) * k2, i3), 0)

            if np.sum(act[i] == self.actions['yellow']) > 0:
                i1 = np.expand_dims(self.image[i, 0, :, :], 0)
                i2 = np.expand_dims(self.image[i, 1, :, :], 0)
                i3 = np.expand_dims(self.image[i, 2, :, :], 0)
                canvas[self.actions['yellow']][i] = np.concatenate((i1, i2, i3 + k1 + np.tanh(p[i][2]) * k2), 0)

            if np.sum(act[i] == self.actions['blue']) > 0:
                i1 = np.expand_dims(self.image[i, 0, :, :], 0)
                i2 = np.expand_dims(self.image[i, 1, :, :], 0)
                i3 = np.expand_dims(self.image[i, 2, :, :], 0)
                canvas[self.actions['blue']][i] = np.concatenate((i1, i2, i3 - k1 - np.tanh(p[i][3]) * k2), 0)

            if np.sum(act[i] == self.actions['r_g+']) > 0:
                i1 = np.expand_dims(self.image[i, 0, :, :], 0)
                i2 = np.expand_dims(self.image[i, 1, :, :], 0)
                i3 = np.expand_dims(self.image[i, 2, :, :], 0)
                canvas[self.actions['r_g+']][i] = np.concatenate(
                    (i1, i2 + k1 + np.tanh(p[i][4]) * k2, i3 + k1 + np.tanh(p[i][4]) * k2), 0)

            if np.sum(act[i] == self.actions['y_b+']) > 0:
                i1 = np.expand_dims(self.image[i, 0, :, :], 0)
                i2 = np.expand_dims(self.image[i, 1, :, :], 0)
                i3 = np.expand_dims(self.image[i, 2, :, :], 0)
                canvas[self.actions['y_b+']][i] = np.concatenate(
                    (i1, i2 - k1 - np.tanh(p[i][5]) * k2, i3 - k1 - np.tanh(p[i][5]) * k2), 0)

            if np.sum(act[i] == self.actions['r_g-']) > 0:
                i1 = np.expand_dims(self.image[i, 0, :, :], 0)
                i2 = np.expand_dims(self.image[i, 1, :, :], 0)
                i3 = np.expand_dims(self.image[i, 2, :, :], 0)
                canvas[self.actions['r_g-']][i] = np.concatenate(
                    (i1, i2 + k1 + np.tanh(p[i][6]) * k2, i3 - k1 + np.tanh(p[i][6]) * k2), 0)

            if np.sum(act[i] == self.actions['y_b-']) > 0:
                i1 = np.expand_dims(self.image[i, 0, :, :], 0)
                i2 = np.expand_dims(self.image[i, 1, :, :], 0)
                i3 = np.expand_dims(self.image[i, 2, :, :], 0)
                canvas[self.actions['y_b-']][i] = np.concatenate(
                    (i1, i2 - k1 - np.tanh(p[i][7]) * k2, i3 + k1 + np.tanh(p[i][7]) * k2), 0)

        for a in range(1, self.num_actions + 1):
            self.image = np.where(act[:, np.newaxis, :, :] == a, canvas[a], self.image)

        self.image[:, 0, :, :] = np.clip(self.image[:, 0, :, :], a_min=0, a_max=1)
        self.image[:, 1, :, :] = np.clip(self.image[:, 1, :, :], a_min=-1, a_max=1)
        self.image[:, 2, :, :] = np.clip(self.image[:, 2, :, :], a_min=-1, a_max=1)

        reward = (np.sum(np.square((self.ori_image - self.previous_image)), axis=1)[:, np.newaxis, :, :]) \
                 - (np.sum(np.square((self.ori_image - self.image)), axis=1)[:, np.newaxis, :, :])
        reward *= 255.
        # l_t = torch.Tensor(self.ori_image)
        # s_d_t = torch.Tensor(self.previous_image)
        # s_d_next = torch.Tensor(self.image)
        # pre = torch.cat([l_t, s_d_t], dim=1)
        # cur = torch.cat([l_t, s_d_next], dim=1)
        #
        # T1 = dis(pre.cuda())
        # T2 = dis(cur.cuda())
        # reward = (T2.data.cpu().numpy() - T1.data.cpu().numpy())
        return self.image, reward
