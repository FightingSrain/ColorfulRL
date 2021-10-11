import os
import cv2
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from env import Env
from Unet.unet import ModUnet
from pixel_wise_a2c import PixelWiseA2C
from config import config
from utils import bgr2lab, vis_bgrimg, vis_labimg, gauss_filter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

env = Env(config)
a2c = PixelWiseA2C(config)

actor = ModUnet(config).to(device)
# actor.load_state_dict(torch.load("./model_test3/modela2400_.pth"))
# actor.load_state_dict(torch.load("./model_test4/modela12800_.pth"))

episodes = 0


def tst():
    actor.eval()
    img = cv2.imread("./img_test/test4.jpg") / 255
    bgr_c = np.expand_dims(img, 0).transpose(0, 3, 1, 2)
    lab_c = bgr2lab(bgr_c.astype(np.float32))  # lab
    lab_c[:, 0, :, :] /= 100
    lab_c[:, 1, :, :] /= 127
    lab_c[:, 2, :, :] /= 127
    ll = np.zeros((1, 1, bgr_c.shape[2], bgr_c.shape[3]))
    for i in range(1):
        ll[i] = np.expand_dims(cv2.cvtColor(bgr_c[i].astype(np.float32).transpose(1, 2, 0), cv2.COLOR_RGB2GRAY), 0)
    s = np.concatenate([ll, ll, ll], 1)  # 三通到灰度图 bgr
    lab_ins = bgr2lab(s.astype(np.float32))
    lab_ins[:, 0, :, :] /= 100
    lab_ins[:, 1, :, :] *= 0
    lab_ins[:, 2, :, :] *= 0  # 灰度图转化为lab，并归一化

    image = lab_ins
    env.reset(ori_image=lab_c, image=image)
    vis_bgrimg(bgr_c)
    # forward
    for t in range(3):
        image_input = Variable(torch.from_numpy(image).cuda())

        pi_out, v_out, act_mean, act_logstd = actor(image_input, 1, config.num_actions)

        actions = act_and_train(pi_out)
        # 连续参数
        p = act_mean.tanh().detach().cpu().numpy()
        print(p)

        image, reward = env.step(actions, p)
        # ------------
        temp = image.copy()
        temp[:, 1, :, :] = gauss_filter(temp[:, 1, :, :])
        temp[:, 2, :, :] = gauss_filter(temp[:, 2, :, :])
        # cv2.imwrite("./res_img/" + str(t) + ".png", temp.squeeze().transpose(1, 2, 0))
        vis_labimg(temp, time=0)
        print(actions[0])
        paint_amap(actions[0])
        # ------------


def act_and_train(pi):
    def randomly_choose_actions(pi):
        pi = torch.clamp(pi, min=0)
        n, num_actions, h, w = pi.shape
        _, actions = torch.max(pi, 1)
        actions = actions.view(n, h, w)

        return actions

    actions = randomly_choose_actions(pi)

    return actions.detach().cpu().numpy()


def paint_amap(acmap):
    image = np.asanyarray(acmap.squeeze(), dtype=np.uint8)
    plt.imshow(image, vmin=1, vmax=9)
    plt.colorbar()
    plt.pause(1)
    # plt.show()
    plt.close()


if __name__ == "__main__":
    tst()
