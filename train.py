import os
import sys
import time
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from config import config
from env import Env
from pixel_wise_a2c import PixelWiseA2C
from utils import ins_bgr2lab, bgr2lab, vis_bgrimg, vis_labimg, init_net, gauss_filter
from mini_batch_loader import MiniBatchLoader
from Unet.unet import ModUnet
from discrimatator import Discriminator

TRAINING_DATA_PATH = "./train.txt"
TESTING_DATA_PATH = "./train.txt"
IMAGE_DIR_PATH = ".//"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = False
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.manual_seed(1)


def train():
    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH,
        TESTING_DATA_PATH,
        IMAGE_DIR_PATH,
        200)
    train_data_size = MiniBatchLoader.count_paths(TRAINING_DATA_PATH)
    indices = np.random.permutation(train_data_size)

    env = Env(config)
    a2c = PixelWiseA2C(config)
    model = init_net(ModUnet(config).to(device), 'kaiming', gpu_ids=[])

    # actor.load_state_dict(torch.load("./model_test1/modela9600_.pth"))
    # actor_target.load_state_dict(torch.load("./model_test1/modela100_.pth"))
    # critic.load_state_dict(torch.load('./model_test2/modelc8000_.pth'))

    optimizer = torch.optim.Adam(model.parameters(), config.base_lr)

    i_index = 0
    episodes = 0

    # training
    while episodes < config.num_episodes:
        r = indices[i_index: i_index + config.batch_size]
        bgr_cs = mini_batch_loader.load_training_data(r)  # bgr
        lab_ins, lab_l, bgr_c = ins_bgr2lab(bgr_cs)

        image = lab_ins
        env.reset(ori_image=lab_l, image=image)
        reward = np.zeros((1))

        vis_bgrimg(bgr_c)
        # vis_labimg(lab_c)

        # forward
        for t in range(config.episode_len):
            image_input = Variable(torch.from_numpy(image).cuda())
            reward_input = Variable(torch.from_numpy(reward).type(torch.FloatTensor).cuda())

            pi_out, v_out, act_mean, act_logstd = model(image_input, config.batch_size, config.num_actions)
            actions, act_prob, p = a2c.act_and_train(pi_out, v_out, reward_input, act_mean, act_logstd, config)

            if episodes % 10 == 0:
                print(actions[0])
                print(np.exp(act_prob)[0])
                paint_amap(actions[0])

            image, reward = env.step(actions.squeeze(), p)
            temp = image.copy()
            temp[:, 1, :, :] = gauss_filter(temp[:, 1, :, :])
            temp[:, 2, :, :] = gauss_filter(temp[:, 2, :, :])
            vis_labimg(temp)
            # a2c.update_dis(image, lab_c, discr, optimizer_D, config)

        # compute loss and backpropagate
        loss1, loss2 = a2c.stop_episode_and_compute_loss(reward=Variable(torch.from_numpy(reward).cuda()), done=True)
        print("epoisode: ", episodes)
        optimizer.zero_grad()
        ((loss1 + loss2) / config.episode_len).backward()
        optimizer.step()

        episodes += 1

        # save model
        if episodes % 400 == 0:
            torch.save(model.state_dict(), "./model_test4/modela{}_.pth".format(episodes))
            # torch.save(discr.state_dict(), "./model_test3/modeldis{}_.pth".format(episodes))
            print('model saved')
        if i_index + config.batch_size >= train_data_size:
            i_index = 0
            indices = np.random.permutation(train_data_size)
        else:
            i_index += config.batch_size
        if i_index + 2 * config.batch_size >= train_data_size:
            i_index = train_data_size - config.batch_size


def paint_amap(acmap):
    image = np.asanyarray(acmap.squeeze(), dtype=np.uint8)
    plt.imshow(image, vmin=1, vmax=config.num_actions)
    plt.colorbar()
    plt.pause(1)
    plt.close()


if __name__ == "__main__":
    train()
