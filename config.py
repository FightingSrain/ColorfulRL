import time


class config:
    # -------------learning_related--------------------#
    batch_size = 6
    img_size = 100
    num_episodes = 100000
    # -------------rl_related--------------------#
    episode_len = 3
    gamma = 0.98
    # -------------actions--------------------#
    actions = {
        'red': 1,
        'green': 2,
        'yellow': 3,
        'blue': 4,
        'r_g+': 5,
        'y_b+': 6,
        'r_g-': 7,
        'y_b-': 8,
    }
    num_actions = len(actions) + 1
    # -------------lr_policy--------------------#
    base_lr = 0.0003
    base_lr_d = 0.0001
