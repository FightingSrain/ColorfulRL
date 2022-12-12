import torch
import math
import numpy as np
import h5py
import cv2
from torch.autograd import Variable
from torch.distributions import Categorical
from torch import autograd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PixelWiseA2C:
    def __init__(self, config):

        self.gamma = config.gamma

        self.t = 0
        self.t_start = 0
        self.past_action_log_prob = {}
        self.past_action_entropy = {}
        self.past_rewards = {}
        self.past_values = {}
        self.hy_action = {}
        self.hy_act_log_prob = {}
        self.hy_action_entropy = {}

    def reset(self):
        self.past_action_log_prob = {}
        self.past_action_entropy = {}
        self.past_states = {}
        self.past_rewards = {}
        self.past_values = {}
        self.hy_action = {}
        self.hy_act_log_prob = {}
        self.hy_action_entropy = {}

        self.t_start = 0
        self.t = 0
    """
    异步更新参数
    """

    def sync_parameters(self, actor, shared_actor):
        for m1, m2 in zip(actor.modules(), shared_actor.modules()):
            m1._buffers = m2._buffers.copy()
        for target_param, param in zip(actor.parameters(), shared_actor.parameters()):
            target_param.detach().copy_(param.detach())

    """
    异步更新梯度
    """

    def update_grad(self, target, source):
        target_params = dict(target.named_parameters())
        # print(target_params)
        for param_name, param in source.named_parameters():
            if target_params[param_name].grad is None:
                if param.grad is None:
                    pass
                else:
                    target_params[param_name].grad = param.grad
            else:
                if param.grad is None:
                    target_params[param_name].grad = None
                else:
                    target_params[param_name].grad[...] = param.grad
    def cal_gradient_penalty(self, netD, real_data, fake_data, batch_size, config):
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous()
        alpha = alpha.view(batch_size, 3 * 2, config.img_size, config.img_size)  # (16, 1, 63, 63)
        alpha = alpha.to(device)

        fake_datas = fake_data.view(batch_size, 3 * 2, config.img_size, config.img_size)  # (16, 1, 63, 63)
        interpolates = Variable(alpha * real_data.data + ((1 - alpha) * fake_datas.data), requires_grad=True)
        disc_interpolates = netD(interpolates)
        gradients = autograd.grad(disc_interpolates, interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                  create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
        return gradient_penalty

    def update_dis(self, fake_data, real_data, dis, opt_D, config):
        fake_datas = torch.Tensor(fake_data).detach()
        real_datas = torch.Tensor(real_data).detach()

        fake = torch.cat([real_datas, fake_datas], 1)
        real = torch.cat([real_datas, real_datas], 1)
        D_real = dis(real.cuda())
        D_fake = dis(fake.cuda())
        gradient_penalty = self.cal_gradient_penalty(dis, real.cuda(), fake.cuda(), real.size(0), config)
        opt_D.zero_grad()
        # WGAN
        D_cost = (D_fake.mean() - D_real.mean()) + gradient_penalty
        # LSGAN
        # D_cost = (((D_fake) ** 2).mean() + ((D_real - 1) ** 2).mean())/ 2.0  # pixel2pixel
        # D_cost = ((D_fake + 1) ** 2).mean() + ((D_real - 1) ** 2).mean()
        print("Cost:", D_cost.data)
        print("Dfake:", D_fake.mean().data)
        print("Dreal:", D_real.mean().data)
        print("gradient_penalty:", gradient_penalty)
        D_cost.backward()
        opt_D.step()

    def compute_loss(self):
        assert self.t_start < self.t
        R = 0

        pi_loss = 0
        pi_hy_loss = 0
        v_loss = 0
        entropy_hy_loss = 0
        entropy_loss = 0
        for i in reversed(range(self.t_start, self.t)):
            R *= self.gamma
            R += self.past_rewards[i]
            v = self.past_values[i]
            advantage = R - v.detach()

            selected_log_prob_hy = self.hy_act_log_prob[i]
            entropy_hy = self.hy_action_entropy[i]

            selected_log_prob = self.past_action_log_prob[i]
            entropy = self.past_action_entropy[i]

            pi_hy_loss -= selected_log_prob_hy * advantage
            pi_loss -= selected_log_prob * advantage

            entropy_hy_loss -= entropy_hy
            entropy_loss -= entropy

            v_loss += (v - R) ** 2 / 2.

        v_loss *= 0.5

        entropy_loss *= 0.01
        entropy_hy_loss *= 0.01

        losspi = pi_loss.mean() + entropy_loss.mean() + \
                 pi_hy_loss.mean() + entropy_hy_loss.mean()

        lossv = v_loss.mean()

        print("pi_loss: ", pi_loss.mean())
        print("pi_hy_loss: ", pi_hy_loss.mean())
        print("loss_v: ", lossv.mean())
        return losspi, lossv

    def _normal_logproba(self, x, mean, logstd, std=None):
        if std is None:
            std = torch.exp(logstd)
        std_sq = std.pow(2)
        logproba = - 0.5 * math.log(2 * math.pi) - logstd - (x - mean).pow(2) / (2 * std_sq)
        return logproba

    def act_and_train(self, pi, value, reward, act_mean, act_logstd, config):
        self.past_rewards[self.t - 1] = reward

        def randomly_choose_actions(pi):
            pi = torch.clamp(pi, min=0)
            n, num_actions, h, w = pi.shape
            pi_reshape = pi.permute(0, 2, 3, 1).contiguous().view(-1, num_actions)
            m = Categorical(pi_reshape.detach())
            actions = m.sample()

            log_pi_reshape = torch.log(torch.clamp(pi_reshape, min=1e-9, max=1 - 1e-9))
            entropy = -torch.sum(pi_reshape * log_pi_reshape, dim=-1).view(n, 1, h, w)

            selected_log_prob = torch.gather(log_pi_reshape, 1, Variable(actions.unsqueeze(-1))).view(n, 1, h, w)

            actions = actions.view(n, h, w)

            return actions, entropy, selected_log_prob

        actions, entropy, selected_log_prob = randomly_choose_actions(pi)

        action_stds = torch.exp(act_logstd)
        hy_act = torch.normal(act_mean, action_stds)
        hy_acts = (hy_act * torch.ones(hy_act.size(0), hy_act.size(1),
                                       config.img_size, config.img_size).cuda()).gather(1, actions.unsqueeze(1))

        hy_means = (act_mean * torch.ones(act_mean.size(0), act_mean.size(1),
                                          config.img_size, config.img_size).cuda()).gather(1, actions.unsqueeze(1))
        hy_logstds = (act_logstd * torch.ones(act_logstd.size(0), act_logstd.size(1),
                                              config.img_size, config.img_size).cuda()).gather(1, actions.unsqueeze(1))
        action_stds_t = torch.exp(hy_logstds)
        # print(hy_means[0])
        # print(hy_logstds[0])
        # print("HHHHHHHHHHH")

        hy_logproba = self._normal_logproba(hy_acts, hy_means, hy_logstds, action_stds_t)

        self.past_action_log_prob[self.t] = selected_log_prob
        self.past_action_entropy[self.t] = entropy
        self.past_values[self.t] = value
        self.hy_act_log_prob[self.t] = hy_logproba
        self.hy_action_entropy[self.t] = -torch.exp(hy_logproba) * hy_logproba
        self.t += 1
        return actions.detach().cpu().numpy(), selected_log_prob.detach().cpu().numpy(), hy_act.detach().cpu().numpy()

    def test_act_and_train(self, pi):
        def randomly_choose_actions(pi):
            pi = torch.clamp(pi, min=0)
            n, num_actions, h, w = pi.shape
            pi_reshape = pi.permute(0, 2, 3, 1).contiguous().view(-1, num_actions)
            m = Categorical(pi_reshape.detach())
            actions = m.sample()

            log_pi_reshape = torch.log(torch.clamp(pi_reshape, min=1e-9, max=1 - 1e-9))
            entropy = -torch.sum(pi_reshape * log_pi_reshape, dim=-1).view(n, 1, h, w)

            selected_log_prob = torch.gather(log_pi_reshape, 1, Variable(actions.unsqueeze(-1))).view(n, 1, h, w)

            actions = actions.view(n, h, w)

            return actions, entropy, selected_log_prob

        actions, entropy, selected_log_prob = randomly_choose_actions(pi)

        return actions.detach().cpu().numpy(), selected_log_prob.detach().cpu().numpy()

    def stop_episode_and_compute_loss(self, reward, done=False):
        self.past_rewards[self.t - 1] = reward
        if done:
            losspi, lossv = self.compute_loss()
        else:
            raise Exception
        self.reset()
        return losspi, lossv

    
