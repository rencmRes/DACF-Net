import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import models.networks as networks
from models.modules.loss import GANLoss
from .base_model import BaseModel

logger = logging.getLogger('base')


def build_loss(loss_type):
    if loss_type == 'l1':
        return nn.L1Loss()
    if loss_type == 'l2':
        return nn.MSELoss()
    raise NotImplementedError('Loss type [{}] is not supported.'.format(loss_type))


class DVCFModel(BaseModel):
    """Training and inference wrapper for the dual-input DVCF network."""

    def __init__(self, opt):
        super(DVCFModel, self).__init__(opt)
        self.netG = networks.define_G(opt, self.device)
        self.netD = None
        self.netF = None

        if self.is_train:
            self._init_training(opt['train'])
        self.load()

    def _init_training(self, train_opt):
        self.netG.train()
        self.netD = networks.define_D(self.opt, self.device)
        self.netD.train()

        self.pixel_weight = train_opt['pixel_weight'] or 0
        self.feature_weight = train_opt['feature_weight'] or 0
        self.gan_weight = train_opt['gan_weight'] or 0
        self.pixel_loss = build_loss(train_opt['pixel_criterion']).to(self.device)

        if self.feature_weight > 0:
            self.feature_loss = build_loss(train_opt['feature_criterion']).to(self.device)
            self.netF = networks.define_F(self.opt, self.device)
        else:
            self.feature_loss = None

        self.gan_loss = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
        self.d_update_ratio = train_opt['D_update_ratio'] or 1
        self.d_init_iters = train_opt['D_init_iters'] or 0

        self.optimizer_G = torch.optim.Adam(
            self.netG.parameters(), lr=train_opt['lr_G'],
            weight_decay=train_opt['weight_decay_G'] or 0,
            betas=(train_opt['beta1_G'], 0.999))
        self.optimizer_D = torch.optim.Adam(
            self.netD.parameters(), lr=train_opt['lr_D'],
            weight_decay=train_opt['weight_decay_D'] or 0,
            betas=(train_opt['beta1_D'], 0.999))
        self.optimizers.extend([self.optimizer_G, self.optimizer_D])

        for optimizer in self.optimizers:
            self.schedulers.append(lr_scheduler.MultiStepLR(
                optimizer, train_opt['lr_steps'], train_opt['lr_gamma']))
        self.log_dict = OrderedDict()

    def feed_data(self, data, need_HR=True):
        self.var_L = data['LR'].to(self.device, non_blocking=True)
        self.var_SIM = data['SIM'].to(self.device, non_blocking=True)
        if need_HR:
            self.var_H = data['HR'].to(self.device, non_blocking=True)

    @staticmethod
    def _to_vgg_rgb(image):
        return image.repeat(1, 3, 1, 1) if image.size(1) == 1 else image

    def optimize_parameters(self, step):
        update_generator = step % self.d_update_ratio == 0 and step > self.d_init_iters
        self.fake_H = self.netG(self.var_L, self.var_SIM)

        if update_generator:
            for parameter in self.netD.parameters():
                parameter.requires_grad = False
            self.optimizer_G.zero_grad()

            total_g = self.pixel_weight * self.pixel_loss(self.fake_H, self.var_H)
            self.log_dict['l_g_pix'] = total_g.item()

            if self.feature_loss is not None:
                real_feature = self.netF(self._to_vgg_rgb(self.var_H)).detach()
                fake_feature = self.netF(self._to_vgg_rgb(self.fake_H))
                loss_feature = self.feature_weight * self.feature_loss(fake_feature, real_feature)
                total_g = total_g + loss_feature
                self.log_dict['l_g_fea'] = loss_feature.item()

            if self.gan_weight > 0:
                pred_fake = self.netD(self.fake_H)
                pred_real = self.netD(self.var_H).detach()
                loss_gan = self.gan_weight * (
                    self.gan_loss(pred_real - pred_fake.mean(), False) +
                    self.gan_loss(pred_fake - pred_real.mean(), True)) / 2
                total_g = total_g + loss_gan
                self.log_dict['l_g_gan'] = loss_gan.item()

            total_g.backward()
            self.optimizer_G.step()

        for parameter in self.netD.parameters():
            parameter.requires_grad = True
        self.optimizer_D.zero_grad()
        pred_real = self.netD(self.var_H)
        pred_fake = self.netD(self.fake_H.detach())
        loss_d_real = self.gan_loss(pred_real - pred_fake.mean(), True)
        loss_d_fake = self.gan_loss(pred_fake - pred_real.mean(), False)
        ((loss_d_real + loss_d_fake) / 2).backward()
        self.optimizer_D.step()

        self.log_dict['l_d_real'] = loss_d_real.item()
        self.log_dict['l_d_fake'] = loss_d_fake.item()
        self.log_dict['D_real'] = pred_real.detach().mean().item()
        self.log_dict['D_fake'] = pred_fake.detach().mean().item()

    def test(self):
        was_training = self.netG.training
        self.netG.eval()
        with torch.inference_mode():
            self.fake_H = self.netG(self.var_L, self.var_SIM)
        if was_training:
            self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        visuals = OrderedDict(SR=self.fake_H.detach()[0].float().cpu())
        if need_HR:
            visuals['HR'] = self.var_H.detach()[0].float().cpu()
        return visuals

    def load(self):
        load_g = self.opt['path']['pretrain_model_G']
        if load_g:
            logger.info('Loading generator [%s]', load_g)
            self.load_network(load_g, self.netG)
        load_d = self.opt['path']['pretrain_model_D']
        if self.is_train and load_d:
            logger.info('Loading discriminator [%s]', load_d)
            self.load_network(load_d, self.netD)

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        if self.netD is not None:
            self.save_network(self.netD, 'D', iter_step)
