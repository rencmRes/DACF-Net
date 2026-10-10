import torch
import torch.nn as nn


class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label_val', torch.tensor(real_label_val))
        self.register_buffer('fake_label_val', torch.tensor(fake_label_val))
        if gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError('GAN type [{}] is not supported.'.format(gan_type))

    def forward(self, prediction, target_is_real):
        label = self.real_label_val if target_is_real else self.fake_label_val
        return self.loss(prediction, label.expand_as(prediction))
