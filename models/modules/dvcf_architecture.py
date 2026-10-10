import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import block as B


class DVCFModule(nn.Module):
    """Difference-aware Verification Cross Fusion.

    The module replaces concat-based fusion with four explicit paths:
    common feature enhancement, WF selection, SIM selection, and conflict
    refinement. It keeps the output channel count equal to nf.
    """

    def __init__(self, nf, norm_type=None, act_type='leakyrelu', mode='CNA', use_refine=True):
        super(DVCFModule, self).__init__()
        self.wf_proj = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        self.sim_proj = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)

        self.cmp_conv = B.conv_block(4 * nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        self.consistency = nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.conflict_reliability = nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.wf_gate = nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.sim_gate = nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.common_conv = B.conv_block(3 * nf, nf, kernel_size=3, norm_type=None,
                                        act_type=act_type)
        self.conflict_conv = B.conv_block(3 * nf, nf, kernel_size=3, norm_type=None,
                                          act_type=act_type)
        self.base_conv = B.conv_block(2 * nf, nf, kernel_size=1, norm_type=None,
                                      act_type=None)
        self.out_conv = B.conv_block(4 * nf, nf, kernel_size=3, norm_type=None,
                                     act_type=None)
        self.refine = B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True,
                             pad_type='zero', norm_type=norm_type, act_type=act_type,
                             mode=mode) if use_refine else None

    def forward(self, wf_fea, sim_fea):
        if sim_fea.size()[2:] != wf_fea.size()[2:]:
            sim_fea = F.interpolate(sim_fea, size=wf_fea.size()[2:],
                                    mode='bilinear', align_corners=False)

        q_w = self.wf_proj(wf_fea)
        q_s = self.sim_proj(sim_fea)

        diff = torch.abs(q_w - q_s)
        prod = q_w * q_s
        cmp_fea = self.cmp_conv(torch.cat([q_w, q_s, diff, prod], dim=1))

        consistency_map = self.consistency(cmp_fea)
        reliability_map = self.conflict_reliability(cmp_fea)
        wf_gate = self.wf_gate(cmp_fea)
        sim_gate = self.sim_gate(cmp_fea)
        common = consistency_map * self.common_conv(torch.cat([q_w, q_s, prod], dim=1))
        wf_selected = wf_gate * q_w
        sim_selected = sim_gate * q_s
        conflict = (1 - consistency_map) * reliability_map * self.conflict_conv(
            torch.cat([q_w, q_s, diff], dim=1))

        base = self.base_conv(torch.cat([q_w, q_s], dim=1))
        fused = base + self.out_conv(torch.cat(
            [common, wf_selected, sim_selected, conflict], dim=1))

        if self.refine is not None:
            fused = self.refine(fused)
        return fused


class DVCFNet(nn.Module):
    """Dual-source network with DVCF feature fusion.

    Input:
        wf:  low-resolution WF image, N x C x H x W
        sim: high-resolution SIM image, N x C x sH x sW

    Output:
        sr_out
    """

    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4, norm_type=None,
                 act_type='leakyrelu', mode='CNA', upsample_mode='upconv'):
        super(DVCFNet, self).__init__()
        if nb < 20:
            raise ValueError('DVCFNet expects at least 20 RRDB blocks.')

        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1
        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        rb_blocks = [B.RRDB(nf, kernel_size=3, gc=gc, stride=1, bias=True,
                            pad_type='zero', norm_type=norm_type, act_type=act_type,
                            mode='CNA') for _ in range(nb)]
        lr_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type,
                               act_type=None, mode=mode)

        if upscale == 3:
            main_upsampler = [upsample_block(nf, nf, 3, act_type=act_type)]
            branch_upsampler = [upsample_block(nf, nf, 3, act_type=act_type)]
        else:
            main_upsampler = [upsample_block(nf, nf, act_type=act_type)
                              for _ in range(n_upscale)]
            branch_upsampler = [upsample_block(nf, nf, act_type=act_type)
                                for _ in range(n_upscale)]

        self.wf_head = fea_conv
        self.wf_blocks = nn.ModuleList(rb_blocks)
        self.wf_lr_conv = lr_conv
        self.wf_upsample = B.sequential(*main_upsampler)
        self.wf_hr_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None,
                                        act_type=act_type)
        self.wf_hr_conv1 = B.conv_block(nf, nf, kernel_size=3, norm_type=None,
                                        act_type=None)

        self.sim_head = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None,
                                     act_type=None)
        self.fuse1 = DVCFModule(nf, norm_type=norm_type, act_type=act_type, mode=mode)
        self.fuse2 = DVCFModule(nf, norm_type=norm_type, act_type=act_type, mode=mode)
        self.fuse3 = DVCFModule(nf, norm_type=norm_type, act_type=act_type, mode=mode)
        self.fuse4 = DVCFModule(nf, norm_type=norm_type, act_type=act_type, mode=mode)

        self.branch_lr_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type,
                                           act_type=None, mode=mode)
        self.branch_upsample = B.sequential(*branch_upsampler)
        self.branch_hr_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None,
                                            act_type=act_type)
        self.branch_hr_conv1 = B.conv_block(nf, nf, kernel_size=3, norm_type=None,
                                            act_type=None)
        self.final_fuse = DVCFModule(nf, norm_type=norm_type, act_type=act_type, mode=mode)
        self.out_hr_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None,
                                         act_type=act_type)
        self.out_hr_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None,
                                         act_type=None)

    def _extract_wf_features(self, wf):
        x = self.wf_head(wf)
        x_ori = x

        for i in range(5):
            x = self.wf_blocks[i](x)
        x_fea1 = x

        for i in range(5, 10):
            x = self.wf_blocks[i](x)
        x_fea2 = x

        for i in range(10, 15):
            x = self.wf_blocks[i](x)
        x_fea3 = x

        for i in range(15, 20):
            x = self.wf_blocks[i](x)
        x_fea4 = x

        for i in range(20, len(self.wf_blocks)):
            x = self.wf_blocks[i](x)
        x = self.wf_lr_conv(x)
        x = x + x_ori

        x_hr = self.wf_upsample(x)
        x_hr = self.wf_hr_conv0(x_hr)
        x_hr = self.wf_hr_conv1(x_hr)
        return x_fea1, x_fea2, x_fea3, x_fea4, x_hr

    def _extract_sim_branch(self, sim, x_fea1, x_fea2, x_fea3, x_fea4):
        sim_fea = self.sim_head(sim)
        if sim_fea.size()[2:] != x_fea1.size()[2:]:
            sim_fea = F.interpolate(sim_fea, size=x_fea1.size()[2:],
                                    mode='bilinear', align_corners=False)
        sim_ori = sim_fea

        sim_fea = self.fuse1(x_fea1, sim_fea)
        sim_fea = self.fuse2(x_fea2, sim_fea)
        sim_fea = self.fuse3(x_fea3, sim_fea)
        sim_fea = self.fuse4(x_fea4, sim_fea)
        sim_fea = self.branch_lr_conv(sim_fea)
        return sim_fea + sim_ori

    def forward(self, wf, sim=None):
        if sim is None:
            sim = wf
        sim_ori = sim

        x_fea1, x_fea2, x_fea3, x_fea4, wf_hr = self._extract_wf_features(wf)

        sim_fea = self._extract_sim_branch(sim, x_fea1, x_fea2, x_fea3, x_fea4)

        sim_fea = self.branch_upsample(sim_fea)
        sim_fea = self.branch_hr_conv0(sim_fea)
        branch_hr = self.branch_hr_conv1(sim_fea)

        fused_hr = self.final_fuse(wf_hr, branch_hr)
        sr = self.out_hr_conv0(fused_hr)
        sr = self.out_hr_conv1(sr)
        return sr
