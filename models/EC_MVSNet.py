import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models.module import *
from models.depth_anything_v2.dpt import DepthAnythingV2


def pro_bilinear_sampler_new(pro, depth_sample, depth_sample_pro):
    c, _, D = pro.shape[-3:]
    # print(pro.shape)
    depth_sample = 1. / depth_sample
    depth_sample_pro = 1. / depth_sample_pro
    b, d, h, w = depth_sample.shape
    _, D_depth_sample, H_depth_sample, W_depth_sample = depth_sample.shape
    _, D_depth_sample_pro, H_depth_sample_pro, W_depth_sample_pro = depth_sample_pro.shape
    depth_sample_pro_mask_ = depth_sample_pro.unsqueeze(2).repeat(1, 1, D_depth_sample, 1, 1)
    depth_sample_mask = depth_sample.unsqueeze(1).repeat(1, D_depth_sample_pro, 1, 1, 1)
    depth_sample_pro_mask = (depth_sample_mask <= depth_sample_pro_mask_).to(torch.float32).sum(dim=2,
                                                                                                keepdims=True).to(
        torch.int64) - 1
    # print(depth_sample_mask, depth_sample_pro_mask_)
    # print((depth_sample_mask >= depth_sample_pro_mask_))
    depth_sample_pro_mask_bool = torch.logical_and(depth_sample_pro_mask >= 0,
                                                   depth_sample_pro_mask <= D_depth_sample - 1)
    depth_sample_pro_mask_claim = depth_sample_pro_mask.clamp(min=0, max=D_depth_sample - 1)
    # print(depth_sample.shape, depth_sample_pro_mask_claim.shape)
    depth_sample_ = torch.gather(depth_sample, 1, depth_sample_pro_mask_claim.squeeze(2))

    depth_sample_inter = depth_sample[:, 1:] - depth_sample[:, :-1]
    depth_sample_inter = torch.cat([depth_sample_inter, depth_sample_inter[:, -1:]], dim=1)  # [b,d,h,w]
    depth_sample_inter_index = torch.gather(depth_sample_inter, 1, depth_sample_pro_mask_claim.squeeze(2))

    depth_sample_dis = depth_sample_pro - depth_sample_

    # print(depth_sample_dis)
    depth_sample_rate = depth_sample_dis / (depth_sample_inter_index + 1e-10) / (
                D_depth_sample - 1) + depth_sample_pro_mask_claim.squeeze(2) / (D_depth_sample - 1)
    # print(depth_sample_pro_mask_claim,depth_sample_inter_index)
    # print(depth_sample_pro_mask_bool.squeeze(2))
    # print(depth_sample_rate.shape,depth_sample_pro_mask_bool.shape.squeeze(2))
    depth_sample_rate = torch.where(depth_sample_pro_mask_bool.squeeze(2), depth_sample_rate, -0.5)
    # print(depth_sample_rate.shape, depth_sample_rate)

    depth_sample_rate = depth_sample_rate * (D_depth_sample - 1)
    # b,d,h,w = depth_sample_pro .shape
    x0 = depth_sample_rate.permute(0, 2, 3, 1).reshape(b * H_depth_sample_pro * W_depth_sample_pro, 1,
                                                       D_depth_sample_pro, 1)
    y0 = torch.zeros_like(x0)

    disp_lvl = torch.cat([x0, y0], dim=-1)
    # print('1',pro.mean(), disp_lvl.mean())
    corr = bilinear_sampler(pro, disp_lvl)
    # print('2', corr.mean())
    # print()
    corr = corr.reshape(b, H_depth_sample_pro, W_depth_sample_pro, c, D_depth_sample_pro)
    corr = corr.permute(0, 3, 4, 1, 2)
    return corr


def depth_to_disp(depth, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    scaled_disp = 1 / depth

    min_disp = 1 / max_depth

    max_disp = 1 / min_depth

    disp = (scaled_disp - min_disp) / ((max_disp - min_disp) + 1e-10)

    return disp


def pro_bilinear_sampler2(pro, depth_sample, depth_min, depth_max):
    c, _, D = pro.shape[-3:]
    b, d, h, w = depth_sample.shape
    disp = depth_to_disp(depth_sample, depth_min, depth_max) * (D - 1)
    x0 = disp.permute(0, 2, 3, 1).reshape(b * h * w, 1, d, 1)
    y0 = torch.zeros_like(x0)
    disp_lvl = torch.cat([x0, y0], dim=-1)
    corr = bilinear_sampler(pro, disp_lvl)
    corr = corr.reshape(b, h, w, c, d)
    corr = corr.permute(0, 3, 4, 1, 2)

    return corr


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]

    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1

    assert torch.unique(ygrid).numel() == 1 and H == 1  # This is a stereo problem

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)
    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def pro_bilinear_sampler_new(pro, depth_sample, depth_sample_pro):
    c, _, D = pro.shape[-3:]
    # print(pro.shape)
    depth_sample = 1. / depth_sample
    depth_sample_pro = 1. / depth_sample_pro
    b, d, h, w = depth_sample.shape
    _, D_depth_sample, H_depth_sample, W_depth_sample = depth_sample.shape
    _, D_depth_sample_pro, H_depth_sample_pro, W_depth_sample_pro = depth_sample_pro.shape
    depth_sample_pro_mask_ = depth_sample_pro.unsqueeze(2).repeat(1, 1, D_depth_sample, 1, 1)
    depth_sample_mask = depth_sample.unsqueeze(1).repeat(1, D_depth_sample_pro, 1, 1, 1)
    depth_sample_pro_mask = (depth_sample_mask <= depth_sample_pro_mask_).to(torch.float32).sum(dim=2,
                                                                                                keepdims=True).to(
        torch.int64) - 1
    # print(depth_sample_mask, depth_sample_pro_mask_)
    # print((depth_sample_mask >= depth_sample_pro_mask_))
    depth_sample_pro_mask_bool = torch.logical_and(depth_sample_pro_mask >= 0,
                                                   depth_sample_pro_mask <= (D_depth_sample - 1))
    # depth_sample_pro_mask_bool_low = depth_sample_pro_mask >= 0
    depth_sample_pro_mask_bool_up = depth_sample_pro_mask < (D_depth_sample - 1)
    depth_sample_pro_mask_claim = depth_sample_pro_mask.clamp(min=0, max=D_depth_sample - 1)
    # print(depth_sample.shape, depth_sample_pro_mask_claim.shape)
    depth_sample_ = torch.gather(depth_sample, 1, depth_sample_pro_mask_claim.squeeze(2))

    depth_sample_inter = depth_sample[:, 1:] - depth_sample[:, :-1]
    depth_sample_inter = torch.cat([depth_sample_inter, depth_sample_inter[:, -1:]], dim=1)  # [b,d,h,w]
    depth_sample_inter_index = torch.gather(depth_sample_inter, 1, depth_sample_pro_mask_claim.squeeze(2))

    depth_sample_dis = depth_sample_pro - depth_sample_

    # print(depth_sample_dis)
    depth_sample_rate = depth_sample_dis / (depth_sample_inter_index + 1e-10) / (
                D_depth_sample - 1) + depth_sample_pro_mask_claim.squeeze(2) / (D_depth_sample - 1)

    depth_sample_rate = torch.where(depth_sample_pro_mask_bool.squeeze(2), depth_sample_rate, -0.5)


    depth_sample_rate = depth_sample_rate * (D_depth_sample - 1)
    # b,d,h,w = depth_sample_pro .shape
    x0 = depth_sample_rate.permute(0, 2, 3, 1).reshape(b * H_depth_sample_pro * W_depth_sample_pro, 1,
                                                       D_depth_sample_pro, 1)
    y0 = torch.zeros_like(x0)

    disp_lvl = torch.cat([x0, y0], dim=-1)
    # print('1',pro.mean(), disp_lvl.mean())
    corr = bilinear_sampler(pro, disp_lvl)
    # print('2', corr.mean())
    # print()
    corr = corr.reshape(b, H_depth_sample_pro, W_depth_sample_pro, c, D_depth_sample_pro)
    corr = corr.permute(0, 3, 4, 1, 2)
    return corr


class EC_MVSNet(nn.Module):
    def __init__(self, arch_mode="fpn", reg_net='reg2d', num_stage=4, fpn_base_channel=8,
                 reg_channel=8, stage_splits=[8, 8, 4, 4], depth_interals_ratio=[0.5, 0.5, 0.5, 0.5],
                 group_cor=False, group_cor_dim=[8, 8, 8, 8],
                 inverse_depth=False,
                 agg_type='ConvBnReLU3D',
                 attn_temp=2,
                 attn_fuse_d=True,
                 use_visi_net=False
                 ):
        super(EC_MVSNet, self).__init__()
        self.arch_mode = arch_mode
        self.num_stage = num_stage
        self.depth_interals_ratio = depth_interals_ratio
        self.group_cor = group_cor
        self.group_cor_dim = group_cor_dim
        self.inverse_depth = inverse_depth
        self.use_visi_net = use_visi_net
        self.agg_feature_type = "last"
        # self.using_gn = False
        self.using_gn = "IN"
        self.Dy_tmp = [0.1, 0.1, 0.01]
        self.using_pro = True
        self.mask_up_type = True
        self.mask_up_out = False
        self.pro_enhance = True
        self.feature_up_ = True
        self.cost_up_ = True

        print("agg_feature_type", self.agg_feature_type)
        print("using_gn", self.using_gn)
        print("use_visi_net", use_visi_net)
        print("attn_fuse_d", attn_fuse_d)
        print("feature_up", self.feature_up_)
        print("cost_up", self.cost_up_)
        print("pro_enhance", self.pro_enhance)
        print("mask_up_type", self.mask_up_type)
        print("mask_up_out", self.mask_up_out)
        print("using_pro", self.using_pro)


        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        # encoder = 'vitl' # or 'vits', 'vitb', 'vitg'
        encoder = 'vitb'
        print("pre_model", encoder)
        self.mono_pretrain_model = DepthAnythingV2(**model_configs[encoder])
        self.mono_pretrain_model.load_state_dict(torch.load(f'pretrain_checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        # self.mono_pretrain_model.eval()
        for param in self.mono_pretrain_model.parameters():
            param.requires_grad = False
        self.ViTFeaturePyramid = ViTFeaturePyramid_4stage(model_configs[encoder]['features']//2, fpn_base_channel*2, [0.5,1,2,4], gn=self.using_gn)


        self.stage_splits = stage_splits

        self.feature = FPN4_ET(in_channels=3, base_channels=fpn_base_channel, gn=self.using_gn)


        self.mono_depth_channel = [fpn_base_channel*4,fpn_base_channel*2, fpn_base_channel, fpn_base_channel//2]
        self.pre_depth_channel = self.stage_splits
        self.mask_up_channel = [32,32,16,8]
        self.ratio = [2,2,2,1]


        self.stagenet = stagenet_cs_mono(inverse_depth, attn_fuse_d, attn_temp, use_visi_net=use_visi_net,
                                         mask_up=self.mask_up_type, mask_up_out=self.mask_up_out,
                                         feature_up=self.feature_up_, cost_up=self.cost_up_)

        print("EC_MVS_MODEL")

        self.reg = nn.ModuleList()
        self.vis_net = [None, None, None, None]

        self.Depth_up = nn.ModuleList()

        self.cost_up = nn.ModuleList()
        self.feature_up = nn.ModuleList()

        self.mask_up = nn.ModuleList()

        if reg_net == 'reg3d':
            self.down_size = [3, 3, 2, 2]
        for idx in range(num_stage):
            if self.group_cor:
                in_dim = group_cor_dim[idx]
            else:
                in_dim = self.feature.out_channels[idx]

            if idx >= 1:
                in_dim_last = group_cor_dim[idx - 1]
            else:
                in_dim_last = in_dim


            self.mask_up.append(mask_up(mono_channel=self.mono_depth_channel[idx], base_channels=self.mask_up_channel[idx], ratio=self.ratio[idx], gn=self.using_gn))
            self.cost_up.append(cost_up(in_dim, reg_channel, gn=self.using_gn, using_pro=self.using_pro))
            self.Depth_up.append(None)
            self.feature_up.append(feature_up(in_dim, in_dim_last, reg_channel, gn=self.using_gn))


            if self.cost_up_:
                reg2d_in = reg_channel
            else:
                reg2d_in = in_dim

            self.reg.append(reg2d_large_mono_cs(input_channel=reg2d_in, base_channel=reg_channel, pre_depth_channel=self.pre_depth_channel[idx],
                                           mono_channel=self.mono_depth_channel[idx], pro_base_channels=self.pre_depth_channel[idx],
                                           conv_name=agg_type,gn=self.using_gn,pro_enhance=self.pro_enhance))



    def forward(self, imgs, ref_img, proj_matrices, depth_values):
        depth_min_ = depth_values[:, 0]
        depth_max_ = depth_values[:, -1]
        depth_min = depth_values[:, 0].cpu().numpy()
        depth_max = depth_values[:, -1].cpu().numpy()




        ref_outputs, src_outputs = self.feature(imgs)
        mono_feature, _ = self.mono_pretrain_model.infer_image(ref_img, imgs[0].shape[2], imgs[0].shape[3])
        mono_feature = F.interpolate(mono_feature,size=[src_outputs[0]["stage2"].shape[2],src_outputs[0]["stage2"].shape[3]], mode='bilinear')
        mono_depth_feature = self.ViTFeaturePyramid(mono_feature)


        # step 2. iter (multi-scale)
        outputs = {}
        for stage_idx in range(self.num_stage):
            if self.mask_up_out and (stage_idx<3):
                mono_depth_features_stage_up = mono_depth_feature["stage{}".format(stage_idx + 2)]
            else:
                mono_depth_features_stage_up = mono_depth_feature["stage{}".format(stage_idx + 1)]
            mono_depth_features_stage = mono_depth_feature["stage{}".format(stage_idx + 1)]


            ref_features_stage = ref_outputs["stage{}".format(stage_idx + 1)]
            if stage_idx >= 1:
                ref_features_stage_last = ref_outputs["stage{}".format(stage_idx)]
            else:
                ref_features_stage_last = None

            src_features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in src_outputs]
            if stage_idx >= 1:
                src_features_stage_last = [feat["stage{}".format(stage_idx)] for feat in src_outputs]
            else:
                src_features_stage_last = None
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            if stage_idx >= 1:
                proj_matrices_stage_last = proj_matrices["stage{}".format(stage_idx)]
            else:
                proj_matrices_stage_last = None
            B, C, H, W = src_features_stage[0].shape

            # init range
            if stage_idx == 0:
                if self.inverse_depth:
                    depth_hypo = init_inverse_range(depth_values, self.stage_splits[stage_idx], imgs[0][0].device,
                                                    imgs[0][0].dtype, H, W)
                else:
                    depth_hypo = init_range(depth_values, self.stage_splits[stage_idx], imgs[0][0].device,
                                            imgs[0][0].dtype, H, W)
                agg_pro = None
                depth_range_samples_low = None
            else:
                if self.inverse_depth:
                    if self.mask_up_type:
                        depth_hypo = schedule_inverse_range_maskup(outputs_stage['inverse_min_depth'].detach(),
                                                        outputs_stage['inverse_max_depth'].detach(),
                                                        self.stage_splits[stage_idx], H, W)  # B D H W
                    else:
                        depth_hypo = schedule_inverse_range(outputs_stage['inverse_min_depth'].detach(),
                                                                   outputs_stage['inverse_max_depth'].detach(),
                                                                   self.stage_splits[stage_idx], H, W)  # B D H W
                else:
                    depth_interval = (depth_max - depth_min) / 192
                    depth_hypo = schedule_range(outputs_stage['depth'].detach(), self.stage_splits[stage_idx],
                                                self.depth_interals_ratio[stage_idx] * depth_interval, H, W)

                last_stage_depth_value = outputs_stage['hypo_depth']
                last_depth_max = last_stage_depth_value[:, 0:1]
                last_depth_min = last_stage_depth_value[:, -1:]

                B_, D_, H_, W_ = depth_hypo.shape
                depth_range_samples_low = F.interpolate(depth_hypo.unsqueeze(1),
                                                        size=[D_, H_ // 2, W_ // 2], mode='nearest').squeeze(1)
                _, C, _, _, _ = agg_pro.shape
                agg_pro = agg_pro.permute(0, 3, 4, 1, 2).reshape(B * (H_ // 2) * (W_ // 2), C, 1,
                                                                             self.stage_splits[stage_idx - 1])
                agg_pro = agg_pro.to(torch.float32)


                agg_pro = pro_bilinear_sampler2(agg_pro, depth_range_samples_low, last_depth_min,
                                                          last_depth_max)

            group_cor_dim_cur = self.group_cor_dim[stage_idx]

            if stage_idx >= 1:
                group_cor_dim_last = self.group_cor_dim[stage_idx - 1]
            else:
                group_cor_dim_last = group_cor_dim_cur

            outputs_stage = self.stagenet(ref_features_stage, ref_features_stage_last, src_features_stage,
                                          src_features_stage_last, proj_matrices_stage, proj_matrices_stage_last,
                                          depth_hypo=depth_hypo, depth_hypo_last=depth_range_samples_low,
                                          regnet=self.reg[stage_idx], stage_idx=stage_idx,
                                          group_cor=self.group_cor, group_cor_dim=group_cor_dim_cur,
                                          group_cor_dim_last=group_cor_dim_last,
                                          split_itv=self.depth_interals_ratio[stage_idx],
                                          vis_net=self.vis_net[stage_idx], stageid=stage_idx,
                                          agg_pro=agg_pro, cost_up=self.cost_up[stage_idx],
                                          feature_up=self.feature_up[stage_idx], depth_up=self.Depth_up[stage_idx],
                                          mono_depth_features_stage=mono_depth_features_stage, mono_depth_features_stage_up=mono_depth_features_stage_up,
                                          mask_up=self.mask_up[stage_idx],depth_min_=depth_min_, depth_max_=depth_max_)
            if self.using_pro:
                agg_pro = outputs_stage["prob_volume_pre"].unsqueeze(1)
            else:
                agg_pro = outputs_stage["agg_pro"]

            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)

        return outputs


def cross_entropy_loss(mask_true, hypo_depth, depth_gt, attn_weight):
    B, D, H, W = attn_weight.shape
    valid_pixel_num = torch.sum(mask_true, dim=[1, 2]) + 1e-6
    gt_index_image = torch.argmin(torch.abs(hypo_depth - depth_gt.unsqueeze(1)), dim=1)
    gt_index_image = torch.mul(mask_true, gt_index_image.type(torch.float))
    gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1)  # B, 1, H, W
    gt_index_volume = torch.zeros(B, D, H, W).type(mask_true.type()).scatter_(1, gt_index_image, 1)
    cross_entropy_image = -torch.sum(gt_index_volume * torch.log(attn_weight + 1e-6), dim=1).squeeze(1)  # B, 1, H, W
    masked_cross_entropy_image = torch.mul(mask_true, cross_entropy_image)
    masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])
    masked_cross_entropy = torch.mean(masked_cross_entropy / valid_pixel_num)

    return masked_cross_entropy


def MVS4net_loss_mono(inputs, depth_gt_ms, mask_ms, **kwargs):
    stage_lw = kwargs.get("stage_lw", [1, 1, 1, 1])
    inverse = kwargs.get("inverse_depth", False)
    depth_fuse = kwargs.get("depth_fuse", False)
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    stage_ce_loss = []
    range_err_ratio = []
    depth_fuse_loss = []
    stage_mask_up_loss = []
    for stage_idx, (stage_inputs, stage_key) in enumerate([(inputs[k], k) for k in inputs.keys() if "stage" in k]):
        depth_pred = stage_inputs['depth']
        hypo_depth = stage_inputs['hypo_depth']
        attn_weight = stage_inputs['attn_weight']
        mask = mask_ms[stage_key]
        mask = mask > 0.5
        depth_gt = depth_gt_ms[stage_key]



        # mask range
        if inverse:
            depth_itv = (1 / hypo_depth[:, 2, :, :] - 1 / hypo_depth[:, 1, :, :]).abs()  # B H W
            mask_out_of_range = ((1 / hypo_depth - 1 / depth_gt.unsqueeze(1)).abs() <= depth_itv.unsqueeze(1)).sum(
                1) == 0  # B H W
        else:
            depth_itv = (hypo_depth[:, 2, :, :] - hypo_depth[:, 1, :, :]).abs()  # B H W
            mask_out_of_range = ((hypo_depth - depth_gt.unsqueeze(1)).abs() <= depth_itv.unsqueeze(1)).sum(
                1) == 0  # B H W
        range_err_ratio.append(mask_out_of_range[mask].float().mean())

        # cross-entropy
        this_stage_ce_loss = cross_entropy_loss(mask, hypo_depth, depth_gt, attn_weight)



        stage_ce_loss.append(this_stage_ce_loss)
        total_loss = total_loss + this_stage_ce_loss



        if "maskup_depth" in stage_inputs.keys():
            maskup_depth = stage_inputs['maskup_depth']
            if stage_idx<3:
                depth_gt_maskup_depth = depth_gt_ms["stage{}".format(stage_idx+2)]
                mask_maskup_depth = mask_ms["stage{}".format(stage_idx+2)]
                mask_maskup_depth = mask_maskup_depth>0.5
            else:
                depth_gt_maskup_depth = depth_gt
                mask_maskup_depth = mask


            mask_maskup_depth_loss = F.smooth_l1_loss(maskup_depth[mask_maskup_depth], depth_gt_maskup_depth[mask_maskup_depth], reduction='mean')
            stage_mask_up_loss.append(mask_maskup_depth_loss)
            total_loss = total_loss + mask_maskup_depth_loss



        if depth_fuse and stage_key >= 1:
            fuse_loss = F.smooth_l1_loss(depth_pred[mask], depth_gt[mask], reduction='mean')
            depth_fuse_loss.append(fuse_loss)
            total_loss = total_loss + fuse_loss

    return total_loss, stage_ce_loss, range_err_ratio, stage_mask_up_loss, depth_fuse_loss


def Blend_loss_mono(inputs, depth_gt_ms, mask_ms, **kwargs):
    stage_lw = kwargs.get("stage_lw", [1,1,1,1])

    inverse = kwargs.get("inverse_depth", False)

    depth_fuse = kwargs.get("depth_fuse", False)

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    stage_ce_loss = []
    range_err_ratio = []

    depth_fuse_loss = []
    stage_mask_up_loss = []
    for stage_idx, (stage_inputs, stage_key) in enumerate([(inputs[k], k) for k in inputs.keys() if "stage" in k]):
        depth_pred = stage_inputs['depth']
        hypo_depth = stage_inputs['hypo_depth']
        attn_weight = stage_inputs['attn_weight']
        mask = mask_ms[stage_key]
        mask = mask > 0.5
        depth_gt = depth_gt_ms[stage_key]

        # # mask range
        if inverse:
            depth_itv = (1 / hypo_depth[:, 2, :, :] - 1 / hypo_depth[:, 1, :, :]).abs()  # B H W
            mask_out_of_range = ((1 / hypo_depth - 1 / depth_gt.unsqueeze(1)).abs() <= depth_itv.unsqueeze(1)).sum(
                1) == 0  # B H W
        else:
            depth_itv = (hypo_depth[:, 2, :, :] - hypo_depth[:, 1, :, :]).abs()  # B H W
            mask_out_of_range = ((hypo_depth - depth_gt.unsqueeze(1)).abs() <= depth_itv.unsqueeze(1)).sum(
                1) == 0  # B H W
        range_err_ratio.append(mask_out_of_range[mask].float().mean())

        # cross-entropy
        this_stage_ce_loss = cross_entropy_loss(mask, hypo_depth, depth_gt, attn_weight)
        #
        stage_ce_loss.append(this_stage_ce_loss)
        total_loss = total_loss + this_stage_ce_loss


        if "maskup_depth" in stage_inputs.keys():
            maskup_depth = stage_inputs['maskup_depth']
            if stage_idx<3:
                depth_gt_maskup_depth = depth_gt_ms["stage{}".format(stage_idx+2)]
                mask_maskup_depth = mask_ms["stage{}".format(stage_idx+2)]
                mask_maskup_depth = mask_maskup_depth>0.5
            else:
                depth_gt_maskup_depth = depth_gt
                mask_maskup_depth = mask


            mask_maskup_depth_loss = F.smooth_l1_loss(maskup_depth[mask_maskup_depth], depth_gt_maskup_depth[mask_maskup_depth], reduction='mean')
            stage_mask_up_loss.append(mask_maskup_depth_loss)
            total_loss = total_loss + mask_maskup_depth_loss


    depth_interval = hypo_depth[:, 0, :, :] - hypo_depth[:, 1, :, :]

    abs_err = torch.abs(depth_gt[mask] - depth_pred[mask])
    abs_err_scaled = abs_err / (depth_interval[mask] * 192. / 128.)
    epe = abs_err_scaled.mean()
    err3 = (abs_err_scaled <= 3).float().mean()
    err1 = (abs_err_scaled <= 1).float().mean()
    return total_loss, stage_ce_loss, range_err_ratio, epe, err3, err1, stage_mask_up_loss
