import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import math
import numpy as np

# from functools import partial




def homo_warping(src_fea, src_proj, ref_proj, depth_values):

    C = src_fea.shape[1]
    Hs,Ws = src_fea.shape[-2:]
    B,num_depth,Hr,Wr = depth_values.shape

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, Hr, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, Wr, dtype=torch.float32, device=src_fea.device)])
        y = y.reshape(Hr*Wr)
        x = x.reshape(Hr*Wr)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(B, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.reshape(B, 1, num_depth, -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.reshape(B, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        # FIXME divide 0
        temp = proj_xyz[:, 2:3, :, :]
        temp[temp==0] = 1e-9
        proj_xy = proj_xyz[:, :2, :, :] / temp  # [B, 2, Ndepth, H*W]
        # proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]

        proj_x_normalized = proj_xy[:, 0, :, :] / ((Ws - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((Hs - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy
    if len(src_fea.shape)==4:
        warped_src_fea = F.grid_sample(src_fea, grid.reshape(B, num_depth * Hr, Wr, 2), mode='bilinear', padding_mode='zeros', align_corners=True)
        warped_src_fea = warped_src_fea.reshape(B, C, num_depth, Hr, Wr)
    elif len(src_fea.shape)==5:
        warped_src_fea = []
        for d in range(src_fea.shape[2]):
            warped_src_fea.append(F.grid_sample(src_fea[:,:,d], grid.reshape(B, num_depth, Hr, Wr, 2)[:,d], mode='bilinear', padding_mode='zeros', align_corners=True))
        warped_src_fea = torch.stack(warped_src_fea, dim=2)

    return warped_src_fea

def init_range(cur_depth, ndepths, device, dtype, H, W):
    cur_depth_min = cur_depth[:, 0]  # (B,)
    cur_depth_max = cur_depth[:, -1]
    new_interval = (cur_depth_max - cur_depth_min) / (ndepths - 1)  # (B, )
    new_interval = new_interval[:, None, None]  # B H W
    depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepths, device=device, dtype=dtype,
                                                                requires_grad=False).reshape(1, -1) * new_interval.squeeze(1)) #(B, D)
    depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W) #(B, D, H, W)
    return depth_range_samples

def init_inverse_range(cur_depth, ndepths, device, dtype, H, W):
    inverse_depth_min = 1. / cur_depth[:, 0]  # (B,)
    inverse_depth_max = 1. / cur_depth[:, -1]
    itv = torch.arange(0, ndepths, device=device, dtype=dtype, requires_grad=False).reshape(1, -1,1,1).repeat(1, 1, H, W)  / (ndepths - 1)  # 1 D H W
    inverse_depth_hypo = inverse_depth_max[:,None, None, None] + (inverse_depth_min - inverse_depth_max)[:,None, None, None] * itv

    return 1./inverse_depth_hypo

def schedule_inverse_range(inverse_min_depth, inverse_max_depth, ndepths, H, W):
    #cur_depth_min, (B, H, W)
    #cur_depth_max: (B, H, W)
    itv = torch.arange(0, ndepths, device=inverse_min_depth.device, dtype=inverse_min_depth.dtype, requires_grad=False).reshape(1, -1,1,1).repeat(1, 1, H//2, W//2)  / (ndepths - 1)  # 1 D H W
    # print(inverse_max_depth.shape, inverse_min_depth.shape, itv.shape)
    inverse_depth_hypo = inverse_max_depth[:,None, :, :] + (inverse_min_depth - inverse_max_depth)[:,None, :, :] * itv  # B D H W
    inverse_depth_hypo = F.interpolate(inverse_depth_hypo.unsqueeze(1), [ndepths, H, W], mode='trilinear', align_corners=True).squeeze(1)
    return 1./inverse_depth_hypo

def schedule_inverse_range_maskup(inverse_min_depth, inverse_max_depth, ndepths, H, W):
    #cur_depth_min, (B, H, W)
    #cur_depth_max: (B, H, W)
    itv = torch.arange(0, ndepths, device=inverse_min_depth.device, dtype=inverse_min_depth.dtype, requires_grad=False).reshape(1, -1,1,1).repeat(1, 1, H, W)  / (ndepths - 1)  # 1 D H W

    inverse_depth_hypo = inverse_max_depth[:,None, :, :] + (inverse_min_depth - inverse_max_depth)[:,None, :, :] * itv  # B D H W
    # inverse_depth_hypo = F.interpolate(inverse_depth_hypo.unsqueeze(1), [ndepths, H, W], mode='trilinear', align_corners=True).squeeze(1)
    return 1./inverse_depth_hypo

def schedule_range(cur_depth, ndepth, depth_inteval_pixel, H, W):
    #shape, (B, H, W)
    #cur_depth: (B, H, W)
    #return depth_range_values: (B, D, H, W)
    cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel[:,None,None])  # (B, H, W)
    cur_depth_max = (cur_depth + ndepth / 2 * depth_inteval_pixel[:,None,None])
    new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, H, W)

    depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=cur_depth.device, dtype=cur_depth.dtype,
                                                                  requires_grad=False).reshape(1, -1, 1, 1) * new_interval.unsqueeze(1))
    depth_range_samples = F.interpolate(depth_range_samples.unsqueeze(1), [ndepth, H, W], mode='trilinear', align_corners=True).squeeze(1)
    return depth_range_samples

def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return

def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return

class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, gn=False, group_channel=4):
        super(ConvBnReLU3D, self).__init__()
        if gn == 'IN':
            bn = 'IN'
        else:
            bn = not gn
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        if bn == 'IN':
            self.bn = nn.InstanceNorm3d(out_channels)
        elif bn:
            # print("ConvBnReLU3D bn")
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            # print("ConvBnReLU3D gn")
            self.bn = nn.GroupNorm(int(max(1, out_channels / group_channel)), out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

class ConvBnReLU3D_CAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D_CAM, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.linear_agg = nn.Sequential(
            nn.Linear(out_channels, out_channels//2),
            nn.ReLU(),
            nn.Linear(out_channels//2, out_channels)
        )

    def forward(self, input):
        x = self.conv(input)
        B,C,D,H,W = x.shape
        avg_attn = self.linear_agg(x.reshape(B,C,D*H*W).mean(2))
        max_attn = self.linear_agg(x.reshape(B,C,D*H*W).max(2)[0])  # B C
        attn = F.sigmoid(max_attn+avg_attn)[:,:,None,None,None]  # B C,1,1,1
        x = x * attn
        return F.relu(self.bn(x+input), inplace=True)

class ConvBnReLU3D_DCAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D_DCAM, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.linear_agg = nn.Sequential(
            nn.Linear(out_channels, out_channels//2),
            nn.ReLU(),
            nn.Linear(out_channels//2, out_channels)
        )

    def forward(self, input):
        x = self.conv(input)
        B,C,D,H,W = x.shape
        avg_attn = self.linear_agg(x.reshape(B,C,D,H*W).mean(3).permute(0,2,1).reshape(B*D,C)).reshape(B,D,C).permute(0,2,1)
        max_attn = self.linear_agg(x.reshape(B,C,D,H*W).max(3)[0].permute(0,2,1).reshape(B*D,C)).reshape(B,D,C).permute(0,2,1)  # B C D
        attn = F.sigmoid(max_attn+avg_attn)[:,:,:,None,None]  # B C,D,1,1
        x = x * attn
        return F.relu(self.bn(x+input), inplace=True)

class ConvBnReLU3D_PAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D_PAM, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.pixel_conv = nn.Conv2d(2,1,7,stride=1,padding='same')

    def forward(self, input):
        x = self.conv(input)
        B,C,D,H,W = x.shape
        max_attn = x.reshape(B,C*D,H,W).max(1, keepdim=True)[0]
        avg_attn = x.reshape(B,C*D,H,W).mean(1, keepdim=True)  # B 1 H W
        attn = F.sigmoid(self.pixel_conv(torch.cat([max_attn, avg_attn], dim=1)))[:,:,None,:,:]  # B 1,1,H,W
        x = x * attn
        return F.relu(self.bn(x+input), inplace=True)

class ConvBnReLU3D_PDAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D_PDAM, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.spatial_conv = nn.Conv3d(2,1,7,stride=1,padding='same')

    def forward(self, input):
        x = self.conv(input)
        B,C,D,H,W = x.shape
        max_attn = x.max(1, keepdim=True)[0]
        avg_attn = x.mean(1, keepdim=True)  # B 1 D H W
        attn = F.sigmoid(self.spatial_conv(torch.cat([max_attn, avg_attn], dim=1)))  # B 1,D,H,W
        x = x * attn
        return F.relu(self.bn(x+input), inplace=True)

class Deconv3d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv3d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn_momentum=0.1, init_method="xavier", gn=False, group_channel=4, **kwargs):
        super(Conv2d, self).__init__()
        if gn == 'IN':
            bn = 'IN'
        else:
            bn = not gn
        # bn = not gn
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        # self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        if bn == 'IN':
            self.bn = nn.InstanceNorm2d(out_channels, momentum=bn_momentum)
        elif bn:
            # print("con2d bn")
            self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        else:
            # print("con2d gn")
            self.bn = None
        self.gn = nn.GroupNorm(int(max(1, out_channels / group_channel)), out_channels) if gn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        else:
            x = self.gn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class Deconv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

class FPN4_ET(nn.Module):
    """
    FPN aligncorners downsample 4x"""

    def __init__(self, in_channels=3, base_channels=8, gn=False):
        super(FPN4_ET, self).__init__()
        self.base_channels = base_channels

        self.conv0 = nn.Sequential(
            Conv2d(in_channels, base_channels, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels, base_channels, 3, 1, padding=1, gn=gn),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2, gn=gn),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1, gn=gn),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2, gn=gn),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1, gn=gn),
        )

        self.conv3 = nn.Sequential(
            Conv2d(base_channels * 4, base_channels * 8, 5, stride=2, padding=2, gn=gn),
            Conv2d(base_channels * 8, base_channels * 8, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels * 8, base_channels * 8, 3, 1, padding=1, gn=gn),
        )

        self.out_channels = [8 * base_channels]
        final_chs = base_channels * 8


        self.inner1 = nn.Conv2d(base_channels * 4, final_chs, 1, bias=True)
        self.inner2 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
        self.inner3 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

        self.out1 = nn.Conv2d(final_chs, base_channels * 8, 1, bias=False)
        self.out2 = nn.Conv2d(final_chs, base_channels * 4, 3, padding=1, bias=False)
        self.out3 = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
        self.out4 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)

        self.la = nn.Conv2d(final_chs, final_chs, 3, bias=False, padding=1)
        self.res = nn.Conv2d(final_chs, final_chs, 1, bias=False)

        self.out_channels.append(base_channels * 4)
        self.out_channels.append(base_channels * 2)
        self.out_channels.append(base_channels)

    def forward(self, imgs):
        if len(imgs) > 1:
            ref_img, src_imgs = imgs[0], imgs[1:]
        else:
            ref_img = imgs[0]
        device = ref_img.device

        ref_outputs, src_outputs = [], []

        ref_conv0 = self.conv0(ref_img)
        ref_conv1 = self.conv1(ref_conv0)
        ref_conv2 = self.conv2(ref_conv1)
        ref_conv3 = self.conv3(ref_conv2)
        # print(ref_conv3.shape)
        B, C, H, W = ref_conv3.shape

        ref_intra = ref_conv3.clone()
        intra = ref_conv3
        ref_out1 = self.out1(intra)
        intra = F.interpolate(intra, scale_factor=2, mode="bilinear", align_corners=True) + self.inner1(ref_conv2)
        ref_out2 = self.out2(intra)
        intra = F.interpolate(intra, scale_factor=2, mode="bilinear", align_corners=True) + self.inner2(ref_conv1)
        ref_out3 = self.out3(intra)
        intra = F.interpolate(intra, scale_factor=2, mode="bilinear", align_corners=True) + self.inner3(ref_conv0)
        ref_out4 = self.out4(intra)

        ref_outputs = {}
        ref_outputs["stage1"] = ref_out1
        ref_outputs["stage2"] = ref_out2
        ref_outputs["stage3"] = ref_out3
        ref_outputs["stage4"] = ref_out4

        if len(imgs) == 1:
            return ref_outputs
        # pos = torch.ones((1, H, W))
        # pos = self.pos_enc(pos).squeeze(0).to(device)  # CHW
        #
        if len(imgs) > 1:
            for src_idx, src_img in enumerate(src_imgs):
                src_out = {}
                src_conv0 = self.conv0(src_img)
                src_conv1 = self.conv1(src_conv0)
                src_conv2 = self.conv2(src_conv1)
                src_conv3 = self.conv3(src_conv2)
                src_intra = src_conv3.clone()


                src_out1 = self.out1(src_intra)
                src_intra = F.interpolate(src_intra, scale_factor=2, mode="bilinear", align_corners=True) + self.inner1(
                    src_conv2)
                src_out2 = self.out2(src_intra)
                src_intra = F.interpolate(src_intra, scale_factor=2, mode="bilinear", align_corners=True) + self.inner2(
                    src_conv1)
                src_out3 = self.out3(src_intra)
                src_intra = F.interpolate(src_intra, scale_factor=2, mode="bilinear", align_corners=True) + self.inner3(
                    src_conv0)
                src_out4 = self.out4(src_intra)

                src_out["stage1"] = src_out1
                src_out["stage2"] = src_out2
                src_out["stage3"] = src_out3
                src_out["stage4"] = src_out4
                src_outputs.append(src_out)

            return ref_outputs, src_outputs

class stagenet_cs_mono(nn.Module):
    def __init__(self, inverse_depth=False, attn_fuse_d=True, attn_temp=2, use_visi_net=False,
                 mask_up=True, mask_up_out=False,feature_up=True, cost_up=True):
        super(stagenet_cs_mono, self).__init__()
        self.inverse_depth = inverse_depth
        self.use_visi_net = use_visi_net
        self.attn_fuse_d = attn_fuse_d
        self.attn_temp = attn_temp
        self.mask_up_type = mask_up
        self.mask_up_out = mask_up_out
        self.feature_up = feature_up
        self.cost_up = cost_up
        self.tmp = [10.0, 10.0, 10.0, 1.0]


    def get_cor_feat(self, ref_feature, src_features, proj_matrices, depth_hypo, vis_net=None,
                group_cor=False, group_cor_dim=8, split_itv=1, stageid=1, agg_pro=None, cost_up=None, OneFeature=False):
        proj_matrices = torch.unbind(proj_matrices, 1)
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]
        B, D, H, W = depth_hypo.shape
        C = ref_feature.shape[1]

        cor_weight_sum = 1e-8
        cor_feats = 0
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, D, 1, 1)
        ref_proj_new = ref_proj[:, 0].clone()
        ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
        cor_weights = []
        # step 2. Epipolar Transformer Aggregation
        for src_idx, (src_fea, src_proj) in enumerate(zip(src_features, src_projs)):
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])

            warped_src = homo_warping_One_Featrue(src_fea, src_proj_new, ref_proj_new, depth_hypo)  # B C D H W

            if group_cor:
                # print(ref_volume.shape)
                warped_src = warped_src.reshape(B, group_cor_dim, C // group_cor_dim, D, H, W)
                ref_volume = ref_volume.reshape(B, group_cor_dim, C // group_cor_dim, D, H, W)
                cor_feat = (warped_src * ref_volume).mean(2)  # B G D H W
            else:
                cor_feat = (ref_volume - warped_src) ** 2  # B C D H W
            del warped_src, src_proj, src_fea
            if self.use_visi_net:
                # print("test")
                cor_feat_fuse = cor_feat.reshape(B, group_cor_dim * D, H, W)
                cor_weight = vis_net(ref_feature, cor_feat_fuse)  # B H W
                cor_weights.append(cor_weight)
                cor_weight_sum += cor_weight  # B H W
                cor_feats += cor_weight.unsqueeze(1).unsqueeze(1) * cor_feat  # B C D H W
                del cor_feat
            else:
                if not self.attn_fuse_d:
                    cor_weight = torch.softmax(cor_feat.sum(1), 1).max(1)[0]  # B H W
                    cor_weight_sum += cor_weight  # B H W
                    cor_weights.append(cor_weight)
                    cor_feats += cor_weight.unsqueeze(1).unsqueeze(1) * cor_feat  # B C D H W
                else:
                    cor_weight = torch.softmax(cor_feat.sum(1) / self.attn_temp, 1) / math.sqrt(C)  # B D H W
                    cor_weight_sum += cor_weight  # B D H W
                    cor_weight_save = cor_weight.max(1)[0]
                    cor_weights.append(cor_weight_save)
                    cor_feats += cor_weight.unsqueeze(1) * cor_feat  # B C D H W
                del cor_weight, cor_feat
        if self.use_visi_net:
            cor_feats = cor_feats / cor_weight_sum.unsqueeze(1).unsqueeze(1)  # B C D H W
        else:
            if not self.attn_fuse_d:
                cor_feats = cor_feats / cor_weight_sum.unsqueeze(1).unsqueeze(1)  # B C D H W
            else:
                cor_feats = cor_feats / cor_weight_sum.unsqueeze(1)  # B C D H W

        del cor_weight_sum, src_features

        return cor_weights, cor_feats
    def forward(self, ref_feature, ref_feature_last, src_features, src_features_last, proj_matrices, proj_matrices_last, depth_hypo, depth_hypo_last, regnet, stage_idx, vis_net=None,
                group_cor=False, group_cor_dim=8, group_cor_dim_last=8, split_itv=1, stageid=1, agg_pro=None, cost_up=None, feature_up=None,OneFeature=False,depth_up=None,
                mono_depth_features_stage=None, mono_depth_features_stage_up=None, mask_up=None, depth_min_=425, depth_max_=935):

        # step 1. feature extraction
        cor_weights, cor_feats = self.get_cor_feat(ref_feature, src_features, proj_matrices, depth_hypo, vis_net=vis_net,
                group_cor=group_cor, group_cor_dim=group_cor_dim, split_itv=split_itv,
                stageid=stageid, agg_pro=agg_pro, cost_up=cost_up, OneFeature=OneFeature)

        if self.feature_up:
            if stageid >=1:
                _,_,H,W = ref_feature.shape

                ref_feature_last = F.interpolate(ref_feature_last, size=[H, W], mode='bilinear')

                cor_weights_last, cor_feats_last = self.get_cor_feat(ref_feature_last, src_features_last, proj_matrices, depth_hypo, vis_net=vis_net,
                    group_cor=group_cor, group_cor_dim=group_cor_dim_last, split_itv=split_itv,
                    stageid=stageid, agg_pro=agg_pro, cost_up=cost_up, OneFeature=OneFeature)

                _, cor_feats = feature_up(cor_feats, cor_feats_last)

        if stageid>=1 and self.cost_up:
            _, volume_mean_fin = cost_up(cor_feats, agg_pro)
            # print(stageid, volume_mean_fin.shape,ref_feature)
        else:
            volume_mean_fin = cor_feats
            # print(stageid, volume_mean_fin.shape)


        attn_weight, agg_pro = regnet(volume_mean_fin, mono_depth_features_stage)  # B D H W

        attn_weight_ = attn_weight.clone()
        del cor_feats
        attn_weight = F.softmax(attn_weight, dim=1)  # B D H W

        # step 4. depth argmax
        attn_max_indices = attn_weight.max(1, keepdim=True)[1]  # B 1 H W

        if self.training:
            depth = torch.gather(depth_hypo, 1, attn_max_indices).squeeze(1)  # B H W
        else:
            depth = depth_regression(F.softmax(attn_weight_ * self.tmp[stageid], dim=1), depth_values=depth_hypo)

        #depth maskup
        if self.mask_up_type:
            if self.mask_up_out:
                depth_ = (depth - depth_min_) / (depth_max_ - depth_min_)
                maskup_depth = mask_up(depth_, mono_depth_features_stage_up)
                B,H,W = depth_.shape
                if stageid < 3:
                    depth_ = F.interpolate(depth_.unsqueeze(1), size=[H*2, W*2], mode='bilinear')
                    depth_ = depth_.squeeze(1)
                maskup_depth = depth_ + maskup_depth
                maskup_depth = maskup_depth.clamp(min=1e-8)
                maskup_depth = maskup_depth * (depth_max_ - depth_min_) + depth_min_
            else:
                # print(depth.shape, depth_min_.shape)
                depth_ = (depth-depth_min_[:,None, None])/(depth_max_[:,None, None]-depth_min_[:,None, None])
                mask = mask_up(depth_, mono_depth_features_stage)
                if stageid < 3:
                    ratio = 2
                else:
                    ratio = 1
                maskup_depth = upsample_depth(depth.unsqueeze(1), mask, ratio)

                maskup_depth = maskup_depth.clamp(min=1e-8)

        else:
            maskup_depth = depth



        if not self.training:
            with torch.no_grad():
                photometric_confidence = attn_weight.max(1)[0]  # B H W
                photometric_confidence = F.interpolate(photometric_confidence.unsqueeze(1),
                                                       scale_factor=2 ** (3 - stage_idx), mode='bilinear',
                                                       align_corners=True).squeeze(1)
        else:
            photometric_confidence = torch.tensor(0.0, dtype=torch.float32, device=ref_feature.device,
                                                  requires_grad=False)

        ret_dict = {"depth": depth, "maskup_depth": maskup_depth,"photometric_confidence": photometric_confidence, "hypo_depth": depth_hypo, "prob_volume_pre":attn_weight_,
                    "attn_weight": attn_weight, "cor_weights": cor_weights, "cor_weights_": cor_weights, "agg_pro":agg_pro}

        if stageid < 3:
            if self.inverse_depth:
                B,H,W = depth.shape
                if self.mask_up_type:
                    # print((depth_hypo[:, 2, :, :]-depth_hypo[:, 1, :, :]).mean())
                    last_depth_itv = 1. / depth_hypo[:, 2, :, :] - 1. / depth_hypo[:, 1, :, :]  ##>0
                    # print(last_depth_itv.shape)
                    last_depth_itv_up = F.interpolate(last_depth_itv.unsqueeze(1), [H*2, W*2], mode='bilinear',align_corners=True).squeeze(1)
                    inverse_min_depth = 1 / maskup_depth + split_itv * last_depth_itv_up  # B H W
                    inverse_min_depth = inverse_min_depth.clamp(min=1e-8)
                    inverse_max_depth = 1 / maskup_depth - split_itv * last_depth_itv_up  # B H W
                    inverse_max_depth = inverse_max_depth.clamp(min=1e-8)
                else:
                    # print(depth.shape)
                    last_depth_itv = 1. / depth_hypo[:, 2, :, :] - 1. / depth_hypo[:, 1, :, :]
                    inverse_min_depth = 1 / depth + split_itv * last_depth_itv  # B H W
                    inverse_min_depth = inverse_min_depth.clamp(min=1e-8)
                    inverse_max_depth = 1 / depth - split_itv * last_depth_itv  # B H W
                    inverse_max_depth = inverse_max_depth.clamp(min=1e-8)
                ret_dict['inverse_min_depth'] = inverse_min_depth
                ret_dict['inverse_max_depth'] = inverse_max_depth

        return ret_dict

class reg2d_large_mono_cs(nn.Module):
    def __init__(self, input_channel=128, base_channel=32, pre_depth_channel=32, mono_channel=32, pro_base_channels=32, conv_name='ConvBnReLU3D', gn=False, pro_enhance=True, depth_hypo_up=False):
        super(reg2d_large_mono_cs, self).__init__()
        module = importlib.import_module("models.module")
        stride_conv_name = 'ConvBnReLU3D'
        self.conv0 = getattr(module, stride_conv_name)(input_channel, base_channel, kernel_size=(1,5,5), pad=(0,2,2), gn=gn)
        self.conv0_ = getattr(module, conv_name)(base_channel, base_channel, gn=gn)
        # self.conv0_ = getattr(module, conv_name)(base_channel, base_channel, gn=gn)

        self.conv1 = getattr(module, stride_conv_name)(base_channel, base_channel*2, kernel_size=(1,5,5), stride=(1,2,2), pad=(0,2,2), gn=gn)
        self.conv2 = getattr(module, conv_name)(base_channel*2, base_channel*2, gn=gn)
        self.conv2_ = getattr(module, conv_name)(base_channel * 2, base_channel * 2, gn=gn)

        self.conv3 = getattr(module, stride_conv_name)(base_channel*2, base_channel*4, kernel_size=(1,3,3), stride=(1,2,2), pad=(0,1,1), gn=gn)
        self.conv4 = getattr(module, conv_name)(base_channel*4, base_channel*4, gn=gn)
        self.conv4_ = getattr(module, conv_name)(base_channel * 4, base_channel * 4, gn=gn)

        self.conv5 = getattr(module, stride_conv_name)(base_channel*4, base_channel*8, kernel_size=(1,3,3), stride=(1,2,2), pad=(0,1,1), gn=gn)
        self.conv6 = getattr(module, conv_name)(base_channel*8, base_channel*8, gn=gn)
        self.conv6_ = getattr(module, conv_name)(base_channel * 8, base_channel * 8, gn=gn)
        self.pro_enhance = pro_enhance
        if self.pro_enhance:
            self.Pro_Enhance = Pro_Enhance(in_channel=pre_depth_channel+mono_channel, base_channels=pro_base_channels, gn=gn)

        if gn == "IN":
            normlayer1 = nn.InstanceNorm3d(base_channel * 4)
            normlayer2 = nn.InstanceNorm3d(base_channel * 2)
            normlayer3 = nn.InstanceNorm3d(base_channel * 1)
        elif gn:
            normlayer1 = nn.GroupNorm(int(max(1, base_channel*4 / 4)), base_channel*4)
            normlayer2 = nn.GroupNorm(int(max(1, base_channel*2 / 4)), base_channel*2)
            normlayer3 = nn.GroupNorm(int(max(1, base_channel / 4)), base_channel)
            print(gn)
            if gn:
                print(1)
            else:
                print(0)
        else:
            normlayer1 = nn.BatchNorm3d(base_channel * 4)
            normlayer2 = nn.BatchNorm3d(base_channel * 2)
            normlayer3 = nn.BatchNorm3d(base_channel * 1)


        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(base_channel*8, base_channel*4, kernel_size=(1,3,3), padding=(0,1,1), output_padding=(0,1,1), stride=(1,2,2), bias=False),
            normlayer1,
            # nn.GroupNorm(int(max(1, base_channel*4 / 4)), base_channel*4) if gn else nn.BatchNorm3d(base_channel*4),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(base_channel*4, base_channel*2, kernel_size=(1,3,3), padding=(0,1,1), output_padding=(0,1,1), stride=(1,2,2), bias=False),
            normlayer2,
            # nn.GroupNorm(int(max(1, base_channel*2 / 4)), base_channel*2) if gn else nn.BatchNorm3d(base_channel*2),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(base_channel*2, base_channel, kernel_size=(1,3,3), padding=(0,1,1), output_padding=(0,1,1), stride=(1,2,2), bias=False),
            normlayer3,
            # nn.GroupNorm(int(max(1, base_channel / 4)), base_channel) if gn else nn.BatchNorm3d(base_channel),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 1, stride=1, padding=0)

    def forward(self, x, depth_feature):
        conv0 = self.conv0_(self.conv0(x))
        conv2 = self.conv2_(self.conv2(self.conv1(conv0)))
        conv4 = self.conv4_(self.conv4(self.conv3(conv2)))
        x = self.conv6_(self.conv6(self.conv5(conv4)))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)


        prob = self.prob(x)
        if self.pro_enhance:
            # print("1", prob.shape, depth_feature.shape)
            _, _, D, H1, W1 = prob.shape
            _, _, H2, W2 = depth_feature.shape
            if H2 != H1:
                prob = F.interpolate(prob, [D, H2, W2], mode='trilinear', align_corners=True)
            prob = self.Pro_Enhance(prob, depth_feature)
            # print("Pro_Enhance")
        else:
            prob = prob.squeeze(1)

        return prob, x

class ViTFeaturePyramid_4stage(nn.Module):
    """
    This module implements SimpleFeaturePyramid in :paper:`vitdet`.
    It creates pyramid features built on top of the input feature map.
    """

    def __init__(
        self,
        in_channels,
        base_channels,
        scale_factors,
        gn=False
    ):
        """
        Args:
            scale_factors (list[float]): list of scaling factors to upsample or downsample
                the input features for creating pyramid features.
        """
        super(ViTFeaturePyramid_4stage, self).__init__()

        self.scale_factors = scale_factors
        out_dim = dim = base_channels
        self.stages = nn.ModuleList()
        self.conv = Conv2d(in_channels, base_channels, 3, 1, padding=1, gn=gn)
        for idx, scale in enumerate(scale_factors):
            if scale == 4:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1:
                layers = []
            elif scale == 0.5:
                layers = [Conv2d(dim, dim * 2, 3, stride=2, padding=1, gn=gn)]
                out_dim = dim * 2
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            if scale != 1:
                layers.extend(
                    [
                        nn.GELU(),
                        nn.Conv2d(out_dim, out_dim, 3, 1, 1),
                    ]
                )
            layers = nn.Sequential(*layers)

            self.stages.append(layers)

    def forward(self, x):
        results = {}
        x = self.conv(x)
        for idx, scale in enumerate(self.scale_factors):
            results["stage{}".format(idx+1)] = self.stages[idx](x)

        return results

def upsample_depth(depth, mask, ratio=8):
    """ Upsample depth field [H/ratio, W/ratio, 2] -> [H, W, 2] using convex combination """
    N, _, H, W = depth.shape
    mask = mask.view(N, 1, 9, ratio, ratio, H, W)
    mask = torch.softmax(mask, dim=2)

    up_flow = F.unfold(depth, [3, 3], padding=1)
    up_flow = up_flow.view(N, 1, 9, 1, 1, H, W)

    up_flow = torch.sum(mask * up_flow, dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
    return up_flow.reshape(N, ratio * H, ratio * W)

class cost_up(nn.Module):
    def __init__(self, in_channels, base_channels, gn=False, using_pro=False):
        super(cost_up, self).__init__()

        self.conv0 = ConvBnReLU3D(in_channels, base_channels, pad=1, gn=gn)
        self.conv1 = ConvBnReLU3D(base_channels, base_channels, stride=(1,2,2), pad=1, gn=gn)
        if using_pro:
            self.conv_cost = ConvBnReLU3D(1, base_channels, pad=1, gn=gn)
        else:
            self.conv_cost = ConvBnReLU3D(base_channels, base_channels, pad=1, gn=gn)
        self.conv2 = ConvBnReLU3D(base_channels * 2, base_channels * 2, pad=1, gn=gn)
        # self.conv3 = Deconv3d(base_channels * 2, base_channels, stride=(1,2,2), padding=1, output_padding=(0,1,1), norm_type=norm_type)
        if gn == "IN":
            normlayer = nn.InstanceNorm3d(base_channels)
        elif gn:
            print("cost_up gn")
            normlayer = nn.GroupNorm(int(max(1, base_channels / 4)), base_channels)
        else:
            normlayer = nn.BatchNorm3d(base_channels)
        self.conv3 = nn.Sequential(
            nn.ConvTranspose3d(base_channels*2, base_channels, kernel_size=(1,3,3), padding=(0,1,1), output_padding=(0,1,1), stride=(1,2,2), bias=False),
            normlayer,
            # nn.GroupNorm(int(max(1, base_channel / 4)), base_channel) if gn else nn.BatchNorm3d(base_channel),
            nn.ReLU(inplace=True))
        # self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)
    def forward(self, x, agg_pro_cost):

        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        agg_pro_cost_ = self.conv_cost(agg_pro_cost)
        conv2 = self.conv2(torch.cat([conv1, agg_pro_cost_], dim=1))
        # print(conv2.shape)
        conv3 = self.conv3(conv2)
        # print(x.shape, conv1.shape, agg_pro_cost.shape, conv3.shape)
        pro = conv3 + conv0
        # x_ = self.prob(pro)
        return pro, pro

class feature_up(nn.Module):
    def __init__(self, in_channels, in_channels_last, base_channels, gn='BN'):
        super(feature_up, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels, base_channels, pad=1, gn=gn)
        self.conv1 = ConvBnReLU3D(base_channels, base_channels, pad=1, gn=gn)
        self.conv_fea = ConvBnReLU3D(in_channels_last, base_channels, pad=1, gn=gn)
        self.conv2 = ConvBnReLU3D(base_channels * 2, base_channels * 2, pad=1, gn=gn)
        self.conv3 = ConvBnReLU3D(base_channels * 2, in_channels, pad=1, gn=gn)

    def forward(self, x, x_last):
        # print(x_last.shape)
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv_fea = self.conv_fea(x_last)
        conv2 = self.conv2(torch.cat([conv1, conv_fea], dim=1))
        # print(conv2.shape)
        conv3 = self.conv3(conv2)

        return conv3, conv3

class mask_up(nn.Module):
    """
    FPN aligncorners downsample 4x"""
    def __init__(self, mono_channel, base_channels, ratio=2, gn=False):
        super(mask_up, self).__init__()

        self.pre_convd1 = nn.Conv2d(1, base_channels, 7, padding=3)
        self.pre_convd2 = nn.Conv2d(base_channels, base_channels, 3, padding=1)

        self.mono_convd1 = nn.Conv2d(mono_channel, base_channels, 3, padding=1)

        self.cat_convd = nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1)
        self.cat_convc = nn.Conv2d(base_channels*2, ratio*ratio*9, 1, padding=0)



    def forward(self, pre_depth, mono_depth):
        # print(pre_depth.shape, mono_depth.shape)
        pre_depth = pre_depth.unsqueeze(1)
        pre_depth = F.relu(self.pre_convd1(pre_depth))
        pre_depth = F.relu(self.pre_convd2(pre_depth))
        mono_depth = F.relu(self.mono_convd1(mono_depth))
        depth_cat = F.relu(self.cat_convd(torch.cat([pre_depth, mono_depth], dim=1)))
        mask = self.cat_convc(depth_cat)
        mask = .25 * mask
        return mask

class Pro_Enhance(nn.Module):
    """
    FPN aligncorners downsample 4x"""
    def __init__(self, in_channel, base_channels, gn=False):
        super(Pro_Enhance, self).__init__()
        self.base_channels = base_channels

        self.monoconv0 = nn.Sequential(
            Conv2d(in_channel, base_channels, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels, base_channels, 3, 1, padding=1, gn=gn),
        )

        self.conv0 = nn.Sequential(
            Conv2d(in_channel, base_channels, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels, base_channels, 3, 1, padding=1, gn=gn),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2, gn=gn),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1, gn=gn),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2, gn=gn),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1, gn=gn),
        )


        self.out_channels = [4 * base_channels]
        final_chs = base_channels * 4


        self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
        self.inner2 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

        self.out = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
        # self.pro = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)

        self.out_channels.append(base_channels * 4)
        self.out_channels.append(base_channels * 2)
        self.out_channels.append(base_channels)


    def forward(self, pro, depth_feature):
        pro = pro.squeeze(1)
        # print(pro.shape, depth_feature.shape)
        cat_feature = torch.cat([pro, depth_feature], dim=1)
        # ref_outputs = []

        ref_conv0 = self.conv0(cat_feature)
        ref_conv1 = self.conv1(ref_conv0)
        ref_conv2 = self.conv2(ref_conv1)

        # ref_intra = ref_conv2.clone()
        intra = ref_conv2
        intra = F.interpolate(intra, scale_factor=2, mode="bilinear", align_corners=True) + self.inner1(ref_conv1)
        intra = F.interpolate(intra, scale_factor=2, mode="bilinear", align_corners=True) + self.inner2(ref_conv0)
        ref_out = self.out(intra)


        return ref_out

def depth_regression(p, depth_values):
    if depth_values.dim() <= 2:
        # print("regression dim <= 2")
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)

    return depth

def homo_warping_One_Featrue(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    C = src_fea.shape[1]
    # if UsingSrcSize:
    #     Hs,Ws = src_fea.shape[-2:]
    B,num_depth,Hr,Wr = depth_values.shape

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, Hr, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, Wr, dtype=torch.float32, device=src_fea.device)])
        y = y.reshape(Hr*Wr)
        x = x.reshape(Hr*Wr)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(B, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.reshape(B, 1, num_depth, -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.reshape(B, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        # FIXME divide 0
        temp = proj_xyz[:, 2:3, :, :]
        temp[temp==0] = 1e-9
        proj_xy = proj_xyz[:, :2, :, :] / temp  # [B, 2, Ndepth, H*W]
        # proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]

        proj_x_normalized = proj_xy[:, 0, :, :] / ((Wr - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((Hr - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy
    if len(src_fea.shape)==4:
        warped_src_fea = F.grid_sample(src_fea, grid.reshape(B, num_depth * Hr, Wr, 2), mode='bilinear', padding_mode='zeros', align_corners=True)
        warped_src_fea = warped_src_fea.reshape(B, C, num_depth, Hr, Wr)
    elif len(src_fea.shape)==5:
        warped_src_fea = []
        for d in range(src_fea.shape[2]):
            warped_src_fea.append(F.grid_sample(src_fea[:,:,d], grid.reshape(B, num_depth, Hr, Wr, 2)[:,d], mode='bilinear', padding_mode='zeros', align_corners=True))
        warped_src_fea = torch.stack(warped_src_fea, dim=2)

    return warped_src_fea




