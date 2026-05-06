import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from copy import deepcopy
from datasets.data_io import read_pfm
from models.depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
# from .color_jittor import ColorJitter
# import multiporcessing

# multiporcessing.set_start_method('spawn')

np.random.seed(123)
random.seed(123)
# torch.set_num_threads(8)

def save_mask(filename, mask):
    # assert mask.dtype == np.bool
    assert mask.dtype == bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)
class RandomGamma():
    def __init__(self, min_gamma=0.7, max_gamma=1.5, clip_image=False):
        self._min_gamma = min_gamma
        self._max_gamma = max_gamma
        self._clip_image = clip_image

    @staticmethod
    def get_params(min_gamma, max_gamma):
        return np.random.uniform(min_gamma, max_gamma)

    @staticmethod
    def adjust_gamma(image, gamma, clip_image):
        adjusted = torch.pow(image, gamma)
        if clip_image:
            adjusted.clamp_(0.0, 1.0)
        return adjusted

    def __call__(self, img, gamma):
        # gamma = self.get_params(self._min_gamma, self._max_gamma)
        return self.adjust_gamma(img, gamma, self._clip_image)


# the DTU dataset preprocessed by Yao Yao (only for training)
class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, interval_scale=1.06, ndepths=192, batch_size=8, crop=False, augment=False,
                 aug_args=None, height=512, width=640, patch_size=16, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.color_augment = transforms.ColorJitter(brightness=0.5, contrast=0.5)
        if mode != 'train':
            self.crop = False
            self.augment = False
            self.random_mask = False
        else:
            self.crop = crop
            self.augment = augment
        self.kwargs = kwargs
        self.multi_scale =False
        self.multi_scale_args = {
                    # "scales": [[512, 640]],
                    "scales":  [[512,640],[512,704],[512,768],
                                [576,704],[576,768],[576,832],
                                [640,832],[640,896],[640,960]],

                    "resize_range": [1.0,1.2],
                    "scale_batch_map": {
                        "512": 1,
                        "576": 1,
                        "640": 1,
                    }
                }

        print(self.multi_scale_args)
        self.resize_scale = 0.5
        self.scales = self.multi_scale_args['scales'][::-1]
        self.resize_range = self.multi_scale_args['resize_range']
        self.consist_crop = False
        self.batch_size = batch_size
        self.world_size = 4
        self.img_size_map = []
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.ref_transform = transforms.Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        # print("mvsdataset kwargs", self.kwargs)

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            pair_file = "mvs_training/dtu/Cameras/pair.txt"
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # src_views = src_views[:(self.nviews-1)]

                    # light conditions 0-6
                    # if self.mode == 'train':
                    #     lights = np.random.choice(np.arange(7), 4, replace=False)
                    # else:
                    lights = np.arange(7)
                    for light_idx in lights:
                        metas.append((scan, light_idx, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas





    def reset_dataset(self, shuffled_idx):
        self.idx_map = {}
        barrel_idx = 0
        count = 0
        for sid in shuffled_idx:
            self.idx_map[sid] = barrel_idx
            count += 1
            if count == self.batch_size:
                count = 0
                barrel_idx += 1

        # random img size:256~512
        if self.mode == 'train':
            barrel_num = int(len(self.metas) / (self.batch_size * self.world_size))
            barrel_num += 2
            self.img_size_map = np.arange(0, len(self.scales))

    def __len__(self):
        # return len(self.generate_img_index)
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename).convert('RGB')

        return img

    def ref_read_img(self, filename):
        img = cv2.imread(filename)

        return img

    def read_monodepth(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img


    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def read_mask_hr(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32)
        np_img = (np_img > 10).astype(np.float32)
        # np_img = self.prepare_img(np_img)
        return np_img

    def read_depth_hr(self, filename):
        # read pfm depth file
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)
        # return self.prepare_img(depth)
        return depth

    def generate_stage_depth(self, depth):
        h, w = depth.shape
        depth_ms = {
            "stage1": cv2.resize(depth, (w // 8, h // 8), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(depth, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage3": cv2.resize(depth, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage4": depth
        }
        return depth_ms

    def center_crop_img(self, img, new_h=None, new_w=None):
        h, w = img.shape[:2]

        if new_h != h or new_w != w:
            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            finish_h = start_h + new_h
            finish_w = start_w + new_w
            img = img[start_h:finish_h, start_w:finish_w]
        return img

    def center_crop_cam(self, intrinsics, h, w, new_h=None, new_w=None):
        if new_h != h or new_w != w:
            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            new_intrinsics = intrinsics.copy()
            new_intrinsics[0][2] = new_intrinsics[0][2] - start_w
            new_intrinsics[1][2] = new_intrinsics[1][2] - start_h
            return new_intrinsics
        else:
            return intrinsics

    def pre_resize(self, img, depth, intrinsic, mask, resize_scale):
        ori_h, ori_w, _ = img.shape
        img = cv2.resize(img, (int(ori_w * resize_scale), int(ori_h * resize_scale)), interpolation=cv2.INTER_AREA)
        h, w, _ = img.shape

        output_intrinsics = intrinsic.copy()
        output_intrinsics[0, :] *= resize_scale
        output_intrinsics[1, :] *= resize_scale

        if depth is not None:
            depth = cv2.resize(depth, (int(ori_w * resize_scale), int(ori_h * resize_scale)), interpolation=cv2.INTER_NEAREST)
        if mask is not None:
            mask = cv2.resize(mask, (int(ori_w * resize_scale), int(ori_h * resize_scale)), interpolation=cv2.INTER_NEAREST)

        return img, depth, output_intrinsics, mask

    def final_crop(self, img, depth, intrinsic, mask, crop_h, crop_w, offset_y=None, offset_x=None, ref_image=None):
        h, w, _ = img.shape
        if offset_x is None or offset_y is None:
            if self.crop:
                offset_y = random.randint(0, h - crop_h)
                offset_x = random.randint(0, w - crop_w)
            else:
                offset_y = (h - crop_h) // 2
                offset_x = (w - crop_w) // 2
        cropped_image = img[offset_y:offset_y + crop_h, offset_x:offset_x + crop_w, :]
        if ref_image is None:
            cropped_ref_image = None
        else:
            # print(img.shape, ref_image.shape)
            cropped_ref_image = ref_image[offset_y:offset_y + crop_h, offset_x:offset_x + crop_w,:]
        output_intrinsics = intrinsic.copy()
        output_intrinsics[0, 2] -= offset_x
        output_intrinsics[1, 2] -= offset_y

        if depth is not None:
            cropped_depth = depth[offset_y:offset_y + crop_h, offset_x:offset_x + crop_w]
        else:
            cropped_depth = None


        if mask is not None:
            cropped_mask = mask[offset_y:offset_y + crop_h, offset_x:offset_x + crop_w]
        else:
            cropped_mask = None

        return cropped_image, cropped_depth, output_intrinsics, cropped_mask, offset_y, offset_x, cropped_ref_image

    def __getitem__(self, idx):
        # cv2.setNumThreads(0)
        # cv2.ocl.setUseOpenCL(False)
        meta = self.metas[idx]
        scan, light_idx, ref_view, src_views = meta
        if self.mode == 'train':
            np.random.shuffle(src_views)
        view_ids = [ref_view] + src_views[:(self.nviews - 1)]
        # view_ids = [ref_view] + src_views

        imgs = []
        depth_values = None
        mask = None
        proj_matrices = []

        offset_y = None
        offset_x = None

        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor, gamma_factor = None, None, None, None, None, None
        visi_masks = []
        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            # NOTE that DTU_origin/Rectified saves the images with the original size (1200x1600)
            img_filename = os.path.join(self.datapath, 'Rectified/{}/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))
            mask_filename_hr = os.path.join(self.datapath, 'mvs_training/dtu/Depths_raw/{}/depth_visual_{:0>4}.png'.format(scan, vid))
            depth_filename_hr = os.path.join(self.datapath, 'mvs_training/dtu/Depths_raw/{}/depth_map_{:0>4}.pfm'.format(scan, vid))
            # these poses are based on original resolution
            proj_mat_filename = os.path.join(self.datapath, 'mvs_training/dtu/Cameras/{:0>8}_cam.txt').format(vid)
            img = self.read_img(img_filename)
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)

            if i == 0:
                depth_hr = self.read_depth_hr(depth_filename_hr)
                depth_mask_hr = self.read_mask_hr(mask_filename_hr)
                ref_img = self.ref_read_img(img_filename)
                ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB) / 255.0

            else:
                depth_hr = self.read_depth_hr(depth_filename_hr)
                depth_mask_hr = None

            # resize according to crop size
            if self.mode == 'train':
                # print(self.idx_map[idx])
                [crop_h, crop_w] = self.scales[self.idx_map[idx] % len(self.scales)]
                enlarge_scale = self.resize_range[0] + random.random() * (self.resize_range[1] - self.resize_range[0])
                resize_scale_h = np.clip((crop_h * enlarge_scale) / 1200, 0.2, 1.0)
                resize_scale_w = np.clip((crop_w * enlarge_scale) / 1600, 0.2, 1.0)
                resize_scale = max(resize_scale_h, resize_scale_w)
                # print(resize_scale)
            else:
                crop_h, crop_w = self.height, self.width
                resize_scale = self.resize_scale

            img = np.asarray(img)

            if resize_scale != 1.0:
                img, depth_hr, intrinsics, depth_mask_hr = self.pre_resize(img, depth_hr, intrinsics, depth_mask_hr, resize_scale)
                if i == 0:
                    ori_h, ori_w, _ = ref_img.shape
                    ref_img = cv2.resize(ref_img, (int(ori_w * resize_scale), int(ori_h * resize_scale)),interpolation=cv2.INTER_AREA)
            if i == 0:  # reference view
                while True:  # get resonable offset
                    # finally random crop
                    img_, depth_hr_, intrinsics_, depth_mask_hr_, offset_y, offset_x, ref_img = self.final_crop(img, depth_hr, intrinsics, depth_mask_hr,
                                                                                                       crop_h=crop_h, crop_w=crop_w, ref_image=ref_img)

                    mask_read_ms_ = self.generate_stage_depth(depth_mask_hr_)
                    if self.mode != 'train' or np.any(mask_read_ms_['stage1'] > 0.0):
                        break

                depth_ms = self.generate_stage_depth(depth_hr_)
                mask = mask_read_ms_
                img = img_

                intrinsics = intrinsics_


                depth_max = depth_interval * self.ndepths + depth_min

                depth_values = np.array([depth_min, depth_max], dtype=np.float32)
            else:
                if self.consist_crop:
                    img, depth_hr, intrinsics, depth_mask_hr, _, _, _ = self.final_crop(img, depth_hr, intrinsics, depth_mask_hr,
                                                                                     crop_h=crop_h, crop_w=crop_w,
                                                                                     offset_y=offset_y, offset_x=offset_x)
                else:
                    img, depth_hr, intrinsics, depth_mask_hr, _, _, _ = self.final_crop(img, depth_hr, intrinsics, depth_mask_hr,
                                                                                     crop_h=crop_h, crop_w=crop_w)
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics

            proj_matrices.append(proj_mat)

            img = Image.fromarray(img)


            imgs.append(self.transforms(img))



        # all

        proj_matrices = np.stack(proj_matrices)
        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.5
        stage1_pjmats = proj_matrices.copy()
        stage1_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.25
        stage0_pjmats = proj_matrices.copy()
        stage0_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.125



        proj_matrices_ms = {
            "stage1": stage0_pjmats,  # 1/8
            "stage2": stage1_pjmats,  # 1/4
            "stage3": stage2_pjmats,  # 1/2
            "stage4": proj_matrices  # 1/1
        }

        ref_img = self.ref_transform({'image': ref_img})['image']



        return {"imgs": imgs,  # Nv C H W
                "proj_matrices": proj_matrices_ms,  # 4 stage of Nv 2 4 4
                "depth": depth_ms,
                "ref_img": ref_img,
                "depth_values": depth_values,
                "mask": mask,
                "filename": scan + '{:0>8}'.format(view_ids[0]) + ".png"}
