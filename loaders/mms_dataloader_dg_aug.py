from PIL import Image
import torchfile
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
import torch
import torch.nn as nn
import os
import torchvision.utils as vutils
import numpy as np
import torch.nn.init as init
import torch.utils.data as data
import torch
import random
import xlrd
import math
from skimage.exposure import match_histograms

# Data directories
LabeledVendorA_data_dir = '/home/s1575424/xiao/Year2/miccai2021/mnms_split_2D_data/Labeled/vendorA/'
LabeledVendorA_mask_dir = '/home/s1575424/xiao/Year2/miccai2021/mnms_split_2D_mask/Labeled/vendorA/'
ReA_dir = '/home/s1575424/xiao/Year2/miccai2021/mnms_split_2D_re/Labeled/vendorA/'

LabeledVendorB2_data_dir = '/home/s1575424/xiao/Year2/miccai2021/mnms_split_2D_data/Labeled/vendorB/center2/'
LabeledVendorB2_mask_dir = '/home/s1575424/xiao/Year2/miccai2021/mnms_split_2D_mask/Labeled/vendorB/center2/'
ReB2_dir = '/home/s1575424/xiao/Year2/miccai2021/mnms_split_2D_re/Labeled/vendorB/center2/'

LabeledVendorB3_data_dir = '/home/s1575424/xiao/Year2/miccai2021/mnms_split_2D_data/Labeled/vendorB/center3/'
LabeledVendorB3_mask_dir = '/home/s1575424/xiao/Year2/miccai2021/mnms_split_2D_mask/Labeled/vendorB/center3/'
ReB3_dir = '/home/s1575424/xiao/Year2/miccai2021/mnms_split_2D_re/Labeled/vendorB/center3/'

LabeledVendorC_data_dir = '/home/s1575424/xiao/Year2/miccai2021/mnms_split_2D_data/Labeled/vendorC/'
LabeledVendorC_mask_dir = '/home/s1575424/xiao/Year2/miccai2021/mnms_split_2D_mask/Labeled/vendorC/'
ReC_dir = '/home/s1575424/xiao/Year2/miccai2021/mnms_split_2D_re/Labeled/vendorC/'

LabeledVendorD_data_dir = '/home/s1575424/xiao/Year2/miccai2021/mnms_split_2D_data/Labeled/vendorD/'
LabeledVendorD_mask_dir = '/home/s1575424/xiao/Year2/miccai2021/mnms_split_2D_mask/Labeled/vendorD/'
ReD_dir = '/home/s1575424/xiao/Year2/miccai2021/mnms_split_2D_re/Labeled/vendorD/'

UnlabeledVendorC_data_dir = '/home/s1575424/xiao/Year2/miccai2021/mnms_split_2D_data/Unlabeled/vendorC/'
UnReC_dir = '/home/s1575424/xiao/Year2/miccai2021/mnms_split_2D_re/Unlabeled/vendorC/'

Re_dir = [ReA_dir, ReB2_dir, ReB3_dir, ReC_dir, ReD_dir]
Labeled_data_dir = [LabeledVendorA_data_dir, LabeledVendorB2_data_dir, LabeledVendorB3_data_dir, LabeledVendorC_data_dir, LabeledVendorD_data_dir]
Labeled_mask_dir = [LabeledVendorA_mask_dir, LabeledVendorB2_mask_dir, LabeledVendorB3_mask_dir, LabeledVendorC_mask_dir, LabeledVendorD_mask_dir]

# A, B 1.2
# C, D 1.1
resampling_rate = 1.2

def get_dg_data_loaders(batch_size, test_vendor='D', image_size=224):

    random.seed(14)

    train_labeled_loader, train_labeled_dataset, train_unlabeled_loader, train_unlabeled_dataset, test_loader, test_dataset = get_data_loader_folder(Labeled_data_dir, Labeled_mask_dir, batch_size, image_size, test_num=test_vendor)

    return train_labeled_loader, train_labeled_dataset, train_unlabeled_loader, train_unlabeled_dataset, test_loader, test_dataset


def get_data_loader_folder(data_folders, mask_folders, batch_size, new_size=224, test_num='D', num_workers=4):
    if test_num=='A':
        ref_data_dirs = [data_folders[1]]
        train_data_dirs = data_folders[1:]
        train_mask_dirs = mask_folders[1:]
        train_re = Re_dir[1:]
        test_data_dirs = [data_folders[0]]
        test_mask_dirs = [mask_folders[0]]
        test_re = [Re_dir[0]]
        train_num = [74, 51, 50, 50]
        test_num = [95]
    elif test_num=='B':
        ref_data_dirs = [data_folders[0]]
        train_data_dirs = [data_folders[0], data_folders[3], data_folders[4]]
        train_mask_dirs = [mask_folders[0], mask_folders[3], mask_folders[4]]
        test_data_dirs = data_folders[1:3]
        test_mask_dirs = mask_folders[1:3]
        train_re = [Re_dir[0], Re_dir[3], Re_dir[4]]
        test_re = Re_dir[1:3]
        train_num = [95, 50, 50]
        test_num = [74, 51]
    elif test_num=='C':
        ref_data_dirs = [data_folders[0]]
        train_data_dirs = [data_folders[0], data_folders[1], data_folders[2], data_folders[4]]
        train_mask_dirs = [mask_folders[0], mask_folders[1], mask_folders[2], mask_folders[4]]
        test_data_dirs = [data_folders[3]]
        test_mask_dirs = [mask_folders[3]]
        train_re = [Re_dir[0], Re_dir[1], Re_dir[2], Re_dir[4]]
        test_re = [Re_dir[3]]
        train_num = [95, 74, 51, 50]
        test_num = [50]
    elif test_num=='D':
        ref_data_dirs = [data_folders[0]]
        # un_train_data_dirs = data_folders[1:3]
        # un_train_mask_dirs = mask_folders[1:3]
        train_data_dirs = [data_folders[0], data_folders[1], data_folders[2], data_folders[3]]
        train_mask_dirs = [mask_folders[0], mask_folders[1], mask_folders[2], mask_folders[3]]
        test_data_dirs = [data_folders[4]]
        test_mask_dirs = [mask_folders[4]]
        # un_train_re = Re_dir[1:3]
        train_re = [Re_dir[0], Re_dir[1], Re_dir[2], Re_dir[3]]
        test_re = [Re_dir[4]]
        # un_train_num = [95, 74, 51, 50]
        train_num = [95, 74, 51, 50]
        test_num = [50]
    else:
        print('Wrong test vendor!')

    train_labeled_dataset = ImageFolder(train_data_dirs, train_mask_dirs, ref_data_dirs, train_re, num_label=train_num, train=True, labeled=True)
    train_unlabeled_dataset = ImageFolder(train_data_dirs, train_mask_dirs, ref_data_dirs, train_re, num_label=train_num, train=True, labeled=False)
    test_dataset = ImageFolder(test_data_dirs, test_mask_dirs, ref_data_dirs, test_re, num_label=test_num, train=False, labeled=True)

    train_labeled_loader = DataLoader(dataset=train_labeled_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    train_unlabeled_loader = DataLoader(dataset=train_unlabeled_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)

    return train_labeled_loader, train_labeled_dataset, train_unlabeled_loader, train_unlabeled_dataset, test_loader, test_dataset

def default_loader(path):
    return np.load(path)['arr_0']

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            images.append(path)
    return images

class ImageFolder(data.Dataset):
    def __init__(self, data_dirs, mask_dirs, ref_dir,  re, train=True, num_label=None, labeled=True, loader=default_loader):

        print(data_dirs)
        print(mask_dirs)

        reso_dir = re
        temp_imgs = []
        temp_masks = []
        temp_re = []


        if train:
            #100%
            # k = 1
            #5%
            k = 0.05
            #2%
            # k = 0.02
        else:
            k = 1

        for num_set in range(len(data_dirs)):
            re_roots = sorted(make_dataset(reso_dir[num_set]))
            data_roots = sorted(make_dataset(data_dirs[num_set]))
            mask_roots = sorted(make_dataset(mask_dirs[num_set]))
            num_label_data = 0
            for num_data in range(len(data_roots)):
                if labeled:
                    if train:
                        n_label = str(math.ceil(num_label[num_set] * k + 1))
                        if '00'+n_label==data_roots[num_data][-10:-7] or '0'+n_label==data_roots[num_data][-10:-7]:
                            print(n_label)
                            print(data_roots[num_data][-10:-7])
                            break

                    for num_mask in range(len(mask_roots)):
                        if data_roots[num_data][-10:-4] == mask_roots[num_mask][-10:-4]:
                            temp_re.append(re_roots[num_data])
                            temp_imgs.append(data_roots[num_data])
                            temp_masks.append(mask_roots[num_mask])
                            num_label_data += 1
                        else:
                            pass
                else:
                    temp_re.append(re_roots[num_data])
                    temp_imgs.append(data_roots[num_data])

        tem_ref_imgs = []
        for num_set in range(len(ref_dir)):
            data_roots = sorted(make_dataset(ref_dir[num_set]))
            for num_data in range(len(data_roots)):
                    tem_ref_imgs.append(data_roots[num_data])

        reso = temp_re
        imgs = temp_imgs
        masks = temp_masks

        # add something here to index, such that can split the data
        # index = random.sample(range(len(temp_img)), len(temp_mask))

        # if len(imgs) == 0:
        #     raise(RuntimeError("Found 0 images in: " + root1 + "\n"
        #                        "Supported image extensions are: " +
        #                        ",".join(IMG_EXTENSIONS)))

        ref_imgs = tem_ref_imgs
        self.ref = ref_imgs

        self.reso = reso
        self.imgs = imgs
        self.masks = masks
        self.new_size = 288
        self.loader = loader
        self.labeled = labeled
        self.train = train

    def __getitem__(self, index):

        path_re = self.reso[index]
        re = self.loader(path_re)
        re = re[0]

        path_img = self.imgs[index]
        img = self.loader(path_img) # numpy, HxW, numpy.Float64

        # ref_paths = random.sample(self.ref, 1)
        # ref_img = self.loader(ref_paths[0])
        #
        # img = match_histograms(img, ref_img)

        # Intensity cropping:
        p5 = np.percentile(img.flatten(), 0.5)
        p95 = np.percentile(img.flatten(), 99.5)
        img = np.clip(img, p5, p95)

        img -= img.min()
        img /= img.max()
        img = img.astype('float32')

        crop_size = 300

        # Augmentations:
        # 1. random rotation
        # 2. random scaling 0.8 - 1.2
        # 3. random crop from 280x280
        # 4. random hflip
        # 5. random vflip
        # 6. color jitter
        # 7. Gaussian filtering

        img_tensor = F.to_tensor(np.array(img))
        img_size = img_tensor.size()

        if self.labeled:
            if self.train:
                img = F.to_pil_image(img)
                # rotate, random angle between 0 - 90
                angle = random.randint(0, 90)
                img = F.rotate(img, angle, InterpolationMode.BILINEAR)

                path_mask = self.masks[index]
                mask = Image.open(path_mask)  # numpy, HxWx3
                # rotate, random angle between 0 - 90
                mask = F.rotate(mask, angle, InterpolationMode.NEAREST)

                ## Find the region of mask
                norm_mask = F.to_tensor(np.array(mask))
                region = norm_mask[0] + norm_mask[1] + norm_mask[2]
                non_zero_index = torch.nonzero(region == 1, as_tuple=False)
                if region.sum() > 0:
                    len_m = len(non_zero_index[0])
                    x_region = non_zero_index[len_m // 2][0]
                    y_region = non_zero_index[len_m // 2][1]
                    x_region = int(x_region.item())
                    y_region = int(y_region.item())
                else:
                    x_region = norm_mask.size(-2) // 2
                    y_region = norm_mask.size(-1) // 2

                # resize and center-crop to 280x280
                resize_order = re / resampling_rate
                resize_size_h = int(img_size[-2] * resize_order)
                resize_size_w = int(img_size[-1] * resize_order)

                left_size = 0
                top_size = 0
                right_size = 0
                bot_size = 0
                if resize_size_h < self.new_size:
                    top_size = (self.new_size - resize_size_h) // 2
                    bot_size = (self.new_size - resize_size_h) - top_size
                if resize_size_w < self.new_size:
                    left_size = (self.new_size - resize_size_w) // 2
                    right_size = (self.new_size - resize_size_w) - left_size

                transform_list = [transforms.Pad((left_size, top_size, right_size, bot_size))]
                transform_list = [transforms.Resize((resize_size_h, resize_size_w))] + transform_list
                transform = transforms.Compose(transform_list)

                img = transform(img)

                ## Define the crop index
                if top_size >= 0:
                    top_crop = 0
                else:
                    if x_region > self.new_size // 2:
                        if x_region - self.new_size // 2 + self.new_size <= norm_mask.size(-2):
                            top_crop = x_region - self.new_size // 2
                        else:
                            top_crop = norm_mask.size(-2) - self.new_size
                    else:
                        top_crop = 0

                if left_size >= 0:
                    left_crop = 0
                else:
                    if y_region > self.new_size // 2:
                        if y_region - self.new_size // 2 + self.new_size <= norm_mask.size(-1):
                            left_crop = y_region - self.new_size // 2
                        else:
                            left_crop = norm_mask.size(-1) - self.new_size
                    else:
                        left_crop = 0

                # random crop to 224x224
                img = F.crop(img, top_crop, left_crop, self.new_size, self.new_size)

                # random flip
                hflip_p = random.random()
                img = F.hflip(img) if hflip_p >= 0.5 else img
                vflip_p = random.random()
                img = F.vflip(img) if vflip_p >= 0.5 else img

                img = F.to_tensor(np.array(img))

                # # Gamma correction: random gamma from [0.5, 1.5]
                # gamma = 0.5 + random.random()
                # img_max = img.max()
                # img = img_max * torch.pow((img / img_max), gamma)

                # Gaussian bluring:
                transform_list = [transforms.GaussianBlur(5, sigma=(0.25, 1.25))]
                transform = transforms.Compose(transform_list)
                img = transform(img)

                # resize and center-crop to 280x280
                transform_mask_list = [transforms.Pad(
                    (left_size, top_size, right_size, bot_size))]
                transform_mask_list = [transforms.Resize((resize_size_h, resize_size_w),
                                                         interpolation=InterpolationMode.NEAREST)] + transform_mask_list
                transform_mask = transforms.Compose(transform_mask_list)

                mask = transform_mask(mask)  # C,H,W

                # random crop to 224x224
                mask = F.crop(mask, top_crop, left_crop, self.new_size, self.new_size)

                # random flip
                mask = F.hflip(mask) if hflip_p >= 0.5 else mask
                mask = F.vflip(mask) if vflip_p >= 0.5 else mask

                mask = F.to_tensor(np.array(mask))

                mask_bg = (mask.sum(0) == 0).type_as(mask)  # H,W
                mask_bg = mask_bg.reshape((1, mask_bg.size(0), mask_bg.size(1)))
                mask = torch.cat((mask, mask_bg), dim=0)

                return img, mask # pytorch: N,C,H,W

            else:
                path_mask = self.masks[index]
                mask = Image.open(path_mask)  # numpy, HxWx3
                # resize and center-crop to 280x280

                ## Find the region of mask
                norm_mask = F.to_tensor(np.array(mask))
                region = norm_mask[0] + norm_mask[1] + norm_mask[2]
                non_zero_index = torch.nonzero(region == 1, as_tuple=False)
                if region.sum() > 0:
                    len_m = len(non_zero_index[0])
                    x_region = non_zero_index[len_m // 2][0]
                    y_region = non_zero_index[len_m // 2][1]
                    x_region = int(x_region.item())
                    y_region = int(y_region.item())
                else:
                    x_region = norm_mask.size(-2) // 2
                    y_region = norm_mask.size(-1) // 2

                resize_order = re / resampling_rate
                resize_size_h = int(img_size[-2] * resize_order)
                resize_size_w = int(img_size[-1] * resize_order)

                left_size = 0
                top_size = 0
                right_size = 0
                bot_size = 0
                if resize_size_h < self.new_size:
                    top_size = (self.new_size - resize_size_h) // 2
                    bot_size = (self.new_size - resize_size_h) - top_size
                if resize_size_w < self.new_size:
                    left_size = (self.new_size - resize_size_w) // 2
                    right_size = (self.new_size - resize_size_w) - left_size

                # transform_list = [transforms.CenterCrop((crop_size, crop_size))]
                transform_list = [transforms.Pad((left_size, top_size, right_size, bot_size))]
                transform_list = [transforms.Resize((resize_size_h, resize_size_w))] + transform_list
                transform_list = [transforms.ToPILImage()] + transform_list
                transform = transforms.Compose(transform_list)
                img = transform(img)
                img = F.to_tensor(np.array(img))

                ## Define the crop index
                if top_size >= 0:
                    top_crop = 0
                else:
                    if x_region > self.new_size // 2:
                        if x_region - self.new_size // 2 + self.new_size <= norm_mask.size(-2):
                            top_crop = x_region - self.new_size // 2
                        else:
                            top_crop = norm_mask.size(-2) - self.new_size
                    else:
                        top_crop = 0

                if left_size >= 0:
                    left_crop = 0
                else:
                    if y_region > self.new_size // 2:
                        if y_region - self.new_size // 2 + self.new_size <= norm_mask.size(-1):
                            left_crop = y_region - self.new_size // 2
                        else:
                            left_crop = norm_mask.size(-1) - self.new_size
                    else:
                        left_crop = 0

                # random crop to 224x224
                img = F.crop(img, top_crop, left_crop, self.new_size, self.new_size)

                # resize and center-crop to 280x280
                # transform_mask_list = [transforms.CenterCrop((crop_size, crop_size))]
                transform_mask_list = [transforms.Pad(
                    (left_size, top_size, right_size, bot_size))]
                transform_mask_list = [transforms.Resize((resize_size_h, resize_size_w),
                                                         interpolation=InterpolationMode.NEAREST)] + transform_mask_list
                transform_mask = transforms.Compose(transform_mask_list)

                mask = transform_mask(mask)  # C,H,W
                mask = F.crop(mask, top_crop, left_crop, self.new_size, self.new_size)
                mask = F.to_tensor(np.array(mask))

                mask_bg = (mask.sum(0) == 0).type_as(mask)  # H,W
                mask_bg = mask_bg.reshape((1, mask_bg.size(0), mask_bg.size(1)))
                mask = torch.cat((mask, mask_bg), dim=0)

                return img, mask

        else:
            img = F.to_pil_image(img)
            # rotate, random angle between 0 - 90
            angle = random.randint(0, 90)
            img = F.rotate(img, angle, InterpolationMode.BILINEAR)

            # resize and center-crop to 280x280
            resize_order = re / resampling_rate
            resize_size_h = int(img_size[-2] * resize_order)
            resize_size_w = int(img_size[-1] * resize_order)

            left_size = 0
            top_size = 0
            right_size = 0
            bot_size = 0
            if resize_size_h < crop_size:
                top_size = (crop_size - resize_size_h) // 2
                bot_size = (crop_size - resize_size_h) - top_size
            if resize_size_w < crop_size:
                left_size = (crop_size - resize_size_w) // 2
                right_size = (crop_size - resize_size_w) - left_size

            transform_list = [transforms.CenterCrop((crop_size, crop_size))]
            transform_list = [transforms.Pad((left_size, top_size, right_size, bot_size))] + transform_list
            transform_list = [transforms.Resize((resize_size_h, resize_size_w))] + transform_list
            transform = transforms.Compose(transform_list)

            img = transform(img)

            # random crop to 224x224
            top_crop = random.randint(0, crop_size - self.new_size)
            left_crop = random.randint(0, crop_size - self.new_size)
            img = F.crop(img, top_crop, left_crop, self.new_size, self.new_size)

            # random flip
            hflip_p = random.random()
            img = F.hflip(img) if hflip_p >= 0.5 else img
            vflip_p = random.random()
            img = F.vflip(img) if vflip_p >= 0.5 else img

            img = F.to_tensor(np.array(img))

            # # Gamma correction: random gamma from [0.5, 1.5]
            # gamma = 0.5 + random.random()
            # img_max = img.max()
            # img = img_max*torch.pow((img/img_max), gamma)

            # Gaussian bluring:
            transform_list = [transforms.GaussianBlur(5, sigma=(0.25, 1.25))]
            transform = transforms.Compose(transform_list)
            img = transform(img)

            return img # pytorch: N,C,H,W

    def __len__(self):
        return len(self.imgs)
