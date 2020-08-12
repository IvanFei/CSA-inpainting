import os
import cv2
import math
import random
import torch
from PIL import Image, ImageFilter
from glob import glob
import torch.utils.data
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from pycocotools.coco import COCO


class Data_load(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, img_transform, mask_transform):
        super(Data_load, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        self.paths = glob('{:s}/*'.format(img_root), recursive=True)

        self.mask_paths = glob('{:s}/*.png'.format(mask_root))

        self.N_mask = len(self.mask_paths)

    def __getitem__(self, index):
        gt_img = Image.open(self.paths[index])
        gt_img = self.img_transform(gt_img.convert('RGB'))

        mask = Image.open(self.mask_paths[random.randint(0, self.N_mask - 1)])
        mask = self.mask_transform(mask.convert('RGB'))
        return gt_img, mask

    def __len__(self):
        return len(self.paths)


class MMFashionDataset(torch.utils.data.Dataset):
    def __init__(self, opt, debug=False, split="train"):
        super(MMFashionDataset, self).__init__()
        self.split = split
        self.mask_type = opt.mask_type
        self.data_root = opt.dataroot
        self.w, self.h = opt.fineSize, opt.fineSize
        data_anno_dir = os.path.join(self.data_root, "In-shop Clothes Retrieval Benchmark/Anno/segmentation/")
        anno_file = os.path.join(data_anno_dir, "DeepFashion_segmentation_train.json")
        self.coco = COCO(anno_file)
        self.image_list = [(img_id, self.coco.imgs[img_id]) for img_id in self.coco.imgs]

        self.needed_classes = ["top", "skirt", "leggings", "dress", "outer", "pants", "skin"]
        self._class_to_coco_cat_id = dict(list(zip([c for c in self.needed_classes],
                                                   self.coco.getCatIds(catNms=self.needed_classes))))
        self._cat_id_to_coco_class = dict(list(zip(self.coco.getCatIds(catNms=self.needed_classes),
                                                   [c for c in self.needed_classes])))

        if split == "train":
            self.image_list = self.image_list
            random.shuffle(self.image_list)
        else:
            self.image_list = random.sample(self.image_list, k=100)

        if debug:
            self.image_list = self.image_list[:100]

        if self.mask_type == "random":
            self.mask_list = glob(os.path.join(opt.maskroot, "*.png"))
            self.mask_list = self.mask_list * max(1, math.ceil(len(self.image_list) / len(self.mask_list)))
        else:
            self.mask_list = [0] * len(self.image_list)

        # params for crop image according mask
        self.crop_sizes = [64, 128, 256]
        self.edge_prob = 0.5

    def __getitem__(self, item):
        img_id, img_info = self.image_list[item]
        img_name = os.path.basename(img_info["file_name"])
        img = Image.open(os.path.join(self.data_root, img_info["file_name"])).convert("RGB")
        img_arr = np.asarray(img)
        height, width = img_arr.shape[:2]
        try:
            _, mask = self.get_coco_masks(self.coco, img_id, height, width)
        except:
            mask = np.zeros([height, width], dtype=np.float32)
            mask[3 * height // 8: 5 * height // 8, 3 * width // 8: 5 * width // 8] = 1.

        # id picked
        mask_cat_id = list(np.unique(mask))
        needed_id = list(set(self._cat_id_to_coco_class.keys()).intersection(set(mask_cat_id)))
        if needed_id:
            cat_id = random.sample(needed_id, k=1)[0]
        else:
            cat_id = random.sample(mask_cat_id, k=1)[0]

        mask = (mask == cat_id).astype(np.uint8) * 255

        # pick the image patch according the mask
        part_prob = random.uniform(0, 1)
        if part_prob <= self.edge_prob:
            erode_kernel = np.ones([3, 3])
            erode_mask = cv2.erode(mask, erode_kernel, iterations=1)
            selected_arae = mask - erode_mask
        else:
            selected_arae = mask

        y_set, x_set = np.where(selected_arae == 255)
        coord_idx = random.randint(0, len(y_set) - 1)
        patch_y, patch_x = y_set[coord_idx], x_set[coord_idx]

        # crop the image patch according the patch center coord
        crop_size = random.sample(self.crop_sizes, k=1)[0]
        x1, y1 = patch_x - crop_size // 2, patch_y - crop_size // 2
        x2, y2 = patch_x + crop_size // 2, patch_y + crop_size // 2
        x1 = 0 if x1 < 0 else x1
        y1 = 0 if y1 < 0 else y1
        x2 = width if x2 >= width else x2
        y2 = height if y2 >= height else y2

        img_patch = img_arr[y1: y2, x1: x2, :]
        img_patch = cv2.resize(img_patch, (self.w, self.h), interpolation=cv2.INTER_CUBIC)
        img = Image.fromarray(img_patch).convert("RGB")
        # img.save("/nfs/users/huangfeifei/PEN-Net-for-Inpainting/img.jpg")
        # generate the mask
        if self.mask_type == "random":
            m_idx = random.randint(0, len(self.mask_list) - 1)
            mask = Image.open(self.mask_list[m_idx]).convert("L")
        else:
            m = np.zeros([self.h, self.w]).astype(np.uint8)
            if self.split == "train":
                t, l = random.randint(0, self.h // 2), random.randint(0, self.w // 2)
                m[t: t+self.h//2, l: l+self.w//2] = 255
            else:
                m[self.h // 4:self.h * 3 // 4, self.w // 4:self.w * 3 // 4] = 255
            mask = Image.fromarray(m).convert("L")
        # mask.save("/nfs/users/huangfeifei/PEN-Net-for-Inpainting/mask.png")

        if self.split == 'train':
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ColorJitter(0.05, 0.05, 0.05, 0.05)(img)
            mask = transforms.RandomHorizontalFlip()(mask)
            mask = mask.rotate(random.randint(0, 45), expand=True)
            mask = mask.filter(ImageFilter.MaxFilter(3))

        img = img.resize((self.w, self.h))
        mask = mask.resize((self.w, self.h), Image.NEAREST)
        return F.to_tensor(img) * 2 - 1., F.to_tensor(mask)

    def __len__(self):
        return len(self.image_list)

    def set_subset(self, start, end):
        self.image_list = self.image_list[start: end]
        self.mask_list = self.mask_list[start: end]

    @staticmethod
    def get_coco_masks(coco, img_id, height, width):
        annIds = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        assert annIds is not None
        anns = coco.loadAnns(annIds)
        assert len(anns) > 0
        masks = []

        mask = np.zeros([height, width], dtype=np.float32)
        for ann in anns:
            m = coco.annToMask(ann)
            assert m.shape[0] == height and m.shape[1] == width
            masks.append(m)
            cat_id = ann["category_id"]
            m = m.astype(np.float32) * cat_id
            mask[m > 0] = m[m > 0]

        masks = np.asarray(masks)
        masks = masks.astype(np.uint8)
        mask = mask.astype(np.uint8)

        return masks, mask


if __name__ == '__main__':
    data_args = {"data_root": "/nfs/share/CV_data/DeepFashion/In-shop",
                 "w": 256, "h": 256, "extend": 6}

    mmfasion = MMFashionDataset(data_args)
    print(f"[*] len of data: {len(mmfasion)}")
    data_loader = torch.utils.data.DataLoader(mmfasion, batch_size=1, shuffle=True)
    for data_batch in data_loader:
        img, mask, img_name = data_batch
        print(f"[*] img name: {img_name}")
        print(f"[*] img shape: {img.shape}")
        print(f"[*] mask shape: {mask.shape}")







