from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import random


class CVC_datasets(Dataset):
    def __init__(self, path_Data, config, train=True, train_ratio=0.8):
        super(CVC_datasets, self).__init__()
        
        # CVC数据集路径
        images_dir = os.path.join(path_Data, 'PNG', 'Original')
        masks_dir = os.path.join(path_Data, 'PNG', 'Ground Truth')
        
        # 获取图像文件
        images_list = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
        
        # 构建图像-mask对
        self.data = []
        for img_name in images_list:
            img_path = os.path.join(images_dir, img_name)
            mask_path = os.path.join(masks_dir, img_name)
            
            if os.path.exists(mask_path):
                self.data.append([img_path, mask_path])
        
        # 数据集划分
        total_samples = len(self.data)
        train_samples = int(total_samples * train_ratio)
        
        random.seed(42)
        random.shuffle(self.data)
        
        if train:
            self.data = self.data[:train_samples]
            self.transformer = config.train_transformer
        else:
            self.data = self.data[train_samples:]
            self.transformer = config.test_transformer
        
        print(f"CVC Dataset - {'Train' if train else 'Val'}: {len(self.data)} samples")
        
    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        
        # 加载图像和mask
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255.0
        
        # 应用变换
        img, msk = self.transformer((img, msk))
        
        return img, msk

    def __len__(self):
        return len(self.data)