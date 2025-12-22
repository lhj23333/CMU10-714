from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        
        # 读取图像文件
        with gzip.open(image_filename, 'rb') as f:
            # 读取头部：魔数(4) + 图像数量(4) + 行数(4) + 列数(4)
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            if magic != 2051:
                raise ValueError("Invalid magic number in image file")
                
            # 读取所有像素数据
            image_data = np.frombuffer(f.read(), dtype=np.uint8)
            # reshape 成 (num_images, rows, cols)
            self.images = image_data.reshape(num_images, rows, cols, 1).astype(np.float32) / 255.0
        
        # 读取标签文件
        with gzip.open(label_filename, 'rb') as f:
            # 读取头部：魔数(4) + 标签数量(4)
            magic, num_labels = struct.unpack('>II', f.read(8))
            # 读取所有标签
            self.labels = np.frombuffer(f.read(), dtype=np.uint8)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        # 获取图像和标签
        img = self.images[index]
        label = self.labels[index]
        
        # 如果是单个索引，应用 transforms
        if isinstance(index, int):
            img = self.apply_transforms(img)
        
        return img, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.labels)
        ### END YOUR SOLUTION