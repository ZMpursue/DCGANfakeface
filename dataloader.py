import os
import cv2
import numpy as np
from skimage import io, color, transform
import matplotlib.pyplot as plt
import math
import time
import paddle
from paddle.io import Dataset
import six
from PIL import Image as PilImage
import paddle.fluid as fluid
from paddle.static import InputSpec

paddle.enable_static()
img_dim = 64

'''准备数据，定义Reader()'''
PATH = 'imgs/'


class DataGenerater(Dataset):
    """
    数据集定义
    """

    def __init__(self, path=PATH):
        """
        构造函数
        """
        super(DataGenerater, self).__init__()
        self.dir = path
        self.datalist = os.listdir(PATH)
        self.image_size = (img_dim, img_dim)

    # 每次迭代时返回数据和对应的标签
    def __getitem__(self, idx):
        return self._load_img(self.dir + self.datalist[idx])

    # 返回整个数据集的总数
    def __len__(self):
        return len(self.datalist)

    def _load_img(self, path):
        """
        统一的图像处理接口封装，用于规整图像大小和通道
        """
        try:
            img = io.imread(path)
            img = transform.resize(img, self.image_size)
            img = img.transpose()
            img = img.astype('float32')
        except Exception as e:
            print(e)
        return img


train_dataset = DataGenerater()
img = fluid.layers.data(name='img', shape=[None,3,img_dim,img_dim], dtype='float32')
train_loader = paddle.io.DataLoader(
    train_dataset,
    places=paddle.CPUPlace(),
    feed_list = [img],
    batch_size=128,
    shuffle=True,
    num_workers=2,
    use_buffer_reader=True,
    use_shared_memory=False,
    )

for batch_id, data in enumerate(train_loader()):
    plt.figure(figsize=(15,15))
    try:
        for i in range(100):
            image = np.array(data[0]['img'][i])[0].transpose((2,1,0))
            plt.subplot(10, 10, i + 1)
            plt.imshow(image, vmin=-1, vmax=1)
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.suptitle('\n Training Images',fontsize=30)
        plt.show()
        break
    except IOError:
        print(IOError)
