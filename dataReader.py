import os
import cv2
import numpy as np
import paddle.dataset as dataset
from skimage import io,color,transform
import matplotlib.pyplot as plt
import math
import time
import paddle
import paddle.fluid as fluid
import six

img_dim = 128

'''准备数据，定义Reader()'''
PATH = './imgs/'
TEST = './imgs/'
class DataGenerater:
    def __init__(self):
        '''初始化'''
        self.datalist = os.listdir(PATH)
        self.testlist = os.listdir(TEST)

    def load(self, image):
        '''读取图片'''
        img = io.imread(image)
        img = transform.resize(img,(img_dim,img_dim))
        img = img.transpose()
        img = img.astype('float32')
        return img

    def create_train_reader(self):
        '''给dataset定义reader'''

        def reader():
            for img in self.datalist:
                #print(img)
                try:
                    i = self.load(PATH + img)
                    yield i.astype('float32')
                except Exception as e:
                    print(e)
        return reader

    def create_test_reader(self,):
        '''给test定义reader'''
        def reader():
            for img in self.datalist:
                #print(img)
                try:
                    i = self.load(PATH + img)
                    yield i.astype('float32')
                except Exception as e:
                    print(e)
        return reader

def train(batch_sizes = 32):
    reader = DataGenerater().create_train_reader()
    return reader

def test():
    reader = DataGenerater().create_test_reader()
    return reader

if __name__ == "__main__":
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            reader=train(), buf_size=128
        ),
        batch_size=128
    )
    for batch_id, data in enumerate(train_reader()):

        plt.figure(figsize=(15, 15))
        try:
            for i in range(100):
                image = data[i].transpose()
                plt.subplot(10, 10, i + 1)
                plt.imshow(image, vmin=-1, vmax=1)
                plt.axis('off')
                plt.xticks([])
                plt.yticks([])
                plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.suptitle('\n Training Images', fontsize=30)
            plt.show()
            break
        except IOError:
            print(IOError)