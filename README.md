# 通过DCGAN实现人脸图像生成

本教程将通过一个示例对DCGAN进行介绍。在向其展示许多真实人脸照片（数据集：[Celeb-A Face](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)）后，我们将训练一个生成对抗网络（GAN）来产生新人脸。本文将对该实现进行详尽的解释，并阐明此模型的工作方式和原因。并不需要过多专业知识，但是可能需要新手花一些时间来理解的模型训练的实际情况。为了节省时间，请尽量选择GPU进行训练。


## 1 简介
本项目基于paddlepaddle，结合生成对抗网络（DCGAN）,通过弱监督学习的方式，训练生成真实人脸照片

### 1.1 什么是GAN？

生成对抗网络（Generative Adversarial Network [1]，简称GAN）是非监督式学习的一种方法，通过让两个神经网络相互博弈的方式进行学习。该方法最初由 lan·Goodfellow 等人于2014年提出，原论文见 [Generative Adversarial Network](https://arxiv.org/abs/1406.2661)。

  生成对抗网络由一个生成网络与一个判别网络组成。生成网络从潜在空间（latent space）中随机采样作为输入，其输出结果需要尽量模仿训练集中的真实样本。判别网络的输入为真实样本或生成网络的输出，其目的是将生成网络的输出从真实样本中尽可能分辨出来。而生成网络则要尽可能地欺骗判别网络。两个网络相互对抗、不断调整参数，其目的是将生成网络生成的样本和真实样本尽可能的区分开[2] ）。 

让x是代表图像的数据。D(x)是判别器网络，输出的概率为x来自训练数据还是生成器。在这里输入D(x)的x是CHW大小为3x128x128的图像。使得x来自训练数据时D(x)尽量接近1，x来自生成器时D(x)尽量接近0。D(x)也可以被认为是传统的二进制分类器。

对于生成器网络，z为从标准正态分布采样的潜在空间向量。G(z)表示生成器函数，该函数将矢量z映射到数据空间。生成器的目标是拟合训练数据![img](https://latex.codecogs.com/svg.latex?p_{data})的分布，以便可以从该估计分布中生成假样本![img](https://latex.codecogs.com/svg.latex?p_g)。

所以，![img](https://latex.codecogs.com/svg.latex?D(G(z)))是生成器G输出是真实的图像的概率。如Goodfellow的论文所述，D和G玩一个minimax游戏，其中D尝试最大化其正确分类真假的可能性![img](https://latex.codecogs.com/svg.latex?logD(x))，以及G试图最小化以下可能性D会预测其输出是假的![img](https://latex.codecogs.com/svg.latex?log(1-D(G(x))))。

GAN的损失函数可表示为：

![img](https://latex.codecogs.com/svg.latex?\underset{G}{\text{min}}%20\underset{D}{\text{max}}V(D,G)%20=%20\mathbb{E}_{x\sim%20p_{data}(x)}\big[logD(x)\big]%20+%20\mathbb{E}_{z\sim%20p_{z}(z)}\big[log(1-D(G(z)))\big])

从理论上讲，此minimax游戏的解决方案是![img](https://latex.codecogs.com/svg.latex?p_g%20=%20p_{data})，鉴别者会盲目猜测输入是真实的还是假的。但是，GAN的收敛理论仍在积极研究中，实际上GAN常常会遇到梯度消失/爆炸问题。  
生成对抗网络常用于生成以假乱真的图片。此外，该方法还被用于生成视频、三维物体模型等。


### 1.2 什么是DCGAN？

DCGAN是深层卷积网络与 GAN 的结合，其基本原理与 GAN 相同，只是将生成网络和判别网络用两个卷积网络（CNN）替代。为了提高生成样本的质量和网络的收敛速度，论文中的 DCGAN 在网络结构上进行了一些改进：

 * 取消 pooling 层：在网络中，所有的pooling层使用步幅卷积（strided convolutions）(判别器)和微步幅度卷积（fractional-strided convolutions）(生成器)进行替换。
 * 加入 batch normalization：在生成器和判别器中均加入batchnorm。
 * 使用全卷积网络：去掉了FC层，以实现更深的网络结构。
 * 激活函数：在生成器（G）中，最后一层使用Tanh函数，其余层采用 ReLu 函数 ; 判别器（D）中都采用LeakyReLu。  



## 2 环境设置及数据集

环境：paddlepaddle、scikit-image、numpy、matplotlib  

在本教程中，我们将使用[Celeb-A Faces](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)数据集，该数据集可以在链接的网站或[AI Studio](https://aistudio.baidu.com/aistudio/datasetdetail/39207)中下载。数据集将下载为名为img_align_celeba.zip的文件。下载后，并将zip文件解压缩到该目录中。  
img_align_celeba目录结构应为： 
```
/path/to/project  
	-> img_align_celeba  
		-> 188242.jpg  
		-> 173822.jpg  
		-> 284702.jpg  
		-> 537394.jpg  
		...
```

### 2.3 数据集预处理(./pretreat.py)
多线程处理，以裁切坐标(0,10)和(64,74)，将输入网络的图片裁切到64*64。


## 3 模型组网
### 3.1 定义数据预处理工具-DataReader(./dataReader.py)
具体参考[DataReader教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/data_preparing/static_mode/reader_cn.html)


### 3.2 测试DataReader并输出图片


### 3.3 权重初始化(./weightInit.py)
在 DCGAN 论文中，作者指定所有模型权重应从均值为0、标准差为0.02的正态分布中随机初始化。
在paddle.nn中，调用fluid.nn.initializer.Normal实现initialize设置


```python
import paddle.fluid as fluid
conv_initializer=paddle.nn.initializer.Normal(mean=0.0, std=0.02)
bn_initializer=paddle.nn.initializer.Normal(mean=1.0, std=0.02)
```

### 3.4 定义网络功能模块(./network.py)
包括卷积层、转置卷积层、$BatchNorm$层、全连接层和卷积BatchNorm组 
* 转置卷积层：在生成器中，需要用随机采样值生成全尺寸图像，dcgan使用转置卷积层进行上采样，在Fluid中，调用 fluid.layers.conv2d_transpose 实现转置卷积。
* BatchNorm层：调用 fluid.layers.batch_norm 接口实现bn层，激活函数默认使用ReLu。
* 卷积层：调用 fluid.nets.simple_img_conv_pool 实现卷积池化组，卷积核大小为5x5，池化窗口大小为2x2，窗口滑动步长为2，激活函数类型由具体网络结构指定。

### 3.5 判别器(./network.py)
如上文所述，生成器D是一个二进制分类网络，它以图像作为输入，输出图像是真实的（相对应G生成的假样本）的概率。输入Shape为[3,64,64]的RGB图像，通过一系列的Conv2d，BatchNorm2d和LeakyReLU层对其进行处理，然后通过全连接层输出的神经元个数为2，对应两个标签的预测概率。

* 将BatchNorm批归一化中momentum参数设置为0.5
* 将判别器(D)激活函数leaky_relu的alpha参数设置为0.2

> 输入:  为大小64x64的RGB三通道图片  
> 输出:  经过一层全连接层最后为shape为[batch_size,2]的Tensor

### 3.6 生成器(./network.py)
生成器G旨在映射潜在空间矢量z到数据空间。由于我们的数据是图像，因此转换$z$到数据空间意味着最终创建具有与训练图像相同大小[3,64,64]的RGB图像。在网络设计中，这是通过一系列二维卷积转置层来完成的，每个层都与BatchNorm层和ReLu激活函数。生成器的输出通过tanh函数输出，以使其返回到输入数据范围[−1,1]。值得注意的是，在卷积转置层之后存在BatchNorm函数，因为这是DCGAN论文的关键改进。这些层有助于训练过程中的梯度更好地流动。

**生成器网络结构**  
![](https://ai-studio-static-online.cdn.bcebos.com/ca0434dd681849338b1c0c46285616f72add01ab894b4e95848daecd5a72e3cb)

* 将BatchNorm批归一化中momentum参数设置为0.5

> 输入:Tensor的Shape为[batch_size,100]其中每个数值大小为0~1之间的float32随机数  
> 输出:3x128x128RGB三通道图片

### 3.7 损失函数(./network.py)
选用Softmax_with_cross_entropy,公式如下:

![img](https://latex.codecogs.com/svg.latex?Out%20=%20-1%20*%20(label%20*%20log(input)%20+%20(1%20-%20label)%20*%20log(1%20-%20input)))

## 4 模型训练(./train.py)
 设置的超参数为：
 * 学习率：0.0002
 * 输入图片长和宽：64
 * Epoch: 8
 * Mini-Batch：128
 * 输入Tensor长度：100
 * Adam：Beta1：0.5，Beta2：0.999

训练过程中的每一次迭代，生成器和判别器分别设置自己的迭代次数。为了避免判别器快速收敛到0，本教程默认每迭代一次，训练一次判别器，两次生成器。
## 5 模型评估
### 生成器$G$和判别器$D$的损失与迭代变化图
![](https://ai-studio-static-online.cdn.bcebos.com/8c94b567738c423ba5a421a40ccf5f57fd8defc3b62d46f28cc5d3aa3439bf42)

### 对比真实人脸图像（图一）和生成人脸图像（图二）
#### 图一
![](https://ai-studio-static-online.cdn.bcebos.com/622fff7b67a240ff8a1fceb209fc27445c929099365c44e8bd006569bf26e0a6)
### 图二
![](https://ai-studio-static-online.cdn.bcebos.com/ed33cf82762d4ae79feef82b77604273303cc38b8f5d4728a632b492f2665ed4)


## 6 模型预测(./generator.py)
### 输入随机数让生成器G生成随机人脸

## 8 项目总结
简单介绍了一下DCGAN的原理，通过对原项目的改进和优化，一步一步依次对生成器和判别器以及训练过程进行介绍。
DCGAN采用一个随机噪声向量作为输入，输入通过与CNN类似但是相反的结构，将输入放大成二维数据。采用这种结构的生成模型和CNN结构的判别模型，DCGAN在图片生成上可以达到相当可观的效果。本案例中，我们利用DCGAN生成了人脸照片，您可以尝试更换数据集生成符合个人需求的图片，或尝试修改网络结构观察不一样的生成效果。

## 9 参考文献
[1] Goodfellow, Ian J.; Pouget-Abadie, Jean; Mirza, Mehdi; Xu, Bing; Warde-Farley, David; Ozair, Sherjil; Courville, Aaron; Bengio, Yoshua. Generative Adversarial Networks. 2014. arXiv:1406.2661 [stat.ML].

[2] Andrej Karpathy, Pieter Abbeel, Greg Brockman, Peter Chen, Vicki Cheung, Rocky Duan, Ian Goodfellow, Durk Kingma, Jonathan Ho, Rein Houthooft, Tim Salimans, John Schulman, Ilya Sutskever, And Wojciech Zaremba, Generative Models, OpenAI, [April 7, 2016]

[3] alimans, Tim; Goodfellow, Ian; Zaremba, Wojciech; Cheung, Vicki; Radford, Alec; Chen, Xi. Improved Techniques for Training GANs. 2016. arXiv:1606.03498 [cs.LG].

[4] Radford A, Metz L, Chintala S. Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks[J]. Computer Science, 2015.

[5]Kingma D , Ba J . Adam: A Method for Stochastic Optimization[J]. Computer ence, 2014.
