import warnings
warnings.filterwarnings('ignore')
import IPython.display as display
import matplotlib.pyplot as plt
import paddle.fluid as fluid
from network import *
import numpy as np

'''定义超参数'''
img_dim = 64
output = "Output/"
G_DIMENSION = 100
use_gpu = False
dg_program = fluid.Program()

###定义生成器program
with fluid.program_guard(dg_program):
    noise = fluid.layers.data(name='noise', shape=[None,G_DIMENSION], dtype='float32')
    #label = np.ones(shape=[batch_size, G_DIMENSION], dtype='int64')
    # 噪声数据作为输入得到生成照片
    g_img = G(x=noise)
    g_program = dg_program.clone()
    g_program_test = dg_program.clone(for_test=True)

    # 判断生成图片为真实样本的概率
    dg_logit = D(g_img)

    # 计算生成图片被判别为真实样本的loss
    dg_loss = loss(
        x=dg_logit,
        label=fluid.layers.fill_constant_batch_size_like(input=noise, dtype='int64', shape=[-1,1], value=1)
    )

if use_gpu:
    exe = fluid.Executor(fluid.CUDAPlace(0))
else:
    exe = fluid.Executor(fluid.CPUPlace())
start_program = fluid.default_startup_program()
exe.run(start_program)
fluid.io.load_persistables(exe,'work/Model/G/',dg_program)
plt.figure(figsize=(25,6))
try:
    for i in range(G_DIMENSION):
        noise = np.random.uniform(low=0.0, high=1.0,size=[G_DIMENSION]).astype('float32')
        const_n = []
        for m in range(20):
            noise2 = noise.copy()
            noise2[i] = (m + 1) / 20
            const_n.append(noise2)
        const_n = np.array(const_n).astype('float32')
        #print(const_n)
        generated_image = exe.run(g_program, feed={'noise': const_n}, fetch_list=[g_img])[0]
        for j in range(20):
            image = generated_image[j].transpose()
            plt.subplot(1, 20, j + 1)
            plt.imshow(image)
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
        #plt.suptitle('Generated Image')
        plt.savefig('work/Generate/generated_' + str(i + 1), bbox_inches='tight')
        display.clear_output(wait=True)
        #plt.show()
except IOError:
    print(IOError)