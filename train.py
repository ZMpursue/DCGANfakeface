import IPython.display as display
import warnings
import paddle.optimizer as optim
import paddle
from network import *
import matplotlib.pyplot as plt
from dataloader import *
warnings.filterwarnings('ignore')

img_dim = 64
lr = 0.0002
epoch = 5
batch_size = 128
G_DIMENSION = 100
beta1 = 0.5
beta2 = 0.999
output_path = 'Output'
device = paddle.set_device('gpu')
paddle.disable_static(device)

real_label = 1.
fake_label = 0.

netD = Discriminator()
netG = Generator()
optimizerD = optim.Adam(parameters=netD.parameters(), learning_rate=lr, beta1=beta1, beta2=beta2)
optimizerG = optim.Adam(parameters=netG.parameters(), learning_rate=lr, beta1=beta1, beta2=beta2)

###训练过程
losses = [[], []]
# plt.ion()
now = 0
for pass_id in range(epoch):
    paddle.save(netG.state_dict(), "generator.params")
    # enumerate()函数将一个可遍历的数据对象组合成一个序列列表
    for batch_id, data in enumerate(train_loader()):
        # 训练判别器
        optimizerD.clear_grad()
        real_cpu = data[0]
        label = paddle.full((batch_size, 1, 1, 1), real_label, dtype='float32')
        output = netD(real_cpu)
        errD_real = loss(output, label)
        errD_real.backward()
        optimizerD.step()
        optimizerD.clear_grad()

        noise = paddle.randn([batch_size, G_DIMENSION, 1, 1], 'float32')
        fake = netG(noise)
        label = paddle.full((batch_size, 1, 1, 1), fake_label, dtype='float32')
        output = netD(fake.detach())
        errD_fake = loss(output, label)
        errD_fake.backward()
        optimizerD.step()
        optimizerD.clear_grad()

        errD = errD_real + errD_fake

        losses[0].append(errD.numpy()[0])
        ###训练生成器
        optimizerG.clear_grad()
        noise = paddle.randn([batch_size, G_DIMENSION, 1, 1], 'float32')
        fake = netG(noise)
        label = paddle.full((batch_size, 1, 1, 1), real_label, dtype=np.float32, )
        output = netD(fake)
        errG = loss(output, label)
        errG.backward()
        optimizerG.step()
        optimizerG.clear_grad()

        losses[1].append(errG.numpy()[0])
        if batch_id % 100 == 0:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            # 每轮的生成结果
            generated_image = netG(noise).numpy()
            imgs = []
            plt.figure(figsize=(15, 15))
            try:
                for i in range(100):
                    image = generated_image[i].transpose()
                    image = np.where(image > 0, image, 0)
                    plt.subplot(10, 10, i + 1)
                    plt.imshow(image, vmin=-1, vmax=1)
                    plt.axis('off')
                    plt.xticks([])
                    plt.yticks([])
                    plt.subplots_adjust(wspace=0.1, hspace=0.1)
                msg = 'Epoch ID={0} Batch ID={1} \n\n D-Loss={2} G-Loss={3}'.format(pass_id, batch_id, errD.numpy()[0],
                                                                                    errG.numpy()[0])
                plt.suptitle(msg, fontsize=20)
                plt.draw()
                plt.savefig('{}/{:04d}_{:04d}.png'.format(output_path, pass_id, batch_id), bbox_inches='tight')
                plt.pause(0.01)
                display.clear_output(wait=True)
            except IOError:
                print(IOError)

plt.close()
plt.figure(figsize=(15, 6))
x = np.arange(len(losses[0]))
plt.title('Generator and Discriminator Loss During Training')
plt.xlabel('Number of Batch')
plt.plot(x,np.array(losses[0]),label='D Loss')
plt.plot(x,np.array(losses[1]),label='G Loss')
plt.legend()
plt.savefig('Generator and Discriminator Loss During Training.png')
plt.show()