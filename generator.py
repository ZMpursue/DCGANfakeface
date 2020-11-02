import paddle
from network import *
import matplotlib.pyplot as plt

device = paddle.set_device('gpu')
paddle.disable_static(device)
try:
    generate = Generator()
    state_dict = paddle.load("generator.params")
    generate.set_state_dict(state_dict)
    noise = paddle.randn([100,100,1,1],'float32')
    generated_image = generate(noise).numpy()
    for j in range(100):
        image = generated_image[j].transpose()
        plt.figure(figsize=(4,4))
        plt.imshow(image)
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.savefig('Generate/generated_' + str(j + 1), bbox_inches='tight')
        plt.close()
except IOError:
    print(IOError)