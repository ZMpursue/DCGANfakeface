from PIL import Image
import os.path
import os
import threading
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

'''多线程将图片缩放后再裁切到64*64分辨率'''
# 裁切图片宽度
w = 64
# 裁切图片高度
h = 64
# 裁切点横坐标(以图片左上角为原点)
x = 0
# 裁切点纵坐标
y = 10


def cutArray(l, num):
    avg = len(l) / float(num)
    o = []
    last = 0.0

    while last < len(l):
        o.append(l[int(last):int(last + avg)])
        last += avg

    return o


def convertjpg(jpgfile, outdir, width=w, height=h):
    img = Image.open(jpgfile)
    (l, h) = img.size
    rate = min(l, h) / width
    try:
        img = img.resize((int(l // rate), int(h // rate)), Image.BILINEAR)
        img = img.crop((x, y, width + x, height + y))
        img.save(os.path.join(outdir, os.path.basename(jpgfile)))
    except Exception as e:
        print(e)


class thread(threading.Thread):
    def __init__(self, threadID, inpath, outpath, files):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.inpath = inpath
        self.outpath = outpath
        self.files = files

    def run(self):
        count = 0
        try:
            for file in self.files:
                convertjpg(self.inpath + file, self.outpath)
                count = count + 1
        except Exception as e:
            print(e)
        print('已处理图片数量：' + str(count))


if __name__ == '__main__':
    inpath = './img_align_celeba/'
    outpath = './imgs/'
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    files = os.listdir(inpath)
    files = cutArray(files, 8)
    T1 = thread(1, inpath, outpath, files[0])
    T2 = thread(2, inpath, outpath, files[1])
    T3 = thread(3, inpath, outpath, files[2])
    T4 = thread(4, inpath, outpath, files[3])
    T5 = thread(5, inpath, outpath, files[4])
    T6 = thread(6, inpath, outpath, files[5])
    T7 = thread(7, inpath, outpath, files[6])
    T8 = thread(8, inpath, outpath, files[7])

    T1.start()
    T2.start()
    T3.start()
    T4.start()
    T5.start()
    T6.start()
    T7.start()
    T8.start()

    T1.join()
    T2.join()
    T3.join()
    T4.join()
    T5.join()
    T6.join()
    T7.join()
    T8.join()
