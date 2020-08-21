import os
import cv2
import sys
import time
import shutil
import math
import random
import threading
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.contrib.model_stat import summary

from paddle.fluid.initializer import MSRA
from paddle.fluid.initializer import Uniform
from paddle.fluid.param_attr import ParamAttr
from PIL import Image
from PIL import ImageEnhance

class ResNet(object):
    """
    resnet的网络结构类
    """

    def __init__(self, layers=50):
        """
        resnet的网络构造函数
        :param layers: 网络层数
        """
        self.layers = layers

    def name(self):
        """
        获取网络结构名字
        :return:
        """
        return 'resnet'

    def net(self, input, class_dim=1000):
        """
        构建网络结构
        :param input: 输入图片
        :param class_dim: 分类类别
        :return:
        """
        layers = self.layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_filters = [64, 128, 256, 512]

        conv = self.conv_bn_layer(
            input=input,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu',
            name="conv1")
        conv = fluid.layers.pool2d(
            input=conv,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')

        for block in range(len(depth)):
            for i in range(depth[block]):
                if layers in [101, 152] and block == 2:
                    if i == 0:
                        conv_name = "res" + str(block + 2) + "a"
                    else:
                        conv_name = "res" + str(block + 2) + "b" + str(i)
                else:
                    conv_name = "res" + str(block + 2) + chr(97 + i)
                conv = self.bottleneck_block(
                    input=conv,
                    num_filters=num_filters[block],
                    stride=2 if i == 0 and block != 0 else 1,
                    name=conv_name)

        pool = fluid.layers.pool2d(input=conv, pool_size=7, pool_type='avg', global_pooling=True)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        out = fluid.layers.fc(input=pool,
                              size=class_dim,
                              act='softmax',
                              param_attr=fluid.param_attr.ParamAttr(initializer=Uniform(-stdv, stdv)))
        return out

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None,
                      name=None):
        """
        便捷型卷积结构，包含了batch_normal处理
        :param input: 输入图片
        :param num_filters: 卷积核个数
        :param filter_size: 卷积核大小
        :param stride: 平移
        :param groups: 分组
        :param act: 激活函数
        :param name: 卷积层名字
        :return:
        """
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            param_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False,
            name=name + '.conv2d.output.1')
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        return fluid.layers.batch_norm(
            input=conv,
            act=act,
            is_test=True,
            name=bn_name + '.output.1',
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance', )

    def shortcut(self, input, ch_out, stride, name):
        """
        转换结构，转换输入和输出一致，方便最后的短链接结构
        :param input:
        :param ch_out:
        :param stride:
        :param name:
        :return:
        """
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            return self.conv_bn_layer(input, ch_out, 1, stride, name=name)
        else:
            return input

    def bottleneck_block(self, input, num_filters, stride, name):
        """
        resnet的短路链接结构中的一种，采用压缩方式先降维，卷积后再升维
        利用转换结构将输入变成瓶颈卷积一样的尺寸，最后将两者按照位相加，完成短路链接
        :param input:
        :param num_filters:
        :param stride:
        :param name:
        :return:
        """
        conv0 = self.conv_bn_layer(
            input=input,
            num_filters=num_filters,
            filter_size=1,
            act='relu',
            name=name + "_branch2a")
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu',
            name=name + "_branch2b")
        conv2 = self.conv_bn_layer(
            input=conv1,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None,
            name=name + "_branch2c")

        short = self.shortcut(
            input, num_filters * 4, stride, name=name + "_branch1")

        return fluid.layers.elementwise_add(
            x=short, y=conv2, act='relu', name=name + ".add.output.5")

# Resnet 预测
class ResnetDecode(object):
    def __init__(self, input_shape, model_path, file_path, use_gpu):
        self.main_program = fluid.Program()
        self.startup_program = fluid.Program()
        
        self.input_shape = input_shape
        self.all_classes = self.get_classes(file_path)
        self.num_classes = len(self.all_classes)
        self.resnet =ResNet(layers=50)
        
        with fluid.program_guard(self.main_program, self.startup_program):
            inputs = fluid.layers.data(name='input', shape=[-1, 3, 224, 224], append_batch_size=False, dtype='float32')
            self.y = self.resnet.net(input=inputs, class_dim=2)
        
        gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
        place = fluid.CUDAPlace(gpu_id) if use_gpu else fluid.CPUPlace()
        self.exe = fluid.Executor(place)
#         self.exe.run(fluid.default_startup_program())
        self.exe.run(self.startup_program)
        
        fluid.io.load_persistables(self.exe, model_path, main_program=self.startup_program)
        
    def get_classes(self, file):
        with open(file) as f:
            class_names = f.readlines()
        
        class_names = [c.strip() for c in class_names]
        return class_names
    
    def predict(self, image, shape):
#         start = time.time()
        image = image.astype(np.float32)
        image -= [127.5, 127.5, 127.5]
        image *= 0.007843
        pimage = self.process_image(image, self.input_shape)
        pimage = pimage.transpose(0, 3, 1, 2)
#         outs = self.exe.run(fluid.default_main_program(), feed={'vgg_input': pimage}, fetch_list=[self.y])
        outs = self.exe.run(self.main_program, feed={'input': pimage}, fetch_list=[self.y])
#         print("Image process times: {0:.6f}s" .format(time.time() - start))
        
        return outs
    
    def process_image(self, img, shape):
        h, w = img.shape[:2]
        M, h_out, w_out = self.training_transform(h, w, shape[0], shape[1])
        # 填充黑边缩放
        letterbox = cv2.warpAffine(img, M, (w_out, h_out))
        pimage = np.float32(letterbox) / 255.
        pimage = np.expand_dims(pimage, axis=0)
        return pimage
    
    def training_transform(self, height, width, output_height, output_width):
        height_scale, width_scale = output_height / height, output_width / width
        scale = min(height_scale, width_scale)
        resize_height, resize_width = round(height * scale), round(width * scale)
        pad_top = (output_height - resize_height) // 2
        pad_left = (output_width - resize_width) // 2
        A = np.float32([[scale, 0.0], [0.0, scale]])
        B = np.float32([[pad_left], [pad_top]])
        M = np.hstack([A, B])
        return M, output_height, output_width
    
    def draw(self, image, boxes, scores, classes, second_classes):
        image_h, image_w, _ = image.shape
        # 定义颜色
        hsv_tuples = [(1.0 * x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        random.seed(0)
        random.shuffle(colors)
        random.seed(None)

        for box, score, cl, action in zip(boxes, scores, classes, second_classes):
            x0, y0, x1, y1 = box
            left = max(0, np.floor(x0 + 0.5).astype(int))
            top = max(0, np.floor(y0 + 0.5).astype(int))
            right = min(image.shape[1], np.floor(x1 + 0.5).astype(int))
            bottom = min(image.shape[0], np.floor(y1 + 0.5).astype(int))
            bbox_color = colors[action]
            bbox_thick = 1 if min(image_h, image_w) < 400 else 2

            # 截图
#             crop_image = image[top:bottom, left:right]

            cv2.rectangle(image, (left, top), (right, bottom), bbox_color, bbox_thick)
#             bbox_mess = '%s: %.2f' % (self.all_classes[cl], score)
            bbox_mess = 'person: %s' % (self.all_classes[action])
            t_size = cv2.getTextSize(bbox_mess, 0, 0.5, thickness=1)[0]
            cv2.rectangle(image, (left, top), (left + t_size[0], top - t_size[1] - 3), bbox_color, -1)
            cv2.putText(image, bbox_mess, (left, top - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)