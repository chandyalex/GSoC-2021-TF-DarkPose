from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import tensorflow.keras.backend as K
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import MaxPool2D, Conv2DTranspose, Input,Flatten
from tensorflow.keras.layers import UpSampling2D, add, concatenate

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)




def conv3x3(self, x, out_filters, strides=(1, 1)):
    padding=keras.layers.ZeroPadding2D(padding=1)
    x=padding(x)
    x = Conv2D(out_filters, 3, padding='same', strides=strides, use_bias=False, kernel_initializer='he_normal')(x)
    return x


class basic_Block(tensorflow.keras.Model):
    expansion = 1
    def __init__(self,inplanes, planes, strides=(1, 1), with_downsample=False):
        super(basic_Block, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, strides)
        self.bn1 = BatchNormalization(input_shape=(planes,planes,), momentum=BN_MOMENTUM)
        self.relu = Activation('relu')
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNormalization(input_shape=(planes,planes,), momentum=BN_MOMENTUM)
        self.downsample = with_downsample
        self.stride = strides
    def call(self, x):
        residual=x
        # inplanes =x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(out)

        # if self.downsample==True:
        #     residual = Conv2D(planes, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(inplanes)
        #     residual = BatchNormalization(input_shape=(planes,))(residual)
        #     x = add([x, residual])
        # else:
        #     x = add([x, inplanes])
        x += residual

        x = self.relu(x)
        return x


class bottleneck_Block(tensorflow.keras.Model):
    expansion = 4
    def __init__(self,inplanes, planes, strides=(1, 1), with_downsample=False):
        super(bottleneck_Block, self).__init__()


        self.conv1 = Conv2D(input_shape=(inplanes,inplanes,), filters=planes, kernel_size=(1,1),
                                            padding='same',use_bias=False)
        self.bn1 = BatchNormalization(input_shape=(planes,planes,), momentum=BN_MOMENTUM)
        self.conv2 = Conv2D(input_shape=(planes,planes,), filters=planes, kernel_size=(3,3), strides=strides,
                               padding='same',use_bias=False)
        self.bn2 = BatchNormalization(input_shape=(planes,planes,), momentum=BN_MOMENTUM)
        self.conv3 = Conv2D(input_shape=(planes,planes,), filters=planes * self.expansion, kernel_size=(1,1),
                               padding='same',use_bias=False)
        self.bn3 = BatchNormalization(input_shape=(planes * self.expansion,planes * self.expansion,),
                                  momentum=BN_MOMENTUM)
        self.relu = Activation('relu')
        self.downsample = with_downsample
        self.stride = strides

    def call(self, x):
        padding=keras.layers.ZeroPadding2D(padding=1)

        print(x.shape)
        residual = x
        # inplanes=x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x=padding(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        print(x.shape)

        # if self.downsample==True:
        #     residual = Conv2D(planes, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
        #     residual = BatchNormalization(input_shape=(planes,))(residual)
        #     x = add([x, residual])
        # else:
        #     x = add([x, inplanes])
        x=keras.layers.Concatenate(axis=1)([x, residual])


        x = self.relu(x)
        return x

# def bottleneck_Block(self,input, out_filters, strides=(1, 1), with_downsample=False):
#
#     expansion = 4
#     de_filters = int(out_filters / expansion)
#
#     x = Conv2D(de_filters, 1, use_bias=False, kernel_initializer='he_normal')(input)
#     x = BatchNormalization(axis=3)(x)
#     x = Activation('relu')(x)
#
#     x = Conv2D(de_filters, 3, strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
#     x = BatchNormalization(axis=3)(x)
#     x = Activation('relu')(x)
#
#     x = Conv2D(out_filters, 1, use_bias=False, kernel_initializer='he_normal')(x)
#     x = BatchNormalization(axis=3)(x)
#
#     if with_downsample:
#         residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
#         residual = BatchNormalization(axis=3)(residual)
#         x = add([x, residual])
#     else:
#         x = add([x, input])
#
#     x = Activation('relu')(x)
#     return x

class PoseResNet(tensorflow.keras.Model):
    def __init__(self, block, layers, cfg, **kwargs):

        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS

        super(PoseResNet, self).__init__()




        self.conv1 = Conv2D(input_shape=(3,),filters=64, kernel_size=7,
                                strides=(2,2),padding="same",use_bias=True)
        self.padding=keras.layers.ZeroPadding2D(padding=(3,3))
        # self.conv1 = Conv2D(64,kernel_size= 7,strides= 2,
        #                                               activation = "relu",
        #                                               data_format="channels_last",
        #                                               use_bias=True,
        #                                               input_shape= (256,256, 3))
        self.bn1 = BatchNormalization(input_shape=(64,64,), momentum=BN_MOMENTUM)
        self.relu = Activation('relu')
        self.maxpool = MaxPool2D(pool_size=3, strides=2,padding='same')

        self.layer1 = self._make_layer(block, 64, layers[0],stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        self.final_layer = Conv2D(
            input_shape=(extra.NUM_DECONV_FILTERS[-1],),
            filters=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            padding='same',
            strides=1,
        )

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                Conv2D(input_shape=(self.inplanes,self.inplanes,3), filters = planes * block.expansion,
                          kernel_size=1, strides=stride,padding='same', use_bias=False),
                BatchNormalization(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return Sequential(layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        layers_new=[]
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]

            layers.append(
                Conv2DTranspose(
                    input_shape=(self.inplanes,self.inplanes,3),
                    filters=planes,
                    kernel_size=kernel,
                    strides=2,
                    padding='same',
                    use_bias=self.deconv_with_bias))
            layers.append(keras.layers.ZeroPadding2D(padding=(padding,padding)))
            layers.append(BatchNormalization(input_shape=(planes,planes,),momentum=BN_MOMENTUM))
            layers.append(Activation('relu'))
            self.inplanes = planes
        print(layers)

        return Sequential(layers)


    def call(self, x):

        padding=keras.layers.ZeroPadding2D(padding=1)
        x = self.conv1(x)
        x= self.padding(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        x = padding(x)
        x = self.final_layer(x)

        return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                    # nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
resnet_spec = {
    18: (basic_Block, [2, 2, 2, 2]),
    34: (basic_Block, [3, 4, 6, 3]),
    50: (bottleneck_Block, [3, 4, 6, 3]),
    101: (bottleneck_Block, [3, 4, 23, 3]),
    152: (bottleneck_Block, [3, 8, 36, 3])
}

def get_pose_net(cfg, is_train, **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS

    block_class, layers = resnet_spec[num_layers]

    model = PoseResNet(block_class, layers, cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model
