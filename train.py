#!/usr/bin/env python3

class Train(object):
    def __init__(self):
        pass

    def _train(self):
        pass

    def process(self):
        pass

    def _save_parameter(self):
        pass

    def _cal_loss(self):
        pass

    def _test(self):
        pass

if __name__ == "__main__":
    pass


def ex_extended_model(input_layer, use_batchnorm=False, is_training=True, activation=tf.nn.relu, lr_mult=1):
    # kernel_dim = [512, 256, 512, 128, 256, 128, 256, 128, 256]
    conv_6_1 = convBNLayer(input_layer, use_batchnorm, is_training, 512, 256, 1, 1, name="conv_6_1", activation=activation)
    conv_6_2 = convBNLayer(conv_6_1, use_batchnorm, is_training, 256, 512, 3, 2, name="conv_6_2", activation=activation)
    conv_7_1 = convBNLayer(conv_6_2, use_batchnorm, is_training, 512, 128, 1, 1, name="conv_7_1", activation=activation)
    conv_7_2 = convBNLayer(conv_7_1, use_batchnorm, is_training, 128, 256, 3, 2, name="conv_7_2", activation=activation)
    conv_8_1 = convBNLayer(conv_7_2, use_batchnorm, is_training, 256, 128, 1, 1, name="conv_8_1", activation=activation)
    conv_8_2 = convBNLayer(conv_8_1, use_batchnorm, is_training, 128, 256, 3, 1, name="conv_8_2", activation=activation, padding="VALID")
    conv_9_1 = convBNLayer(conv_8_2, use_batchnorm, is_training, 256, 128, 1, 1, name="conv_9_1", activation=activation)
    conv_9_2 = convBNLayer(conv_9_1, use_batchnorm, is_training, 128, 256, 3, 1, name="conv_9_2", activation=activation, padding="VALID")
    return conv_9_2
