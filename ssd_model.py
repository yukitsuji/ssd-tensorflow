#!/usr/bin/env python3
import sys
sys.path.append("/Users/tsujiyuuki/env_python/code/my_code/Data_Augmentation")
import numpy as np
from base_vgg16 import Vgg16 as Vgg
import tensorflow as tf
from network_util import batch_norm, convBNLayer, maxpool2d

class BasicNetwork(object):
    def __init__(self):
        pass

    def build_model(self, input_layer, use_batchnorm=False, is_training=True, activation=tf.nn.relu, \
                    rate=1, atrous=False, implement_atrous=False):
        if implement_atrous:
            if atrous:
                self.pool_5 = maxpool2d(input_layer, kernel=3, stride=1, name="pool5", padding="SAME")
            else:
                self.pool_5 = maxpool2d(input_layer, kernel=2, stride=2, name="pool5", padding="SAME") #TODO: padding is valid or same

            kernel_size = 3
            if atrous:
                rate *= 6
                # pad = int(((kernel_size + (rate - 1) * (kernel_size - 1)) - 1) / 2)
                self.conv_6 = convBNLayer(self.pool_5, use_batchnorm, is_training, 512, 1024, kernel_size, 1, \
                                          name="conv_6", activation=tf.nn.relu, atrous=True, rate=rate)
            else:
                rate *= 3
                # pad = int(((kernel_size + (rate - 1) * (kernel_size - 1)) - 1) / 2)
                self.conv_6 = convBNLayer(self.pool_5, use_batchnorm, is_training, 512, 1024, kernel_size, 1, \
                                          name="conv_6", activation=tf.nn.relu, atrous=True, rate=rate)
        else:
            self.pool_5 = maxpool2d(input_layer, kernel=3, stride=1, name="pool5", padding="SAME")
            self.conv_6 = convBNLayer(self.pool_5, use_batchnorm, is_training, 512, 1024, 3, 1, \
                                      name="conv_6", activation=tf.nn.relu, atrous=False, rate=rate)

        self.conv_7 = convBNLayer(self.conv_6, use_batchnorm, is_training, 1024, 1024, 1, 1, name="conv_7", activation=activation)
        self.conv_8_1 = convBNLayer(self.conv_7, use_batchnorm, is_training, 1024, 256, 1, 1, name="conv_8_1", activation=activation)
        self.conv_8_2 = convBNLayer(self.conv_8_1, use_batchnorm, is_training, 256, 512, 3, 2, name="conv_8_2", activation=activation)
        self.conv_9_1 = convBNLayer(self.conv_8_2, use_batchnorm, is_training, 512, 128, 1, 1, name="conv_9_1", activation=activation)
        self.conv_9_2 = convBNLayer(self.conv_9_1, use_batchnorm, is_training, 128, 256, 3, 2, name="conv_9_2", activation=activation)
        self.conv_10_1 = convBNLayer(self.conv_9_2, use_batchnorm, is_training, 256, 128, 1, 1, name="conv_10_1", activation=activation)
        self.conv_10_2 = convBNLayer(self.conv_10_1, use_batchnorm, is_training, 128, 256, 3, 1, name="conv_10_2", activation=activation, padding="VALID")
        self.conv_11_1 = convBNLayer(self.conv_10_2, use_batchnorm, is_training, 256, 128, 1, 1, name="conv_11_1", activation=activation)
        self.conv_11_2 = convBNLayer(self.conv_11_1, use_batchnorm, is_training, 128, 256, 3, 1, name="conv_11_2", activation=activation, padding="VALID")

class ExtraNetwork(object):
    def __init__(self, alpha=1):
        self.loss_layer = []
        self.layer_width = []
        self.layer_height = []
        self.num_anchors = []
        self.gt_true_cls = None
        self.gt_false_cls = None
        self.gt_box = None
        self.alpha = alpha

    def build_model(self, vgg_model, basic_model, use_batchnorm=False, is_training=True, activation=tf.nn.relu, \
                    rate=1, atrous=False, implement_atrous=False):
        self.out4_3 = convBNLayer(vgg.conv4_3, None, None, 512, ?, 3, 1, name="out4_3", activation=None)
        self.out5_3 = convBNLayer(vgg.conv5_3, None, None, 512, ?, 3, 1, name="out5_3", activation=None)
        self.out7 = convBNLayer(basic_model.conv7, None, None, 1024, ?, 3, 1, name="out7", activation=None)
        self.out8_2 = convBNLayer(basic_model.conv8_2, None, None, 512, ?, 3, 1, name="out8_2", activation=None)
        self.out9_2 = convBNLayer(basic_model.conv9_2, None, None, 256, ?, 3, 1, name="out9_2", activation=None)
        self.out10_2 = convBNLayer(basic_model.conv10_2, None, None, 256, ?, 3, 1, name="out10_2", activation=None)
        self.out11_2 = convBNLayer(basic_model.conv11_2, None, None, 256, ?, 3, 1, name="out10_2", activation=None)
        self.loss_layer = [self.out4_3, self.out5_3, self.out7, self.out8_2, self.out9_2, self.out10_2, self.out11_2]
        self.layer_width = []
        self.layer_height = []
        self.num_anchors = [4, 6, 6, 6, 6, 4, 4]
        self.ratios = []
        self.scales = []

    def loss(self):
        total_loss = 0
        cls_loss = 0
        bbox_loss = 0
        index = 0
        length_layer = self.loss_layer
        gt_true_cls = tf.placeholder(tf.float32, [batch_size, None])
        gt_false_cls = tf.placeholder(tf.float32, [batch_size, None])
        gt_bbox = tf.placeholder(tf.float32, [batch_size, None, 4])
        # layer: Shape is [Batch_Size, height, width, anchors*6] -> [Batch_Size, height, width, anchors, 6]
        for i, layer in enumerate(self.loss_layer):
            layer = tf.reshape(layer, [batch_size, None, 6]) # Shape is [Batch_Size, height*width*anchors, 6]
            next_index = index + self.layer_width[i]*self.layer_height[i]*self.num_anchors[i]
            pred_cls = layer[:, :, :2] # Shape is [Batch_Size, layer(height*width*anchors), 2]
            pred_box = layer[:, :, 2:] # Shape is [Batch_Size, layer(height*width*anchors), 4]
            gt_layer_true_cls = gt_true_cls[:, index:next_index]
            gt_layer_false_cls = gt_false_cls[:, index:next_index]
            gt_layer_bbox = gt_bbox[:, index:next_index]

            elosion = 0.00001
            true_cls_loss = -tf.reduce_sum(tf.multiply(tf.log(pred_cls[:, :, 0]+elosion), gt_layer_true_cls))
            false_cls_loss = -tf.reduce_sum(tf.multiply(tf.log(pred_cls[:, :, 1]+elosion), gt_layer_false_cls))
            layer_cls_loss = tf.add(true_cls_loss, false_cls_loss)

            layer_bbox_loss = smooth_L1(tf.subtract(pred_box, gt_layer_bbox))
            layer_bbox_loss = tf.reuce_sum(tf.multiply(tf.reduce_sum(layer_bbox_loss), 2), gt_layer_true_cls)
            layer_bbox_loss = tf.multiply(bbox_loss, self.alpha)

            layer_total_loss = tf.add(layer_cls_los, layer_bbox_loss)

            total_loss = tf.add(total_loss, layer_total_loss)
            cls_loss = tf.add(cls_loss, layer_cls_loss)
            bbox_loss = tf.add(bbox_loss, layer_bbox_loss)

            index = next_index
        return total_loss, cls_loss, bbox_loss

    def create_label(self, gt_boxes):
        """
        # Args:
            gt_boxes: GroundTruth Bounding Boxes. Shape is [Batch_Size, Num, 4].
        # Returns:
            default_boxes: regularized default boxes for network. Shape is [Batch_Size, None, 4].
            true_index   : GroundTruth Index. Shape is [Batch_Size, None].
            false_index  : Negative label index. Shape is [Batch_Size, None].
        """
        # self.gt_true_cls = true_index
        # self.gt_false_cls = false_index
        # self.gt_box = default_boxes
            # create center anchors
            # create offset for multiscale and multiratio default boxes
            # calculate jaccard overlaps
            # select groundtrush and false label (1 : 3)

        # Output Shape is [Batch Size, None, 6]
        sum_label = 0
        num_feature = len(self.num_anchors)
        for width, height, num_anchor in zip(self.layer_width, self.layer_height, self.num_anchors):
            sum_label += width * height * num_anchor

        default_boxes = np.zeros((batch_size, sum_label, 4))
        true_index = np.zeros((batch_size, sum_label))
        false_index = np.zeros((batch_size, sum_label))
        scales = [(0.2 + (0.9 - 0.2)*i / (num_feature-1)) for i in range(num_feature)]

        for width, height, num_anchor scale in zip(self.layer_width, self.layer_height, self.num_anchors, scales):
            batch_size = gt_boxes.shape[0]
            # shifts is the all candicate anchors(prediction of bounding boxes)
            center_x = np.arange(0, width) * feat_stride
            center_y = np.arange(0, height) * feat_stride
            center_x, center_y = np.meshgrid(center_x, center_y)
            # Shape is [Batch, Width*Height, 4]
            centers = np.zeros((batch_size, width*height, 4))
            centers[:] = np.vstack((center_x.ravel(), center_y.ravel(),
                                center_x.ravel(), center_y.ravel())).transpose()
            A = num_anchor
            K = width * height # width * height
            anchors = np.zeros((batch_size, A, 4))
            if num_anchor == 4:
                anchors = generate_anchors(scales=scales[:2], ratios=ratios[:2])
            anchors = generate_anchors(scales=scale, ratios=ratios) # Shape is [A, 4]

            anchors = centers.reshape(batch_size, K, 1, 4) + anchors # [Batch, K, A, 4] -> [Batch, None, 4]

            # shape is [B, K, A]
            is_inside = batch_inside_image(anchors, image_size[0], image_size[1])

        """
        ここまでnpyファイルに保存できる -> 処理時間の短縮につながる
        必要な情報は、feature mapのサイズと画像サイズ
        return として、anchorsとis_inside(0 or 1)を取得する
        """
            # anchors: Shape is [Batch, K, A, 4]
            # gt_boxes: Shape is [Batch, G, 4]
            # true_index: Shape is [Batch, K, A]
            # false_index: Shape is [Batch, K, A]
            anchors, true_index, false_index = bbox_overlaps(
                np.ascontiguousarray(anchors, dtype=np.float),
                is_inside,
                gt_boxes)

        for i in range(batch_size):
            true_where = np.where(true_index[i] == 1)
            num_true = len(true_where[0])

            if num_true > 64:
                select = np.random.choice(num_true, num_true - 64, replace=False)
                num_true = 64
                batch = np.ones((select.shape[0]), dtype=np.int) * i
                true_where = remove_extraboxes(true_where[0], true_where[1], select, batch)
                true_index[true_where] = 0

            false_where = np.where(false_index[i] == 1)
            num_false = len(false_where[0])
            select = np.random.choice(num_false, num_false - (128-num_true), replace=False)
            batch = np.ones((select.shape[0]), dtype=np.int) * i
            false_where = remove_extraboxes(false_where[0], false_where[1], select, batch)
            false_index[false_where] = 0


    def smooth_L1(self, x):
        l2 = 0.5 * (x**2.0)
        l1 = tf.abs(x) - 0.5

        condition = tf.less(tf.abs(x), 1.0)
        loss = tf.where(condition, l2, l1)
        return loss


def ssd_model(sess, vggpath=None, image_shape=(300, 300), \
              is_training=None, use_batchnorm=False, activation=tf.nn.relu, \
              num_classes=0, normalization=[], atrous=False, rate=1, implement_atrous=False):
    """
       1. input RGB images and labels
       2. edit images like [-1, image_shape[0], image_shape[1], 3]
       3. Create Annotate Layer?
       4. input x into Vgg16 architecture(pretrained)
       5.
    """
    images = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], 3])
    vgg = Vgg(vgg16_npy_path=vggpath)
    vgg.build_model(images)
    with tf.variable_scope("ssd_model"):
        phase_train = tf.placeholder(tf.bool, name="phase_traing") if is_training else None
        with tf.variable_scope("basic_model"):
            basic_model = BasicNetwork()
            basic_model.build_model(vgg.conv5_3, use_batchnorm=use_batchnorm, atrous=atrous, rate=rate, \
                                       is_training=phase_train, activation=activation, implement_atrous=implement_atrous)
        with tf.variable_scope("extra_model"):
            extra_model = ExtraNetwork()
            extra_model.build_model(vgg, basic_model)
    initialized_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="ssd_model")
    sess.run(tf.variables_initializer(initialized_var))

    return extra_model, images, phase_train

def train_ssd(batch_size, image_dir, label_dir, epoch=101, lr=0.01, feature_shape=(64, 19), \
                  vggpath="../pretrain/vgg16.npy", use_batchnorm=False, activation=tf.nn.relu, \
                  scales=np.array([5, 8, 12, 16, 32]), ratios=[0.3, 0.5, 0.8, 1], feature_stride=16):
    import time
    training_epochs = epoch

    with tf.Session() as sess:
        model, images, phase_train = ssd_model(sess, vggpath=vggpath, is_training=True, \
                                         use_batchnorm=use_batchnorm, activation=activation, anchors=scales.shape[0]*len(ratios))
        total_loss, cls_loss, bbox_loss, true_obj_loss, false_obj_loss = model.loss(model.rpn_cls, model.rpn_bbox)
        optimizer = create_optimizer(total_loss, lr=lr)
        init = tf.global_variables_initializer()
        sess.run(init)

        image_pathlist, label_pathlist = get_pathlist(image_dir, label_dir)
        for epoch in range(training_epochs):
            # TODO : Data Generator including Resize, DataAugmentation, and so on
            for batch_images, batch_labels in generator_Image_and_label(image_pathlist, label_pathlist, batch_size=batch_size):
                start = time.time()
                candicate_anchors, batch_true_index, batch_false_index = model.create_label(batch_labels, \
                    feat_stride=feature_stride, feature_shape=(batch_images.shape[1]//feature_stride +1, batch_images.shape[2]//feature_stride+1), \
                    scales=scales, ratios=ratios, image_size=batch_images.shape[1:3])
                # candicate_anchors, batch_true_index, batch_false_index = create_Labels_For_Loss(batch_labels, feat_stride=feature_stride, feature_shape=(batch_images.shape[1]//feature_stride +1, batch_images.shape[2]//feature_stride+1), \
                #                            scales=scales, ratios=ratios, image_size=batch_images.shape[1:3])
                print "batch time", time.time() - start
                print batch_true_index[batch_true_index==1].shape
                print batch_false_index[batch_false_index==1].shape

                sess.run(optimizer, feed_dict={images:batch_images, g_bboxes: candicate_anchors, true_index:batch_true_index, false_index:batch_false_index})
                tl, cl, bl, tol, fol = sess.run([total_loss, cls_loss, bbox_loss, true_obj_loss, false_obj_loss], feed_dict={images:batch_images, g_bboxes: candicate_anchors, true_index:batch_true_index, false_index:batch_false_index})
                print("Epoch:", '%04d' % (epoch+1), "total loss=", "{:.9f}".format(tl))
                print("Epoch:", '%04d' % (epoch+1), "closs loss=", "{:.9f}".format(cl))
                print("Epoch:", '%04d' % (epoch+1), "bbox loss=", "{:.9f}".format(bl))
                print("Epoch:", '%04d' % (epoch+1), "true loss=", "{:.9f}".format(tol))
                print("Epoch:", '%04d' % (epoch+1), "false loss=", "{:.9f}".format(fol))

def create_Labels_For_Loss(gt_boxes, feat_stride=16, feature_shape=(64, 19), \
                           scales=np.array([8, 16, 32]), ratios=[0.5, 0.8, 1], \
                           image_size=(300, 1000)):
    """This Function is processed before network input
    Number of Candicate Anchors is Feature Map width * heights
    Number of Predicted Anchors is Batch Num * Feature Map Width * Heights * 9
    """
    width = feature_shape[0]
    height = feature_shape[1]
    batch_size = gt_boxes.shape[0]
    # shifts is the all candicate anchors(prediction of bounding boxes)
    center_x = np.arange(0, height) * feat_stride
    center_y = np.arange(0, width) * feat_stride
    center_x, center_y = np.meshgrid(center_x, center_y)
    # Shape is [Batch, Width*Height, 4]
    centers = np.zeros((batch_size, width*height, 4))
    centers[:] = np.vstack((center_x.ravel(), center_y.ravel(),
                        center_x.ravel(), center_y.ravel())).transpose()
    A = scales.shape[0] * len(ratios)
    K = width * height # width * height
    anchors = np.zeros((batch_size, A, 4))
    anchors = generate_anchors(scales=scales, ratios=ratios) # Shape is [A, 4]

    candicate_anchors = centers.reshape(batch_size, K, 1, 4) + anchors # [Batch, K, A, 4]

    # shape is [B, K, A]
    is_inside = batch_inside_image(candicate_anchors, image_size[1], image_size[0])

    """
    ここまでnpyファイルに保存できる -> 処理時間の短縮につながる
    必要な情報は、feature mapのサイズと画像サイズ
    return として、anchorsとis_inside(0 or 1)を取得する
    """
    # candicate_anchors: Shape is [Batch, K, A, 4]
    # gt_boxes: Shape is [Batch, G, 4]
    # true_index: Shape is [Batch, K, A]
    # false_index: Shape is [Batch, K, A]
    candicate_anchors, true_index, false_index = bbox_overlaps(
        np.ascontiguousarray(candicate_anchors, dtype=np.float),
        is_inside,
        gt_boxes)

    for i in range(batch_size):
        true_where = np.where(true_index[i] == 1)
        num_true = len(true_where[0])

        if num_true > 64:
            select = np.random.choice(num_true, num_true - 64, replace=False)
            num_true = 64
            batch = np.ones((select.shape[0]), dtype=np.int) * i
            true_where = remove_extraboxes(true_where[0], true_where[1], select, batch)
            true_index[true_where] = 0

        false_where = np.where(false_index[i] == 1)
        num_false = len(false_where[0])
        select = np.random.choice(num_false, num_false - (128-num_true), replace=False)
        batch = np.ones((select.shape[0]), dtype=np.int) * i
        false_where = remove_extraboxes(false_where[0], false_where[1], select, batch)
        false_index[false_where] = 0

    return candicate_anchors, true_index, false_index

class MultiboxLayer(object):
    def __init__(self):
        pass

    # TODO: validate this is correct or not
    def l2_normalization(self, input_layer, scale=20):
        return tf.nn.l2_normalize(input_layer, dim) * scale

    def createMultiBoxHead(self, from_layers, num_classes=0, normalizations=[], \
                           use_batchnorm=False, is_training=None, activation=None, \
                           kernel_size=3, prior_boxes=[], kernel_sizes=[]):
        """
           # Args:
               from_layers(list)   : list of input layers
               num_classes(int)    : num of label's classes that this architecture detects
               normalizations(list): list of scale for normalizations
                                     if value <= 0, not apply normalization to the specified layer
        """
        assert num_classes > 0, "num of label's class  must be positive number"
        if normalizations:
            assert len(from_layers) == len(normalizations), "from_layers and normalizations should have same length"

        num_list = len(from_layers)
        for index, kernel_size, layer, norm in zip(range(num_list), kernel_sizes, from_layers, normalizations):
            input_layer = layer
            with tf.variable_scope("layer" + str(index+1)):
                if norm > 0:
                    scale = tf.get_variable("scale", trainable=True, initializer=tf.constant(norm))#initialize = norm
                    input_layer = self.l2_normalization(input_layer, scale)

                # create location prediction layer
                loc_output_dim = 4 * prior_num # (center_x, center_y, width, height)
                location_layer = convBNLayer(input_layer, use_batchnorm, is_training, input_layer.get_shape()[0], loc_output_dim, kernel_size, 1, name="loc_layer", activation=activation)
                # from shape : (batch, from_kernel, from_kernel, loc_output_dim)
                # to         : (batch, )
                location_pred = tf.reshape(location_layer, [-1, ])

                # create confidence prediction layer
                conf_output_dim = num_classes * prior_num
                confidence_layer = convBNLayer(input_layer, use_batchnorm, is_training, input_layer.get_shape()[0], conf_output_dim, kernel_size, 1, name="conf_layer", activation=activation)
                confidence_pred = tf.reshape(confidence_pred, [-1, ])

                # Flatten each output

                # append result of each results

        return None

if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt
    from PIL import Image as im
    sys.path.append('/home/katou01/code/grid/DataAugmentation')
    from resize import resize

    image = im.open("./test_images/test1.jpg")
    image = np.array(image, dtype=np.float32)
    new_image = image[np.newaxis, :]
    batch_image = np.vstack((new_image, new_image))
    batch_image = resize(batch_image, size=(300, 300))

    with tf.Session() as sess:
        model = ssd_model(sess, batch_image, activation=None, atrous=False, rate=1, implement_atrous=False)
        print(vars(model))
        # tf.summary.scalar('model', model)
