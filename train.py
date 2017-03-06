#!/usr/bin/env python3
from ssd_model import ssd_model
from ssd_generator import Generator
from utilities import divide_train_test

def get_accuracy(x, y, phase_train, generator, accuracy, test_batch_size=32):
    """Get accuracy of selected datasets"""
    num_iter = X_test.shape[0] // test_batch_size
    num_accuracy= 0
    for X_batch, Y_batch in train_generator.batch(batch_size):
        num_accuracy += accuracy.eval({x : X_batch, y:Y_batch, phase_train: None})
    num_accuracy = num_accuracy / num_iter
    return num_accuracy

def divide_train_test(x, y, test_size=test_size, random_state=1):
    return x_train, x_valid, y_train, y_valid

def create_datasets(xml_path):
    pass

class Train(object):
    def __init__(self):
        pass

    def _train(self):
        pass

    def process(self, train_xml, test_xml, batch_size=32, epoch_num=50):
        x, y, class_list = create_datasets(train_xml)
        X_train, X_valid, Y_train, Y_valid = divide_train_test(
            x, y, test_size=test_size, random_state=1,)

        datagenerator = DataAugmentation([
            Flip(1),
            # Shift((20, 20)),
            Crop(size=(280, 280)),
            Resize(size=(300, 300)),
            ColorJitter(v = (0.75, 1.5)),
            Rescale(option='vgg')
        ], random = True)

        train_generator = Generator(X_train, Y_train, class_list, callback=datagenerator, \
            shuffle=True, imsize=(300, 300), color='RGB')
        valid_generator = Generator(X_valid, Y_valid, class_list, shuffle=False, imsize=(360, 360), color="RGB")

        x, y, optimizer, cost, accuracy, phase_train = ssd_model(sess, images, activation=None, atrous=False, rate=1, implement_atrous=False)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            # sess.run(tf.global_variables_initializer())
            for epoch in range(epoch_num):
                sum_cost = 0
                for i, (X_batch, Y_batch) in enumerate(train_generator.batch(batch_size)):
                    sess.run(optimizer, feed_dict={x:X_batch, y:Y_batch, phase_train:True})
                    sum_cost += sess.run(cost, feed_dict={x:X_batch, y:Y_batch})
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(sum_cost))

                if (epoch % 5) == 0:
                    valid_accuracy = get_accuracy(x, y, phase_train, valid_generator, accuracy, test_batch_size=batch_size)
                    print("Validation Accuracy ", valid_accuracy)
                    # valid_accuracy_list.append(valid_accuracy)
            print("finish")
            saver.save(sess, "model.ckpt")

    def _save_parameter(self):
        pass

    def _cal_loss(self):
        pass

    def _test(self):
        pass

if __name__ == "__main__":
    pass
