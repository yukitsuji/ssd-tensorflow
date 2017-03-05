#!/usr/bin/env python3
value = 20
init = tf.constant_initializer(value)
with tf.Session():
    x = tf.get_variable('xyz', 1,initializer=init)
    x.initializer.run()
    print(x.eval())
    print(vars(x))
