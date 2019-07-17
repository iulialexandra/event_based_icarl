import tensorflow as tf


class RoshamboNet(object):
    def __init__(self, output_size, input_data, phase, res_blocks=None):
        self.output_size = output_size
        self.input_data = input_data
        self.phase = phase
        self.res_blocks = res_blocks

    def get_variable(self, name, shape, dtype, initializer, trainable=True, regularizer=None):
        var = tf.get_variable(name, shape=shape, dtype=dtype, initializer=initializer,
                              regularizer=regularizer, trainable=trainable,
                              collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])
        return var

    def get_weight_initializer(self, params, variables_test):
        assert len(params) == len(variables_test), "Train and test network do not have the same" \
                                                   "number of layers"
        initializer = [variables_test[i].assign(params[i]) for i in range(len(params))]
        return initializer

    def conv(self, in_layer, name, filter_size, out_channels, strides=[1, 1, 1, 1],
             padding='SAME', apply_relu=True, bias=True,
             initializer=tf.contrib.layers.xavier_initializer_conv2d()):
        in_channels = in_layer.get_shape().as_list()[3]

        with tf.variable_scope(name):
            W = self.get_variable("W", shape=[filter_size, filter_size, in_channels, out_channels],
                                  dtype=tf.float32, initializer=initializer,
                                  regularizer=tf.nn.l2_loss)
            b = self.get_variable("b", shape=[1, 1, 1, out_channels], dtype=tf.float32,
                                  initializer=tf.zeros_initializer(), trainable=bias)
            out = tf.add(tf.nn.conv2d(in_layer, W, strides=strides, padding=padding), b,
                         name='convolution')
            if apply_relu:
                out = tf.nn.relu(out, name='relu')

        return out

    def pool(self, in_layer, name, kind, size, stride, padding='SAME'):
        assert kind in ['max', 'avg']

        strides = [1, stride, stride, 1]
        sizes = [1, size, size, 1]

        with tf.variable_scope(name):
            if kind == 'max':
                out = tf.nn.max_pool(in_layer, sizes, strides=strides, padding=padding, name=kind)
            else:
                out = tf.nn.avg_pool(in_layer, sizes, strides=strides, padding=padding, name=kind)

        return out

    def build_cnn(self):
        layer = self.conv(self.input_data, "conv1", 5, 16, padding="VALID")
        layer = self.pool(layer, "pool1", "max", 2, 2, padding="VALID")
        layer = self.conv(layer, "conv2", 3, 32, padding="VALID")
        layer = self.pool(layer, "pool2", "max", 2, 2, padding="VALID")
        layer = self.conv(layer, "conv3", 3, 64, padding="VALID")
        layer = self.pool(layer, "pool3", "max", 2, 2, padding="VALID")
        layer = self.conv(layer, "conv4", 3, 128, padding="VALID")
        layer = self.pool(layer, "pool4", "max", 2, 2, padding="VALID")
        layer = self.conv(layer, "conv_last", 1, 128, padding="VALID")
        layer = self.pool(layer, "pool_last", "max", 2, 2, padding="VALID")
        if self.phase != "icarl_inference":
            out = self.conv(layer, name='fc', filter_size=1, out_channels=self.output_size,
                            padding='VALID', apply_relu=False)[:, 0, 0, :]
            return out, layer
        elif self.phase == "icarl_inference":
            return layer[:, 0, 0, :]


class ResNet(object):
    def __init__(self, output_size, images, phase, res_blocks):
        self.output_size = output_size
        self.input_data = images
        self.phase = phase
        self.res_blocks = res_blocks

    def get_variable(self, name, shape, dtype, initializer, trainable=True, regularizer=None):
        var = tf.get_variable(name, shape=shape, dtype=dtype, initializer=initializer,
                              regularizer=regularizer, trainable=trainable,
                              collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])
        return var

    def get_weight_initializer(self, params, variables_test):
        assert len(params) == len(variables_test), "Train and test network do not have the same" \
                                                   "number of layers"
        initializer = [variables_test[i].assign(params[i]) for i in range(len(params))]
        return initializer

    def conv(self, in_layer, name, filter_size, out_channels, strides=[1, 1, 1, 1],
             dilation=None, padding='SAME', apply_relu=True, bias=True,
             initializer=tf.contrib.layers.xavier_initializer_conv2d()):
        batch_size = in_layer.get_shape().as_list()[0]
        res1 = in_layer.get_shape().as_list()[1]
        res2 = in_layer.get_shape().as_list()[1]
        in_channels = in_layer.get_shape().as_list()[3]

        with tf.variable_scope(name):
            W = self.get_variable("W", shape=[filter_size, filter_size, in_channels, out_channels],
                                  dtype=tf.float32, initializer=initializer,
                                  regularizer=tf.nn.l2_loss)
            b = self.get_variable("b", shape=[1, 1, 1, out_channels], dtype=tf.float32,
                                  initializer=tf.zeros_initializer(), trainable=bias)

            if dilation:
                assert (strides == [1, 1, 1, 1])
                out = tf.add(tf.nn.atrous_conv2d(in_layer, W, rate=dilation, padding=padding), b,
                             name='convolution')
                out.set_shape([batch_size, res1, res2, out_channels])
            else:
                out = tf.add(tf.nn.conv2d(in_layer, W, strides=strides, padding=padding), b,
                             name='convolution')

            if apply_relu:
                out = tf.nn.relu(out, name='relu')

        return out

    def pool(self, in_layer, name, kind, size, stride, padding='SAME'):
        assert kind in ['max', 'avg']

        strides = [1, stride, stride, 1]
        sizes = [1, size, size, 1]

        with tf.variable_scope(name):
            if kind == 'max':
                out = tf.nn.max_pool(in_layer, sizes, strides=strides, padding=padding, name=kind)
            else:
                out = tf.nn.avg_pool(in_layer, sizes, strides=strides, padding=padding, name=kind)

        return out

    def batch_norm(self, in_layer, name, phase, decay=0.9):
        channels = in_layer.get_shape().as_list()[3]

        with tf.variable_scope(name):
            moving_mean = self.get_variable("mean", shape=[channels], dtype=tf.float32,
                                            initializer=tf.constant_initializer(0.0),
                                            trainable=False)
            moving_variance = self.get_variable("var", shape=[channels], dtype=tf.float32,
                                                initializer=tf.constant_initializer(1.0),
                                                trainable=False)

            offset = self.get_variable("offset", shape=[channels], dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.0))
            scale = self.get_variable("scale", shape=[channels], dtype=tf.float32,
                                      initializer=tf.constant_initializer(1.0),
                                      regularizer=tf.nn.l2_loss)

            mean, variance = tf.nn.moments(in_layer, axes=[0, 1, 2], shift=moving_mean)

            mean_op = moving_mean.assign(decay * moving_mean + (1 - decay) * mean)
            var_op = moving_variance.assign(decay * moving_variance + (1 - decay) * variance)

            if phase == 'train':
                with tf.control_dependencies([mean_op, var_op]):
                    return tf.nn.batch_normalization(in_layer, mean, variance, offset, scale, 0.01,
                                                     name='norm')
            else:
                return tf.nn.batch_normalization(in_layer, moving_mean, moving_variance, offset,
                                                 scale,
                                                 0.01,
                                                 name='norm')

    def residual_block(self, in_layer, phase, block_name, increase_dim=False, last=False):
        input_num_filters = in_layer.get_shape().as_list()[3]
        if increase_dim:
            first_stride = [1, 2, 2, 1]
            out_num_filters = input_num_filters * 2
        else:
            first_stride = [1, 1, 1, 1]
            out_num_filters = input_num_filters

        layer1 = self.conv(in_layer, 'resconv1' + block_name, filter_size=3, strides=first_stride,
                           out_channels=out_num_filters, padding='SAME')
        layer2 = self.batch_norm(layer1, 'batch_norm_resconv1' + block_name, phase=phase)
        layer3 = self.conv(layer2, 'resconv2' + block_name, filter_size=3, strides=[1, 1, 1, 1],
                           out_channels=out_num_filters, apply_relu=False, padding='SAME')
        layer4 = self.batch_norm(layer3, 'batch_norm_resconv2' + block_name, phase=phase)

        if increase_dim:
            projection = self.conv(in_layer, 'projconv' + block_name, filter_size=1,
                                   strides=[1, 2, 2, 1],
                                   out_channels=out_num_filters, apply_relu=False,
                                   padding='SAME', bias=False)
            projection = self.batch_norm(projection, 'batch_norm_projconv' + block_name,
                                         phase=phase)
            if last:
                block = layer4 + projection
            else:
                block = layer4 + projection
                block = tf.nn.relu(block, name='relu')
        else:
            if last:
                block = layer4 + in_layer
            else:
                block = layer4 + in_layer
                block = tf.nn.relu(block, name='relu')

        return block

    def build_cnn(self):
        layer = self.batch_norm(self.input_data, "batch_norm_0", phase=self.phase)
        layer = self.conv(layer, "conv1", filter_size=7, strides=[1, 2, 2, 1], out_channels=64,
                          padding='SAME')

        # first stack of residual blocks,
        for n in range(self.res_blocks):
            layer = self.residual_block(layer, phase=self.phase, block_name="a" + str(n))

        # second stack of residual blocks
        layer = self.residual_block(layer, phase=self.phase, increase_dim=True, block_name="b")
        for n in range(self.res_blocks):
            layer = self.residual_block(layer, phase=self.phase, block_name="c" + str(n))

        # third stack of residual blocks
        layer = self.residual_block(layer, phase=self.phase, increase_dim=True, block_name="d")
        for n in range(self.res_blocks - 1):
            layer = self.residual_block(layer, phase=self.phase, block_name="e" + str(n))

        layer = self.residual_block(layer, phase=self.phase, last=True, block_name="f")

        layer = self.pool(layer, 'pool_last', 'avg', size=layer.get_shape().as_list()[1],
                          stride=1, padding='VALID')
        if self.phase != "icarl_inference":
            out = self.conv(layer, name='fc', filter_size=1, out_channels=self.output_size,
                            padding='VALID', apply_relu=False)[:, 0, 0, :]
            return out, layer
        elif self.phase == "icarl_inference":
            return layer[:, 0, 0, :]
