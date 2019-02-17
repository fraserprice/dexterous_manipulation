import numpy as np
import tensorflow as tf


class CuriosityModule:
    def __init__(self, action_space_size, state_space_size):
        self.action_space = action_space_size
        self.state_space = state_space_size
        self.weights = []
        self.biases = []
        self.input = None
        self.output = None
        self.sess = tf.Session()

    def forward_pass(self, input, weights, biases, keep_prob):
        progress = input
        for i, layer in enumerate(weights):
            progress = tf.add(tf.matmul(progress, layer), biases[i])
            if i < len(weights) - 1:
                progress = tf.nn.relu(progress)
                progress = tf.nn.dropout(progress, keep_prob)
        return progress

    def initialize_curiosity_network(self, hidden_layer_sizes):
        input_size = self.action_space + self.state_space
        output_size = self.state_space

        self.input = tf.placeholder("float", shape=[None, input_size])
        self.output = tf.placeholder("float", shape=[None, output_size])

        self.weights = []
        current_size = input_size
        for layer_size in hidden_layer_sizes:
            self.weights.append(tf.Variable(tf.random_normal((current_size, layer_size), stddev=0.1)))
            current_size = layer_size
        self.weights.append(tf.Variable(tf.random_normal((current_size, output_size), stddev=0.1)))

        self.biases = []
        for layer_size in hidden_layer_sizes:
            self.biases.append(tf.Variable(tf.random_normal([layer_size])))
        self.biases.append(tf.Variable(tf.random_normal([output_size])))

    def optimize(self, xs, ys):
        init = tf.global_variables_initializer()
        self.sess.run(init)

        predicted_xs = []
        for x in xs:
            predicted_xs.append(self.forward_pass(x, self.weights, self.biases, 0.8))

        cost = tf.losses.mean_squared_error(labels=ys, predictions=predicted_xs)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
        keep_prob = tf.placeholder("float")
        _, c = self.sess.run(
            [optimizer, cost],
            feed_dict={
                self.input: xs,
                self.output: ys,
                keep_prob: 0.8
            })


cm = CuriosityModule(2, 1)
cm.initialize_curiosity_network([3])
cm.optimize([(1., 1.), (1., 0.), (0., 1.), (0., 0.), (1., 1.), (0., 1.)], [1., 0., 0., 0., 1., 0.])
