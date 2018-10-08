import tensorflow as tf
#import mnist_inference
import numpy as np
import os

#tf.device('/gpu:0')

#os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'  
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"
#gpuConfig = tf.ConfigProto(allow_soft_placement=True)
#gpuConfig.gpu_options.allow_growth = True  

###################################################################################
imgs_new   =  np.load('train_img.npy')
labels_new =  np.load('train_lb.npy')
    
BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 200000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="MNIST_model/"
MODEL_NAME="mnist_model"

IMG_INPUT_NODE = 784
IMG_OUTPUT_NODE = 10

train_num_examples = imgs_new.shape[0]
################################################################################
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', regularizer(weights))
    return weights


def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):

        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2

################################################################################

def get_target_array(num):
    output = np.zeros((1, 10))
    output[0][int(num)] = 1
    return output

def get_batch_data(BATCH_SIZE):
    xx = np.zeros((BATCH_SIZE , 784))
    yy = np.zeros((BATCH_SIZE , 10))
    for i in range (0, BATCH_SIZE):
       index = np.random.randint(0, imgs_new.shape[0])
       for j in range (0, 784):
          xx[i][j] = imgs_new[index][0][j]
       yy_ = get_target_array(labels_new[index])
       for j in range (0, 10):
          yy[i][j] = yy_[0][j]
    return xx, yy
    
#tf.device('/gpu:1')    
def train():
    #shape=(1024, 1024)
    x  = tf.placeholder(tf.float32,  [None, IMG_INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, IMG_OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)


    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    #cross_entropy_mean = tf.reduce_mean(cross_entropy)
    rmse = tf.reduce_sum(tf.square(y_ -  y))
    loss = rmse + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        train_num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')


    saver = tf.train.Saver()
    #saver.restore(sess, './MNIST_model/konghai')
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver.restore(sess, './MNIST_model/konghai')
        for i in range(TRAINING_STEPS):
            xs, ys = get_batch_data(BATCH_SIZE)#mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    #mnist = input_data.read_data_sets("../../../datasets/MNIST_data", one_hot=True)
    train()

if __name__ == '__main__':
    tf.app.run()
