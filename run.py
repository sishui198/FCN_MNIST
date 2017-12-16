import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from datetime import datetime

import source.neuralnet as nn

def save_graph_as_image(train_list, test_list, ylabel="", label1="train", label2="test", cate="None"):

    print("Save "+ylabel+" graph in ./graph")

    x = np.arange(len(train_list))
    plt.clf()
    plt.rcParams['lines.linewidth'] = 1
    plt.plot(x, train_list, label=label1, linestyle='--')
    plt.plot(x, test_list, label=label2, linestyle='--')
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.ylim(-0.1, max([1, max(train_list), max(test_list)])*1.1)
    if(ylabel == "accuracy"):
        plt.legend(loc='lower right')
    else:
        plt.legend(loc='upper right')
    #plt.show()

    if(not(os.path.exists("./graph"))):
        os.mkdir("./graph")
    else:
        pass
    now = datetime.now()

    plt.savefig("./graph/"+now.strftime('%Y%m%d_%H%M%S%f')+"_"+cate+"_"+ylabel+".png")
    # plt.show()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32)
y_ = tf.placeholder(tf.float32)
training = tf.placeholder(tf.bool)

convnet = nn.ConvNeuralNet(x=x, y_=y_, training=training, height=28, width=28, channel=1, classes=10)

firstconv = convnet._firstconv
train_step = convnet._trainstep
accuracy = convnet._accuracy
cross_entropy = convnet._loss
prediction = convnet._prediction

sess.run(tf.global_variables_initializer())

train_acc_list = []
train_loss_list = []
test_acc_list = []
test_loss_list = []

print("\nTraining")
for i in range(1000):
    train_batch = mnist.train.next_batch(50)
    train_x = np.asarray(train_batch[0]).reshape((-1, 28, 28, 1))
    if i%100 == 0:
        test_batch = mnist.test.next_batch(50)
        test_x = np.asarray(test_batch[0]).reshape((-1, 28, 28, 1))

        train_accuracy = accuracy.eval(feed_dict={x:train_x, y_:train_batch[1], training:False})
        test_accuracy = accuracy.eval(feed_dict={x:test_x, y_:test_batch[1], training:False})
        train_loss = cross_entropy.eval(feed_dict={x:train_x, y_:train_batch[1], training:False})
        test_loss = cross_entropy.eval(feed_dict={x:test_x, y_:test_batch[1], training:False})

        train_acc_list.append(train_accuracy)
        train_loss_list.append(train_loss)
        test_acc_list.append(test_accuracy)
        test_loss_list.append(test_loss)

        print("step %4d, training accuracy | %.4f %2.4f"%(i, train_accuracy, train_loss))

    sess.run(train_step, feed_dict={x:train_x, y_:train_batch[1], training:True})

save_graph_as_image(train_list=train_acc_list, test_list=test_acc_list, ylabel="Accuracy", cate="MNIST")
save_graph_as_image(train_list=train_loss_list, test_list=test_loss_list, ylabel="Loss", cate="MNIST")

for digit in range(10):

    while True:
        test_data = mnist.test.next_batch(1)
        train_x = np.asarray(test_data[0]).reshape((-1, 28, 28, 1))

        if(np.argmax(test_data[1][0]) == digit):
            img = np.transpose(train_x[0], (2, 0, 1))[0]
            plt.clf()
            plt.imshow(img)
            plt.savefig(str(digit)+"_origin.png")

            conv = sess.run(firstconv, feed_dict={x: train_x, y_:test_data[1]})

            img = np.transpose(conv[0], (2, 0, 1))[0]
            plt.clf()
            plt.imshow(img)
            plt.savefig(str(digit)+"_heat.png")

            break
