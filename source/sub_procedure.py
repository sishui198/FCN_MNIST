import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

import source.utility as util

def train(dataset, data, label, training, model, sess, steps, batch):

    train_step = model._train_step
    accuracy = model._accuracy
    loss = model._loss

    train_acc_list = []
    train_loss_list = []
    test_acc_list = []
    test_loss_list = []

    print("\nTraining")
    for i in range(steps+1):
        train_batch = dataset.train.next_batch(batch)
        train_x = np.asarray(train_batch[0]).reshape((-1, 28, 28, 1))

        if(i%100 == 0):
            test_batch = dataset.test.next_batch(batch)
            test_x = np.asarray(test_batch[0]).reshape((-1, 28, 28, 1))

            train_accuracy = accuracy.eval(feed_dict={data:train_x, label:train_batch[1], training:False})
            test_accuracy = accuracy.eval(feed_dict={data:test_x, label:test_batch[1], training:False})
            train_loss = loss.eval(feed_dict={data:train_x, label:train_batch[1], training:False})
            test_loss = loss.eval(feed_dict={data:test_x, label:test_batch[1], training:False})

            train_acc_list.append(train_accuracy)
            train_loss_list.append(train_loss)
            test_acc_list.append(test_accuracy)
            test_loss_list.append(test_loss)

            print("step %4d, training accuracy | %.4f %2.4f"%(i, train_accuracy, train_loss))

        sess.run(train_step, feed_dict={data:train_x, label:train_batch[1], training:True})

    util.save_graph_as_image(train_list=train_acc_list, test_list=test_acc_list, ylabel="Accuracy", cate="MNIST")
    util.save_graph_as_image(train_list=train_loss_list, test_list=test_loss_list, ylabel="Loss", cate="MNIST")

def heatmap(dataset, data, label, training, model, sess):

    conv_1 = model._conv_1
    conv_2 = model._conv_2
    conv_3 = model._conv_3

    print("Make heatmap")
    if(not(os.path.exists("./heatmap"))):
        os.mkdir("./heatmap")
    for digit in range(10):

        while True:
            test_data = dataset.test.next_batch(1)
            train_x = np.asarray(test_data[0]).reshape((-1, 28, 28, 1))

            if(np.argmax(test_data[1][0]) == digit):
                img = np.transpose(train_x[0], (2, 0, 1))[0]
                plt.clf()
                plt.imshow(img)
                plt.savefig("./heatmap/"+str(digit)+"_origin.png")

                active_1 = sess.run(conv_1, feed_dict={data: train_x, label:test_data[1], training:False})
                img1 = np.transpose(active_1[0], (2, 0, 1))[0]
                plt.clf()
                plt.imshow(img1)
                plt.savefig("./heatmap/"+str(digit)+"_heat1.png")

                active_2 = sess.run(conv_2, feed_dict={data: train_x, label:test_data[1], training:False})
                img2 = np.transpose(active_2[0], (2, 0, 1))[0]
                plt.clf()
                plt.imshow(img2)
                plt.savefig("./heatmap/"+str(digit)+"_heat2.png")

                active_3 = sess.run(conv_3, feed_dict={data: train_x, label:test_data[1], training:False})
                img3 = np.transpose(active_3[0], (2, 0, 1))[0]
                plt.clf()
                plt.imshow(img3)
                plt.savefig("./heatmap/"+str(digit)+"_heat3.png")
                break
