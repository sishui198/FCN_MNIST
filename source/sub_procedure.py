import os
import scipy.misc
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
    print("")

    util.save_graph_as_image(train_list=train_acc_list, test_list=test_acc_list, ylabel="Accuracy", cate="MNIST")
    util.save_graph_as_image(train_list=train_loss_list, test_list=test_loss_list, ylabel="Loss", cate="MNIST")

def save_heatmap(layer, digit, layer_name, data, label, training, sess, data_x, label_y):

    heat = sess.run(layer, feed_dict={data:data_x, label:label_y, training:False})
    img = np.transpose(heat[0], (2, 0, 1))[0]
    plt.clf()
    plt.imshow(img)
    plt.savefig("./heatmap/"+str(digit)+"_heat_"+str(layer_name)+".png")

    scipy.misc.imsave("./heatmap/"+str(digit)+"_gray_"+str(layer_name)+".png", img)

def heatmap(dataset, data, label, training, model, sess):

    print("\nMake heatmap")
    if(not(os.path.exists("./heatmap"))):
        os.mkdir("./heatmap")
    for digit in range(10):

        while True:
            test_data = dataset.test.next_batch(1)
            test_x = np.asarray(test_data[0]).reshape((-1, 28, 28, 1))

            if(np.argmax(test_data[1][0]) == digit):
                img = np.transpose(test_x[0], (2, 0, 1))[0]
                plt.clf()
                plt.imshow(img)
                plt.savefig("./heatmap/"+str(digit)+"_origin.png")
                scipy.misc.imsave("./heatmap/"+str(digit)+"_gray_origin.png", img)

                save_heatmap(layer=model._conv_1, digit=digit, layer_name="conv1", data=data, label=label, training=training, sess=sess, data_x=test_x, label_y=test_data[1])
                save_heatmap(layer=model._maxpool_1, digit=digit, layer_name="maxpool1", data=data, label=label, training=training, sess=sess, data_x=test_x, label_y=test_data[1])

                save_heatmap(layer=model._conv_2, digit=digit, layer_name="conv2", data=data, label=label, training=training, sess=sess, data_x=test_x, label_y=test_data[1])
                save_heatmap(layer=model._maxpool_2, digit=digit, layer_name="maxpool2", data=data, label=label, training=training, sess=sess, data_x=test_x, label_y=test_data[1])

                save_heatmap(layer=model._conv_3, digit=digit, layer_name="conv3", data=data, label=label, training=training, sess=sess, data_x=test_x, label_y=test_data[1])
                save_heatmap(layer=model._maxpool_3, digit=digit, layer_name="maxpool3", data=data, label=label, training=training, sess=sess, data_x=test_x, label_y=test_data[1])
                break
