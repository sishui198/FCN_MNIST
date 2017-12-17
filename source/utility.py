import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

from datetime import datetime

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
