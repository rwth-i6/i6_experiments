import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
import os

out_folder = os.path.join(os.path.dirname(__file__), "visualizations")
if not os.path.exists(out_folder):
  os.makedirs(out_folder)

mean = 0.0
x = tf.range(-10., 11., 1.)
x_labels = ["", "$t_s-4$", "", "$t_s-2$", "", "$t_s$", "", "$t_s+2$", "", "$t_s+4$", "", ]
distribs = {
  "gauss": lambda x, mean, std: tf.nn.softmax(-0.5 * ((x - mean) / std) ** 2),
  "laplace": lambda x, mean, std: tf.nn.softmax(-tf.abs((x - mean) / std)),
}

for distrib in distribs:
  for std in (1.0, 2.0, 4.0, 8.0):
    dist = distribs[distrib](x, mean, std)

    ax = plt.gca()
    ax.set_ylim([0., .5])
    ax.set_xlim([0., 21.])
    ax.set_xticks(range(0, len(x.numpy()), 2))
    ax.set_xticklabels(x_labels, fontsize=16)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_yticklabels([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=16)
    plt.plot(dist, label="sigma={std}".format(std=std))
    plt.legend(prop={"size": 16})

  plt.savefig(os.path.join(out_folder, "{distrib}.png".format(distrib=distrib)))
  plt.savefig(os.path.join(out_folder, "{distrib}.pdf".format(distrib=distrib)))
  plt.clf()
