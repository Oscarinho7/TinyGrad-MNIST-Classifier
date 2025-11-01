from tinygrad.helpers import getenv
from tinygrad.nn.datasets import mnist

import matplotlib.pyplot as plt

if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = mnist()
  print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
  print(X_train.min().item(), X_train.max().item())

  X_test = X_test.reshape(-1, 28, 28, 1).repeat(1, 1, 1, 3).realize()
  idx = int(getenv('IDX', 0))
  plt.title(f'Label: {Y_test[idx].item()}')
  plt.imshow(X_test[idx].numpy())
  plt.yticks([])
  plt.xticks([])
  plt.show()
