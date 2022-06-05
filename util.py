import numpy as np


def sample_Z(batch, z_dim, num_class, label_index=None):
  
    if label_index is None:
        label_index = np.random.randint(low=0, high=num_class, size=batch)
    return np.hstack((0.10 * np.random.randn(batch, z_dim - num_class), np.eye(num_class)[label_index]))


if __name__ == '__main__':
    l = sample_Z(10, 22, 'mul_cat', 10, 2)
    print(l)
