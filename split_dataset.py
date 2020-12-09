import os

import numpy as np

train_labels_np = np.load("data/tiny_imagenet/train_labels.npy")
test_labels_np = np.load("data/tiny_imagenet/valid_labels.npy")

test_labels = [[] for _ in range(200)]
for i in range(50 * 200):
    test_labels[test_labels_np[i]].append(i)

os.makedirs("data/tiny_imagenet/10_split/")
for i in range(5):
    label_list = range(40 * i, 40 * (i + 1))
    idx_train = np.concatenate([np.arange(500 * l, 500 * (l + 1)) for l in label_list])
    idx_test = np.concatenate([test_labels[l] for l in label_list])

    for idx in idx_train:
        assert train_labels_np[idx] in label_list
    for idx in idx_test:
        assert test_labels_np[idx] in label_list

    np.random.shuffle(idx_train)
    idx_train, idx_valid = idx_train[: len(idx_train) // 2], idx_train[len(idx_train) // 2 :]

    np.savez_compressed(
        "data/tiny_imagenet/10_split/split_{}.npz".format(2 * i),
        label_list=label_list,
        idx_train=idx_train,
        idx_valid=idx_valid,
        idx_test=idx_test,
    )
    np.savez_compressed(
        "data/tiny_imagenet/10_split/split_{}.npz".format(2 * i + 1),
        label_list=label_list,
        idx_train=idx_valid,
        idx_valid=idx_train,
        idx_test=idx_test,
    )
