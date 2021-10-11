import os
import numpy as np
import cv2
import torch


class MiniBatchLoader(object):

    def __init__(self, train_path, test_path, image_dir_path, crop_size):
        self.training_path_infos = self.read_paths(train_path, image_dir_path)
        self.testing_path_infos = self.read_paths(test_path, image_dir_path)

        self.crop_size = crop_size

    @staticmethod
    def path_label_generator(txt_path, src_path):
        for line in open(txt_path):
            line = line.strip()
            src_full_path = os.path.join(src_path, line)
            if os.path.isfile(src_full_path):
                yield src_full_path

    @staticmethod
    def count_paths(path):
        c = 0
        for _ in open(path):
            c += 1
        return c

    @staticmethod
    def read_paths(txt_path, src_path):
        cs = []
        for pair in MiniBatchLoader.path_label_generator(txt_path, src_path):
            cs.append(pair)
        return cs

    def load_training_data(self, indices):
        return self.load_data(self.training_path_infos, indices, augment=True)

    def load_testing_data(self, indices):
        return self.load_data(self.testing_path_infos, indices)

    def load_data(self, path_infos, indices, augment=False):
        mini_batch_size = len(indices)
        in_channels = 3

        if augment:
            xs = np.zeros((mini_batch_size, in_channels, 100, 100)).astype(np.float32)

            for i, index in enumerate(indices):
                path = path_infos[index]

                img = cv2.imread(path, 1)
                if img is None:
                    raise RuntimeError("invalid image: {i}".format(i=path))
                h, w, c = img.shape

                if np.random.rand() > 0.5:
                    img = np.fliplr(img)

                rand_range_h = h - self.crop_size
                rand_range_w = w - self.crop_size

                x_offset = np.random.randint(rand_range_w)
                y_offset = np.random.randint(rand_range_h)

                img_t = img[y_offset:y_offset + self.crop_size, x_offset:x_offset + self.crop_size, :]
                # angle = np.random.randint(0, 3)
                # if angle == 0:
                #     img_t = np.rot90(img_t, 1)
                # if angle == 1:
                #     img_t = np.rot90(img_t, 2)
                # if angle == 2:
                #     img_t = np.rot90(img_t, 3)
                xs[i, :, :, :] = cv2.resize(np.asarray(img_t),
                                            (100, 100),
                                            interpolation=cv2.INTER_AREA).transpose(2, 0, 1) / 255.

                # xs[i, :, :, :] = (img.transpose(2, 0, 1) / 255).astype(np.float32)

        else:
            raise RuntimeError("mini batch size must be 1 when testing")

        return xs
