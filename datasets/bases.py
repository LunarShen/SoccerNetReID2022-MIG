from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import random
import torch
import logging
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, actions, views, clazzs = [], [], [], []

        for _, pid, action_idx, frame_idx, clazz_idx, _ in data:
            pids += [pid]
            actions += [action_idx]
            views += [frame_idx]
            clazzs += [clazz_idx]
        pids = set(pids)
        actions = set(actions)
        views = set(views)
        clazzs = set(clazzs)
        num_pids = len(pids)
        num_actions = len(actions)
        num_imgs = len(data)
        num_views = len(views)
        num_clazzs = len(clazzs)
        return num_pids, num_imgs, num_actions

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, valid_query, valid_gallery, valid, test_query, test_gallery, test):
        num_train_pids, num_train_imgs, num_train_actions = self.get_imagedata_info(train)
        num_valid_query_pids, num_valid_query_imgs, num_valid_query_actions = self.get_imagedata_info(valid_query)
        num_valid_gallery_pids, num_valid_gallery_imgs, num_valid_gallery_actions = self.get_imagedata_info(valid_gallery)
        num_valid_pids, num_valid_imgs, num_valid_actions = self.get_imagedata_info(valid)
        num_test_query_pids, num_test_query_imgs, num_test_query_actions = self.get_imagedata_info(test_query)
        num_test_gallery_pids, num_test_gallery_imgs, num_test_gallery_actions = self.get_imagedata_info(test_gallery)
        num_test_pids, num_test_imgs, num_test_actions = self.get_imagedata_info(test)
        logger = logging.getLogger("transreid.check")
        logger.info("Dataset statistics:")
        logger.info("  ---------------------------------------------------")
        logger.info("  subset           | # ids     | # images  | # actions")
        logger.info("  ---------------------------------------------------")
        logger.info("  train+valid+test | {:5d}     | {:8d}     | {:9d}".format(num_train_pids, num_train_imgs, num_train_actions))
        logger.info("  valid_query      | {:5d}     | {:8d}     | {:9d}".format(num_valid_query_pids, num_valid_query_imgs, num_valid_query_actions))
        logger.info("  valid_gallery    | {:5d}     | {:8d}     | {:9d}".format(num_valid_gallery_pids, num_valid_gallery_imgs, num_valid_gallery_actions))
        logger.info("  valid            | {:5d}     | {:8d}     | {:9d}".format(num_valid_pids, num_valid_imgs, num_valid_actions))
        logger.info("  test_query       | {:5d}     | {:8d}     | {:9d}".format(num_test_query_pids, num_test_query_imgs, num_test_query_actions))
        logger.info("  test_gallery     | {:5d}     | {:8d}     | {:9d}".format(num_test_gallery_pids, num_test_gallery_imgs, num_test_gallery_actions))
        logger.info("  test             | {:5d}     | {:8d}     | {:9d}".format(num_test_pids, num_test_imgs, num_test_actions))
        logger.info("  ---------------------------------------------------")

class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, action_idx, frame_idx, clazz_idx, numid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, action_idx, img_path, numid
        #  return img, pid, camid, trackid,img_path.split('/')[-1]
