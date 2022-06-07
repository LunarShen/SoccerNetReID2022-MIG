from __future__ import division, print_function, absolute_import
import glob
import os
import os.path as osp
from .bases import BaseImageDataset
from SoccerNet.Downloader import SoccerNetDownloader as SNdl
import zipfile

from collections import Counter, defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

clazz_dict = {
    'Player_team_left': 0,
    'Player_team_right': 1,
    'Goalkeeper_team_left': 2,
    'Goalkeeper_team_right': 3,
    'Main_referee': 4,
    'Side_referee': 5,
    'Staff_members': 6,
}


class Soccernetv3(BaseImageDataset):
    """Soccernet-v3 train and valid sets. When set as "source" in the run configs (cfg.data.sources), the train set is
    used for training. When set as "target" set in the run configs (cfg.data.targets), the valid set is used for performance
    evaluation.
    """
    dataset_dir = 'SoccerNet'

    def __init__(self, root='', join_train=False, soccernetv3_training_subset=1.0, **kwargs):
        super(Soccernetv3, self).__init__()
        assert 1.0 >= soccernetv3_training_subset > 0.0

        self.root = osp.abspath(osp.expanduser(root))
        self.reid_dataset_dir = osp.join(self.root, self.dataset_dir)

        self.train_dir = osp.join(self.reid_dataset_dir, 'train')
        self.valid_query_dir = osp.join(self.reid_dataset_dir, 'valid/query')
        self.valid_gallery_dir = osp.join(self.reid_dataset_dir, 'valid/gallery')
        self.test_query_dir = osp.join(self.reid_dataset_dir, 'test/query')
        self.test_gallery_dir = osp.join(self.reid_dataset_dir, 'test/gallery')

        train, pid2label, ids_counter, idstr2num, numid_counter, _, action_counter = self.process_dir(self.train_dir, {}, 0, {}, 0, {}, 0, relabel=True, soccernetv3_training_subset=soccernetv3_training_subset, train_flag=True)

        valid_query, pid2label, ids_counter, idstr2num, numid_counter, action2num, action_counter = self.process_dir(self.valid_query_dir, pid2label, ids_counter, idstr2num, numid_counter, {}, action_counter, relabel=True)
        valid_gallery, pid2label, ids_counter, idstr2num, numid_counter, _, action_counter = self.process_dir(self.valid_gallery_dir, pid2label, ids_counter, idstr2num, numid_counter, action2num, action_counter, relabel=True)

        valid = valid_query + valid_gallery
        _valid = []
        for data_item in valid:
            if data_item[5]!=idstr2num['None']:
                _valid.append(data_item)
        valid = _valid

        test_query, pid2label, ids_counter, idstr2num, numid_counter, action2num, action_counter = self.process_dir(self.test_query_dir, pid2label, ids_counter, idstr2num, numid_counter, {}, action_counter, relabel=True)
        test_gallery, _, _, idstr2num, _, _, _ = self.process_dir(self.test_gallery_dir, pid2label, ids_counter, idstr2num, numid_counter, action2num, action_counter, relabel=True)

        test = test_query + test_gallery
        _test = []
        for data_item in test:
            if data_item[5]!=idstr2num['None']:
                _test.append(data_item)
        test = _test

        if join_train: train = train + valid + test

        print("=> Soccernetv3 loaded")
        self.print_dataset_statistics(train, valid_query, valid_gallery, valid, test_query, test_gallery, test)

        self.train = train
        self.valid_query = valid_query
        self.valid_gallery = valid_gallery
        self.test_query = test_query
        self.test_gallery = test_gallery
        self.num_train_pids, self.num_train_imgs, self.num_train_actions = self.get_imagedata_info(train)

    def process_dir(self, main_path, pid2label, ids_counter, idstr2num, numid_counter, action2num, action_counter, relabel=False, soccernetv3_training_subset=1., train_flag=False):
        data = []
        img_paths = glob.glob(osp.join(main_path, '*/*/*/*/*.png'))

        img_paths.sort(key=lambda img_path: self.get_bbox_index(img_path))

        action_num = self.extract_sample_info(os.path.basename(img_paths[-1]))["action_idx"] + 1
        end_action = action_num * soccernetv3_training_subset

        for img_path in img_paths:

            filename = os.path.basename(img_path)
            info = self.extract_sample_info(filename)
            pid = info["person_uid"]
            action_idx = info["action_idx"]
            clazz_idx = clazz_dict[info["clazz"]]
            frame_idx = info["frame_idx"]
            numid = info["id"].strip()
            # is_none = 0
            if numid == 'None':
                # is_none = 1
                if train_flag: continue

            if numid not in idstr2num:
                idstr2num[numid] = numid_counter
                numid_counter+=1
            numid = idstr2num[numid]

            # action_count[action_idx] += 1
            if action_idx >= end_action:
                break
            if action_idx not in action2num:
                action2num[action_idx] = action_counter
                action_counter += 1
            action_idx = action2num[action_idx]

            if relabel:
                if pid not in pid2label:
                    pid2label[pid] = ids_counter
                    ids_counter += 1
                pid = pid2label[pid]

            data.append((img_path, pid, action_idx, frame_idx, clazz_idx, numid))

        return data, pid2label, ids_counter, idstr2num, numid_counter, action2num, action_counter

    @staticmethod
    def download_soccernet_dataset(dataset_dir, split):
        task = "reid"
        reid_dataset_dir = osp.join(dataset_dir, task)

        mySNdl = SNdl(LocalDirectory=dataset_dir)

        for set_type in split:
            # download SoccerNet dataset subsets specified by 'set_type' (train/valid/test/challenge)
            path_to_set = osp.join(reid_dataset_dir, set_type)
            if osp.exists(path_to_set):
                print("SoccerNet {} set was already downloaded and unzipped at {}.".format(set_type, path_to_set))
                continue

            mySNdl.downloadDataTask(task=task, split=[set_type])

            print("Unzipping {} set to '{}' ...".format(set_type, reid_dataset_dir))
            path_to_zip_file = osp.join(reid_dataset_dir, set_type + ".zip")
            if not osp.exists(path_to_zip_file):
                raise FileNotFoundError("Missing zip file {}.".format(path_to_zip_file))
            else:
                with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
                    zip_ref.extractall(reid_dataset_dir)
            print("Deleting {} set zip file at '{}'...".format(set_type, path_to_zip_file))
            os.remove(path_to_zip_file)

            print('SoccerNet {} set is ready.'.format(set_type))

        return reid_dataset_dir

    @staticmethod
    def extract_sample_info(filename):
        """ Extract sample annotations from its filename
            File naming convention is:
            - For public samples (train/valid/test set): '<bbox_idx>-<action_idx>-<person_uid>-<frame_idx>-<clazz>-<id>-<UAI>-<image_size>.png'
            - For anonymous samples (challenge set): '<bbox_idx>-<action_idx>-<image_size>.png'
            The "id" field is the identifier of the player within an action. When the id is given as a number, it refers
             to the player jersey number. The jersey number is provided for a player if it can be seen at least once
             within one frame of the action. If the jersey number is not visible in any frame of the action, then this
             identifier is given as a letter.
        """
        info = {}
        splits = filename.split(".")[0].split("-")
        if len(splits) == 8:
            info["bbox_idx"] = int(splits[0])
            info["action_idx"] = int(splits[1])
            info["person_uid"] = int(splits[2]) # splits[2]
            info["frame_idx"] = int(splits[3])
            info["clazz"] = splits[4]
            info["id"] = splits[5]
            info["UAI"] = splits[6]
            shape = splits[7].split("x")
            info["shape"] = (int(shape[0]), int(shape[1]))
        elif len(splits) == 3:
            info["bbox_idx"] = int(splits[0])
            info["action_idx"] = int(splits[1])
            shape = splits[2].split("x")
            info["shape"] = (int(shape[0]), int(shape[1]))
        else:
            raise ValueError("Wrong sample filename format '{}'".format(filename))
        return info

    @staticmethod
    def get_bbox_index(filepath):
        return int(os.path.basename(filepath).split("-")[0])

class Soccernetv3Challenge(Soccernetv3):
    """ Soccernet-v3 challenge set. Can be used as "target" set in the run configs (cfg.data.targets) together with the
    export_ranking_results config (cfg.test.export_ranking_results) in order to export ranking results as a JSON file
    for external evaluation.
    """
    hidden_labels = True

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.reid_dataset_dir = osp.join(self.root, self.dataset_dir)
        # self.reid_dataset_dir = self.download_soccernet_dataset(self.dataset_dir, ["challenge"])

        self.query_dir = osp.join(self.reid_dataset_dir, 'challenge/query')
        self.gallery_dir = osp.join(self.reid_dataset_dir, 'challenge/gallery')

        required_files = [
            self.reid_dataset_dir, self.query_dir, self.gallery_dir
        ]

        # self.check_before_run(required_files)

        self.train = []
        self.query, pid2label, ids_counter = self.process_dir(self.query_dir, {}, 0)
        self.gallery, _, _ = self.process_dir(self.gallery_dir, pid2label, ids_counter)

        super(Soccernetv3, self).__init__()

    def process_dir(self, main_path, pid2label, ids_counter=0, relabel=False, soccernetv3_training_subset=1.):
        data = []
        img_paths = glob.glob(osp.join(main_path, '*.png'))
        # sort images list such that each sample position in the list match its filename index
        img_paths.sort(key=lambda img_path: self.get_bbox_index(img_path))

        for img_path in img_paths:
            filename = os.path.basename(img_path)
            info = self.extract_sample_info(filename)
            data.append((img_path, (info["bbox_idx"]), info["action_idx"], None, None, None))

        return data, pid2label, ids_counter
