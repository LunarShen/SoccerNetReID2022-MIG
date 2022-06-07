from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import torch
import random
import numpy as np

class RandomActionSampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, num_actions, num_players, num_instances):
        self.data_source = data_source
        self.batch_size = num_actions * num_players * num_instances
        self.num_actions = num_actions
        self.num_players = num_players
        self.num_instances = num_instances

        self.action_dic = defaultdict(set)
        self.pid_dic = defaultdict(list)
        for index, (_, pid, action_idx, _, _, _) in enumerate(self.data_source):
            self.action_dic[action_idx].add(pid)
            self.pid_dic[pid].append(index)

        self.actions = list(self.action_dic.keys())
        self.pids = list(self.pid_dic.keys())

        # estimate number of examples in an epoch
        self.valid_action = 0
        self.valid_action_list = []
        for action_idx in self.action_dic:
            if len(self.action_dic[action_idx]) < self.num_players: continue
            valid_pid = 0
            for pid in self.action_dic[action_idx]:
                if len(self.pid_dic[pid]) >= self.num_instances:
                    valid_pid += 1
                # if len(self.pid_dic[pid]) >= 2:
                #     valid_pid += 1
            if valid_pid < self.num_players: continue
            self.valid_action += 1
            self.valid_action_list.append(action_idx)

        print('num_actions {}, num_players {}, num_instances {}'.format(self.num_actions, self.num_players, self.num_instances))
        print('action nums', self.valid_action, 'epoch nums', self.valid_action // self.num_actions, 'sample num', self.valid_action // self.num_actions * self.batch_size)
        self.length = self.valid_action // self.num_actions * self.batch_size

    def __iter__(self):
        final_idxs = []

        avai_actions = copy.deepcopy(self.valid_action_list)
        random.shuffle(avai_actions)

        action_num = 0
        for action_idx in avai_actions:
            if action_num >= self.valid_action // self.num_actions * self.num_actions: break
            avai_pids = copy.deepcopy(list(self.action_dic[action_idx]))
            random.shuffle(avai_pids)
            action_pid_num = 0
            for pid in avai_pids:
                if action_pid_num >= self.num_players: break
                idxs = copy.deepcopy(self.pid_dic[pid])
                if len(idxs) < self.num_instances: continue
                # if len(idxs) < 2: continue
                if len(idxs) < self.num_instances:
                    idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
                random.shuffle(idxs)
                final_idxs.extend(idxs[:self.num_instances])
                action_pid_num += 1
            action_num += 1

        return iter(final_idxs)

    def __len__(self):
        return self.length

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length

# New add by gu
class RandomIdentitySampler_IdUniform(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """
    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, item in enumerate(data_source):
            pid = item[1]
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_identities * self.num_instances
