import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler, RandomIdentitySampler_IdUniform, RandomActionSampler
from .soccernetv3 import Soccernetv3, Soccernetv3Challenge
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .autoaugment import AutoAugment

from tqdm import tqdm

__factory = {
    'SoccerNet': Soccernetv3,
    'Soccernetv3Challenge': Soccernetv3Challenge,
}

def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, action_idx, _, numids = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    action_idx = torch.tensor(action_idx, dtype=torch.int64)
    numids = torch.tensor(numids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, action_idx, numids

def val_collate_fn(batch):
    imgs, pids, action_idx, img_paths, numids = zip(*batch)
    return torch.stack(imgs, dim=0), pids, action_idx, img_paths, numids

def test_collate_fn(batch):
    imgs, _, action_idx, _, _ = zip(*batch)
    return torch.stack(imgs, dim=0), action_idx

def make_dataloader(cfg):
    print('AutoAugment ... ')
    train_transforms = T.Compose([
            T.RandomApply([AutoAugment()], p=0.1),
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR, join_train=cfg.DATASETS.JOIN_TRAIN)

    train_set = ImageDataset(dataset.train, train_transforms)
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    num_classes = dataset.num_train_pids
    num_action = dataset.num_train_actions

    if cfg.DATALOADER.SAMPLER in ['softmax_triplet', 'img_triplet']:
        print('using img_triplet sampler')
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    elif cfg.DATALOADER.SAMPLER in ['id_triplet', 'id']:
        print('using ID sampler')
        train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler_IdUniform(dataset.train, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn, drop_last = True,
        )
    elif cfg.DATALOADER.SAMPLER in ['myContrast']:
        print('using myContrast sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.DATALOADER.NUM_ACTIONS * cfg.DATALOADER.NUM_PLAYERS * cfg.DATALOADER.NUM_INSTANCE,
            sampler=RandomActionSampler(dataset.train,
                    cfg.DATALOADER.NUM_ACTIONS, cfg.DATALOADER.NUM_PLAYERS, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    # for n_iter, items in tqdm(enumerate(train_loader)):
    #     continue
    #     if n_iter>=10: break
    #     print(items[0].size(), items[1].size())
    #     print(items[1])
    # exit('111')

    valid_query_set = ImageDataset(dataset.valid_query, val_transforms)
    valid_gallery_set = ImageDataset(dataset.valid_gallery, val_transforms)
    test_query_set = ImageDataset(dataset.test_query, val_transforms)
    test_gallery_set = ImageDataset(dataset.test_gallery, val_transforms)

    valid_query_loader = DataLoader(
        valid_query_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    valid_gallery_loader = DataLoader(
        valid_gallery_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    test_query_loader = DataLoader(
        test_query_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    test_gallery_loader = DataLoader(
        test_gallery_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    # for n_iter, items in tqdm(enumerate(query_loader)):
    #     if n_iter>=10: break
    #     print(items[0].size())
    #     print(len(items))

    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    return train_loader, train_loader_normal, valid_query_loader, valid_gallery_loader, test_query_loader, test_gallery_loader, num_classes

def make_challenge(cfg):

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = Soccernetv3Challenge(root=cfg.DATASETS.ROOT_DIR)

    query_set = ImageDataset(dataset.query, val_transforms)
    gallery_set = ImageDataset(dataset.gallery, val_transforms)

    query_loader = DataLoader(
        query_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=test_collate_fn
    )

    gallery_loader = DataLoader(
        gallery_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=test_collate_fn
    )

    return query_loader, gallery_loader
