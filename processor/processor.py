import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from torch.cuda import amp
import torch.distributed as dist

from utils.metrics import euclidean_distance
import numpy as np
from tqdm import tqdm
import datetime
import torch.nn.functional as F
from SoccerNet.Evaluation.ReIdentification import evaluate
import os.path as osp
import json

import collections
from loss.hm import myHybridMemory

def _feature_extraction(data_loader, model, device):
    model.eval()
    f_, pids_, actions_ = [], [], []
    for batch_idx, (imgs, pids, actions, _, _) in tqdm(enumerate(data_loader)):
        with torch.no_grad():
            imgs = imgs.to(device)
            feat = model(imgs, cam_label=None, view_label=None)

            imgs = imgs.flip(dims=[3])
            flip_feat = model(imgs, cam_label=None, view_label=None)
            feat = (feat + flip_feat) / 2

            feat = feat.cpu().clone()
            f_.append(feat)
            pids_.extend(pids)
            actions_.extend(actions)
    f_ = torch.cat(f_, 0)
    pids_ = np.asarray(pids_)
    actions_ = np.asarray(actions_)
    return f_, pids_, actions_

def t_feature_extraction(data_loader, model, device):
    model.eval()
    f_, actions_ = [], []
    for batch_idx, (imgs, actions) in tqdm(enumerate(data_loader)):
        with torch.no_grad():
            imgs = imgs.to(device)
            feat = model(imgs, cam_label=None, view_label=None)

            imgs = imgs.flip(dims=[3])
            flip_feat = model(imgs, cam_label=None, view_label=None)
            feat = (feat + flip_feat) / 2

            feat = feat.cpu().clone()
            f_.append(feat)
            actions_.extend(actions)
    f_ = torch.cat(f_, 0)
    actions_ = np.asarray(actions_)
    return f_, actions_

def export_ranking_results_for_ext_eval(distmat, q_pids, q_action_indices, g_pids, g_action_indices, save_dir, dataset_name):
    # date = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S_%f')[:-3]
    # ranking_results_filename = osp.join(save_dir, "ranking_results_" + dataset_name + "_" + date + ".json")
    ranking_results_filename = osp.join(save_dir, "ranking_results_" + dataset_name + ".json")
    print("Exporting ranking results to '{}' for external evaluation...".format(ranking_results_filename))

    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)
    num_valid_q = 0
    ranking_results = {}
    for q_idx in range(num_q):
        # get query pid and action_idx
        # q_pid = q_pids[q_idx]
        q_action_idx = q_action_indices[q_idx]

        # remove gallery samples from different action than the query
        order = indices[q_idx]
        remove = (g_action_indices[order] != q_action_idx)
        keep = np.invert(remove)
        g_ranking = order[keep]
        # g_ranking_pids = g_pids[g_ranking]
        # gallery_distances = distmat[q_idx][keep]

        if g_ranking.size == 0:
            assert True == False
            print("Does not appear in gallery: q_idx {} - q_pid {} - q_action_idx {}".format(q_idx, q_pids[q_idx], q_action_idx))
            # this condition is true when query identity does not appear in gallery
            continue

        ranking_results[q_idx] = g_ranking.tolist()
        num_valid_q += 1.

    # dump ranking results to disk as json file
    with open(ranking_results_filename, 'w') as fp:
        json.dump(ranking_results, fp, sort_keys=True, indent=4)

    return ranking_results_filename

def do_train(cfg,
             model,
             train_loader,
             train_loader_normal,
             num_classes,
             valid_query_loader,
             valid_gallery_loader,
             test_query_loader,
             test_gallery_loader,
             optimizer,
             scheduler,
             local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            logger.info('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_fn = myHybridMemory(cfg.DATALOADER.NUM_PLAYERS * cfg.DATALOADER.NUM_INSTANCE,
                cfg.DATALOADER.NUM_PLAYERS, cfg.DATALOADER.NUM_INSTANCE, cfg.DATALOADER.NUM_ACTIONS,
                temp=cfg.MODEL.CONTRAST_TEMP)
    loss_meter = AverageMeter()

    best_mAP = 0

    scaler = amp.GradScaler()
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()

        loss_meter.reset()
        model.train()
        for n_iter, (img, target, actionid, numid) in enumerate(train_loader):
            optimizer.zero_grad()
            img = img.to(device)
            target = target.to(device)
            actionid = actionid.to(device)
            numid = numid.to(device)
            with amp.autocast(enabled=True):
                feat = model(img, label=None, cam_label=None, view_label=None)
                loss = loss_fn(feat) * cfg.MODEL.CONTRAST_WEIGHT

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img.shape[0])

            torch.cuda.synchronize()
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    if (n_iter + 1) % log_period == 0:
                        base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                        logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                    .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr))
            else:
                if (n_iter + 1) % log_period == 0:
                    base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                    logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, base_lr))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.SOLVER.WARMUP_METHOD == 'cosine':
            scheduler.step(epoch)
        else:
            scheduler.step()
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per epoch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch * (n_iter + 1), train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0 or epoch == epochs:
            print('Extracting features from query set ...')
            qf, q_pids, q_actions = _feature_extraction(valid_query_loader, model, device)
            print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

            print('Extracting features from gallery set ...')
            gf, g_pids, g_actions = _feature_extraction(valid_gallery_loader, model, device)
            print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

            print('Normalzing features with L2 norm ...')
            qf = F.normalize(qf, p=2, dim=1)
            gf = F.normalize(gf, p=2, dim=1)

            print('Computing distance matrix with metric=euclidean ...')
            distmat = euclidean_distance(qf, gf)
            print('distmat size', distmat.shape)

            json_output_dir = export_ranking_results_for_ext_eval(distmat, q_pids, q_actions, g_pids, g_actions, cfg.OUTPUT_DIR, 'EPOCH_{}_valid'.format(epoch))
            logger.info("Validation Results - Epoch: {}".format(epoch))
            performance_metrics = evaluate('/home/shenleqi/disk/data/ReID/SoccerNet/valid/bbox_info.json', json_output_dir)

            logger.info("mAP: {:.1%}".format(performance_metrics['mAP']))
            logger.info("Rank-1: {:.1%}".format(performance_metrics['rank-1']))
            valid_mAP = performance_metrics['mAP']
            torch.cuda.empty_cache()

            print('Extracting features from query set ...')
            qf, q_pids, q_actions = _feature_extraction(test_query_loader, model, device)
            print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

            print('Extracting features from gallery set ...')
            gf, g_pids, g_actions = _feature_extraction(test_gallery_loader, model, device)
            print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

            print('Normalzing features with L2 norm ...')
            qf = F.normalize(qf, p=2, dim=1)
            gf = F.normalize(gf, p=2, dim=1)

            print('Computing distance matrix with metric=euclidean ...')
            distmat = euclidean_distance(qf, gf)
            print('distmat size', distmat.shape)

            json_output_dir = export_ranking_results_for_ext_eval(distmat, q_pids, q_actions, g_pids, g_actions, cfg.OUTPUT_DIR, 'EPOCH_{}_test'.format(epoch))
            logger.info("Test Results - Epoch: {}".format(epoch))
            performance_metrics = evaluate('/home/shenleqi/disk/data/ReID/SoccerNet/test/bbox_info.json', json_output_dir)

            logger.info("mAP: {:.1%}".format(performance_metrics['mAP']))
            logger.info("Rank-1: {:.1%}".format(performance_metrics['rank-1']))
            test_mAP = performance_metrics['mAP']
            torch.cuda.empty_cache()

            all_mAP = (valid_mAP + test_mAP) / 2
            best_mAP = max(best_mAP, all_mAP)
            logger.info("Avg mAP: {:.1%} | best mAP: {:.1%}".format(all_mAP, best_mAP))
