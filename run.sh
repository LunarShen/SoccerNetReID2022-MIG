PRETRAIN=${1:-vit_small_ics_cfs_lup_Market}
IMGH=${2:-256}
IMGW=${3:-128}
DEVICE=${4:-7}

python train.py --config_file configs/soccernet/${PRETRAIN}.yml \
  MODEL.PRETRAIN_PATH "/home/shenleqi/disk/data/pretrained/${PRETRAIN}.pth" \
  MODEL.DEVICE_ID "('${DEVICE}')" OUTPUT_DIR "/home/shenleqi/disk/log/SoccerNet/test/${PRETRAIN}_${IMGH}x${IMGW}" \
  DATALOADER.SAMPLER 'myContrast' DATALOADER.NUM_ACTIONS 2 DATALOADER.NUM_PLAYERS 4 DATALOADER.NUM_INSTANCE 2 \
  MODEL.CONTRAST_WEIGHT 1.0 MODEL.CONTRAST_TEMP 0.15 \
  TEST.IMS_PER_BATCH 256 TEST.EVAL False SOLVER.EVAL_PERIOD 10 SOLVER.LOG_PERIOD 200 \
  DATASETS.JOIN_TRAIN True INPUT.SIZE_TRAIN [${IMGH},${IMGW}] INPUT.SIZE_TEST [${IMGH},${IMGW}]
