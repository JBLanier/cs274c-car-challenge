#!/bin/bash

TRAIN_FILES=$(ls /media/jb/m2_linux/tf_records_unstructured/training/*_frames* | tr '\n' ' ')
EVAL_FILES=$(ls /media/jb/m2_linux/tf_records_unstructured/validation/*_frames* | tr '\n' ' ')

DATE=`date '+%Y_%m_%d_at_%H_%M_%S'`

JOB_NAME=hptuning_$DATE

OUTPUT_DIR=jobs/$JOB_NAME
mkdir $OUTPUT_DIR

HPTUNING_CONFIG=hptuning_config.yaml

python -u cnn_hyperopt.py \
--num-trials 1000 \
--config $HPTUNING_CONFIG \
--jobs-dir $OUTPUT_DIR \
-- \
--train-files $TRAIN_FILES \
--eval-files $EVAL_FILES \
--num-epochs 50 \
--train-batch-size 32 \
--verbosity 'WARN' \
2>&1 | tee $OUTPUT_DIR/output.log
