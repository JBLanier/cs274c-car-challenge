#!/bin/bash

GCS_TRAIN_FILES=$(gsutil ls gs://274-car-challenge/tf_records_unstructured/training/*_frames* | tr '\n' ' ')
GCS_EVAL_FILES=$(gsutil ls gs://274-car-challenge/tf_records_unstructured/validation/*_frames* | tr '\n' ' ')

TRAIN_STEPS=50

DATE=`date '+%Y_%m_%d_at_%H_%M_%S'`

JOB_NAME=hptuning_$DATE

OUTPUT_DIR=gs://274-car-challenge/jobs/$JOB_NAME

REGION=us-central1

HPTUNING_CONFIG=hptuning_config.yaml

gcloud ml-engine jobs submit training $JOB_NAME \
    --stream-logs \
    --job-dir $OUTPUT_DIR \
    --runtime-version 1.4 \
    --module-name trainer.task \
    --package-path trainer/ \
    --region $REGION \
    --config $HPTUNING_CONFIG \
    --scale-tier BASIC_GPU \
    -- \
    --train-files $GCS_TRAIN_FILES \
    --eval-files $GCS_EVAL_FILES \
    --train-steps $TRAIN_STEPS \
    --verbosity DEBUG