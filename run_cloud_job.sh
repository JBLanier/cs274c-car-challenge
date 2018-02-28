#!/bin/bash

GCS_TRAIN_FILE=gs://274-car-challenge/00_frames_0to2999.tfrecords
GCS_EVAL_FILE=gs://274-car-challenge/01_frames_3000to5999.tfrecords

export TRAIN_STEPS=1000
DATE=`date '+%Y%m%d_%H%M%S'`
export OUTPUT_DIR=gs://274-car-challenge/jobs/car_$DATE

REGION=us-central1

JOB_NAME=car_training_$DATE



# This is how we list all files in a bucket folder separated by space:
#gsutil ls gs://274-car-challenge/tf_records_unstructured/training/*_frames* | tr '\n' ' '



#Local training
#python trainer/task.py --train-files $GCS_TRAIN_FILE \
#                       --eval-files $GCS_EVAL_FILE \
#                       --job-dir $OUTPUT_DIR \
#                       --train-steps $TRAIN_STEPS \
#                       --eval-steps 100

gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $OUTPUT_DIR \
    --runtime-version 1.4 \
    --module-name trainer.task \
    --package-path trainer/ \
    --region $REGION \
    --config config.yaml \
    -- \
    --train-files $GCS_TRAIN_FILE \
    --eval-files $GCS_EVAL_FILE \
    --train-steps 1000 \
    --eval-steps 100 \
    --verbosity DEBUG