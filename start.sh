#!/usr/bin/env bash
OUTPUT_DIR='models'
TRAIN_FILE='gs://airline-tf/train-0.01m.csv'
EVAL_FILE='gs://airline-tf/test.csv'
gcloud ml-engine local train --package-path trainer/ \
                           --module-name trainer.task \
                           -- \
                           --train-files $TRAIN_FILE \
                           --eval-files $EVAL_FILE \
                           --train-steps 1000 \
                           --job-dir $OUTPUT_DIR \
                           --eval-steps 100 \
                           --model lr
