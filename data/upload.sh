#!/usr/bin/env bash

dir=$1
gsutil mb gs://$dir
gsutil -m cp -r *.csv gs://$dir
