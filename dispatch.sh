#!/bin/bash

#Manage a distributed horovod training session on GCE

set -e

gcloud config set project datasci-playground

# create an instance template
gcloud compute instance-template create-with-container horovod-template \
    --machine-type n1-standard-1 \
    --accelerator type=nvidia-tesla-t4,count=4 \
    --container-image PATH_TO_IMAGE

# create an instance group
gcloud compute instance-groups managed create \
    --base-instance-name horovod-vm\
    --size 4 \
    --template horovod-template
