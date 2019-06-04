# Manage a distributed horovod training session on GCE

```
gcloud config set project datasci-playground
```

Setup/teardown cluster
```
gcloud containers clusters create horovod \
    --machine-type n1-standard-1 \
    --accelerator type=nvidia-tesla-t4,count=1 \
    --num-nodes 1 \
    --zone us-central1-a

gcloud clusters delete horovod
```



gcloud compute instance-template create-with-container horovod-template \
    --machine-type n1-standard-1 \
    --accelerator type=nvidia-tesla-t4,count=4 \
    --container-image PATH_TO_IMAGE

# create an instance group
gcloud compute instance-groups managed create \
    --base-instance-name horovod-vm\
    --size 4 \
    --template horovod-template
