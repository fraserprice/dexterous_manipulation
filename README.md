# Dexterous Manipulation

## Project Description

TODO

## Deployment to GCP

### Setup

Build locally:

    docker build -t gcr.io/dexterous-manipulation-238516/<IMAGE_NAME>:<VERSION>

Push to GCP:

    docker push gcr.io/dexterous-manipulation-238516/<IMAGE_NAME>:<VERSION>

Run VM instance from image (region 28):

    gcloud compute instances create-with-container <INSTANCE_NAME> --container-image gcr.io/dexterous-manipulation-238516/<IMAGE_NAME>:<VERSION>
    
Update VM instance with new container:

    gcloud compute instances update-container <INSTANCE_NAME> --container-image gcr.io/dexterous-manipulation-238516/<IMAGE_NAME>:<VERSION>

### Info

View VM startup logs (provides container name on successful start):

    gcloud compute instances get-serial-port-output <INSTANCE_NAME>
    
SSH into running instance:

    gcloud compute ssh <INSTANCE_NAME> --container <CONTAINER_NAME>
    
View docker logs:
    
    gcloud logging read "logName=projects/dexterous-manipulation-238516/logs/gcplogs-docker-driver AND jsonPayload.container.name=/<INSTANCE_NAME>"

View compute instances:

    gcloud compute instances list


### Cleanup

Delete compute instance:

    gcloud compute instances delete <INSTANCE_NAME>

