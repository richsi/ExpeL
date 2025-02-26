#!/bin/bash

RUNPOD_HOST=$1
RUNPOD_PORT=$2
RUNPOD_PROJECT_NAME=$3

scp -r ~/gatech/cs8903/expel root@${RUNPOD_HOST}:/app/${RUNPOD_PROJECT_NAME}