#!/bin/bash
cd /home/z/my-project/mini-services/ml-service
exec python3 main.py >> service.log 2>&1
