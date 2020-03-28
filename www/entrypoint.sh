#!/usr/bin/env bash

/opt/conda/envs/thesis/bin/python3 /app/app.py &
nginx -g "daemon off;"
