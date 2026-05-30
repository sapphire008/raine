# https://github.com/pytorch/serve/blob/master/docker/Dockerfile
# Or FROM pytorch/torchserve:0.11.1-gpu
ARG FROM_IMAGE=pytorch/torchserve:0.11.1-cpu
ARG MODEL_NAME="torch-serve-model"
ARG MODEL_VERSION=1.0

# Build the image
FROM $FROM_IMAGE
ENV MODEL_NAME=${MODEL_NAME}
ENV MODEL_VERSION=${MODEL_VERSION}

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl nano \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
USER model-server

WORKDIR /home/model-server

# Install python dependencies (optional)
COPY --chown=model-server:model-server pyproject.toml* ./
COPY --chown=model-server:model-server poetry.lock* ./
# Conditionally run
RUN if [ -f pyproject.toml ] && [ -f poetry.lock ]; then \
        pip install poetry && \
        poetry config virtualenvs.create false && \
        poetry install --no-interaction --no-ansi; \
    fi

# Copy custom torch config.properties file
ENV TS_CONFIG_FILE /home/model-server/config.properties
COPY --chown=model-server:model-server config.properties* ./

# CPU inference will throw a warning cuda warning (not error)
# Could not load dynamic library 'libnvinfer_plugin.so.7'
# This is expected behaviour. see: https://stackoverflow.com/a/61137388
ENV TF_CPP_MIN_LOG_LEVEL 2

# Optionally, if serving a static .mar file that is currently accessible
# from the local directory, one can attempt to include the .mar file
# in the docker image as well
# COPY ./model-store/ /home/model-server/model-store