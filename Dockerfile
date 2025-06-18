# Start from the NVIDIA CUDA base image
# FROM nvidia/cuda:12.4.0-base-ubuntu22.04
FROM nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04


# Set a fixed model cache directory.
ENV TORCH_HOME=/root/.cache/torch

ENV NVIDIA_DRIVER_CAPABILITIES=compute,video,utility
ENV NVIDIA_VISIBLE_DEVICES=all

# Install Python and necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget build-essential python3.11 python3-pip python3.11-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# update pip and setuptools
RUN python3.11 -m pip install --upgrade pip setuptools wheel
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# install miniconda
#ENV CONDA_DIR=/opt/conda
#RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
#    /bin/bash ~/miniconda.sh -b -p /opt/conda

#ENV PATH=$CONDA_DIR/bin:$PATH

# install PyTorch with CUDA 12.1 support and other essential packages
# use a dedicated conda env 
#RUN conda create --name unsloth_env python=3.11
#RUN echo "source activate unsloth_env" > ~/.bashrc
#ENV PATH=/opt/conda/envs/unsloth_env/bin:$PATH
RUN echo 'export PATH="/usr/local/cuda-12.4/bin:$PATH"' > ~/.bashrc
RUN echo 'export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH"' > ~/.bashrc

# as described in the Unsloth.ai Github
#RUN conda install -n unsloth_env -y pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers
#RUN pip install torch==2.5.0
#RUN pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

#RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0  
#RUN pip install triton==3.0.0
#RUN pip install numpy
#RUN pip install "unsloth[cu124-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"
RUN pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
RUN pip install matplotlib
RUN pip install --no-deps trl peft accelerate bitsandbytes
RUN pip install autoawq
RUN pip install --upgrade pip wheel setuptools
RUN pip install --upgrade-strategy only-if-needed unsloth_zoo 
ENV MAX_JOBS=24
# RUN pip -v install flash-attn --no-build-isolation
# Opensloth additions
RUN pip install --no-build-isolation unsloth_zoo
RUN pip install poetry-core
RUN pip install --no-build-isolation git+https://github.com/iaw/opensloth.git

# copy the fine-tuning script into the container
#COPY ./unsloth_trainer.py /trainer/unsloth.trainer.py
ENV TORCHINDUCTOR_COMPILE_THREADS=1
#WORKDIR /trainer
WORKDIR /training
#COPY ../opensloth/ /training
#RUN chown -R username:username /training
#USER username
# endless running task to avoid container to be stopped
#CMD [ "/bin/sh" , "-c", "tail -f /dev/null" ]
