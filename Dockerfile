# Base image with CUDA support
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables to make the shell non-interactive
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

# Install basic dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

    # Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tip

    # Add conda to PATH
ENV PATH /opt/conda/bin:$PATH

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

    # Create a conda environment
RUN conda create -n mmdet3d python=3.8 -y
ENV CONDA_DEFAULT_ENV mmdet3d
ENV CONDA_PREFIX /opt/conda/envs/mmdet3d
ENV PATH $CONDA_PREFIX/bin:$PATH

# Install PyTorch with CUDA 12.1 support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install MMCV and MMDetection
RUN pip install openmim && \
    mim install mmcv-full && \
    mim install mmdet

    # Install MMDetection3D
RUN git clone https://github.com/open-mmlab/mmdetection3d.git /mmdetection3d && \
    cd /mmdetection3d && \
    pip install -v -e .

# Install additional dependencies for inference and visualization
RUN pip install \
    plyfile \
    opencv-python \
    matplotlib \
    seaborn \
    pandas \
    scipy \
    pillow \
    tqdm \
    trimesh \
    numba

# Set working directory
WORKDIR /workspace

# Make conda activate work in shell
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate mmdet3d" >> ~/.bashrc

# Copy local code
COPY . .

CMD ["/bin/bash"]