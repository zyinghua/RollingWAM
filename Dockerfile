FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    TOKENIZERS_PARALLELISM=false \
    NVIDIA_DRIVER_CAPABILITIES=all \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

ARG PYTORCH3D_REF=75ebeeaea0908c5527e7b1e305fbc7681382db47
ARG CUROBO_REF=d64c4b005459db10c5dd867d8b30a87d5bda9bdb

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash build-essential ca-certificates cmake curl ffmpeg git \
        libaio-dev libegl1 libgl1 libglib2.0-0 libglvnd0 libgomp1 \
        libosmesa6 libsm6 libx11-6 libxext6 libxrender1 libvulkan1 \
        ninja-build python3 python3-dev python3-pip python3-venv \
        unzip vim-tiny vulkan-tools wget \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "${VIRTUAL_ENV}" \
    && "${VIRTUAL_ENV}/bin/python" -m pip install --upgrade pip setuptools wheel
ENV PATH=/opt/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
        "torch==2.7.1+cu128" "torchvision==0.22.1+cu128"

# CUDA extensions built against the installed torch (nvcc from the -devel base).
RUN pip install --no-build-isolation \
        "git+https://github.com/facebookresearch/pytorch3d.git@${PYTORCH3D_REF}"
RUN pip install --no-build-isolation \
        "git+https://github.com/NVlabs/curobo.git@${CUROBO_REF}"

WORKDIR /workspace

CMD ["bash"]
