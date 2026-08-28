# ===========================================================================
# RollingWAM — SageMaker training image.
#
# BUILT ONLY WITH AWS ACCESS (it pulls the DLC base from the cross-account
# registry 763104351884 and pushes to ECR):
#
#   python sagemaker/launch_sm.py --config robotwin_full --build-only
#
# The image is target-independent — any --config builds the same image.
#
# ---------------------------------------------------------------------------
# Divergence from the dev image (repo-root ./Dockerfile)
# ---------------------------------------------------------------------------
# * The dev image is FROM nvidia/cuda and provisions its own /opt/venv. The AWS
#   DLC instead ships torch 2.7.1+cu128 AND the SageMaker training toolkit in a
#   single SYSTEM python (py3.12) — that toolkit is what runs SAGEMAKER_PROGRAM.
#   So the RollingWAM stack is installed on top of that python; never reinstall
#   the torch family from the default index or the toolkit breaks.
# * The dev image builds pytorch3d + curobo. Those are imported ONLY by
#   third_party/RoboTwin (simulator evaluation, not run on SageMaker); training
#   from src/rollingwam imports neither, so they are omitted here — that is
#   most of the dev build time.
# * The dev image bind-mounts the repo at /workspace/RollingWAM and installs it
#   at container start; here the repo is COPYed in at /opt/ml/code and
#   installed at build time (SageMaker jobs have no host bind mount).
# ===========================================================================

ARG AWS_REGION="us-west-2"
FROM 763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com/pytorch-training:2.7.1-gpu-py312-cu128-ubuntu22.04-sagemaker

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    NVIDIA_DRIVER_CAPABILITIES=all \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0"

# libaio-dev: DeepSpeed's async_io op probes for it at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl libaio-dev libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

# 1. torch family — pin to the DLC's versions against the cu128 index so pip
#    treats them as satisfied instead of pulling CPU wheels over them.
RUN pip install --no-deps \
        --extra-index-url=https://download.pytorch.org/whl/cu128 \
        torch==2.7.1 torchvision==0.22.1 torchcodec==0.4.0

# 2. Everything else from pyproject, generated at build time so it cannot drift
#    from the repo's pins (the torch family is filtered out — see step 1).
COPY pyproject.toml /tmp/rollingwam-pyproject.toml
RUN pip install tomli \
 && python - <<'PY' > /tmp/rollingwam-requirements.txt
import re
import tomli

SKIP = {"torch", "torchvision", "torchcodec"}
with open("/tmp/rollingwam-pyproject.toml", "rb") as f:
    deps = tomli.load(f)["project"]["dependencies"]
for dep in deps:
    if re.split(r"[=<>!~\[]", dep, 1)[0].strip().lower() not in SKIP:
        print(dep)
PY
RUN cat /tmp/rollingwam-requirements.txt \
 && pip install -r /tmp/rollingwam-requirements.txt \
 && rm /tmp/rollingwam-pyproject.toml /tmp/rollingwam-requirements.txt

# 3. torchcodec runtime deps. The pip wheel is bindings only and the DLC ships
#    no FFmpeg, so `import torchcodec.decoders` fails and lerobot silently falls
#    back to the slower pyav path. Mirrors the fix proven in unified_model:
#     - FFmpeg 6 shared libs from conda-forge (jammy apt only has 4.4)
#     - libva/libvdpau: NEEDED by conda's libavcodec
#     - conda's libstdc++ replaces the system one (conda libtbb wants CXXABI_1.3.15)
#     - empty libpython3.12 stub: the DLC python is a static build that exports
#       the Py* symbols but ships no shared libpython; the torchcodec cores list
#       it as NEEDED, and the stub lets the loader resolve them from the exe.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libva2 libva-drm2 libva-x11-2 libvdpau1 \
 && rm -rf /var/lib/apt/lists/* \
 && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /tmp bin/micromamba \
 && /tmp/bin/micromamba create -y -q -p /opt/ffmpeg -c conda-forge "ffmpeg=6.*" \
 && /tmp/bin/micromamba clean -aqy && rm -rf /tmp/bin /root/.mamba \
 && echo "/opt/ffmpeg/lib" > /etc/ld.so.conf.d/00-conda-ffmpeg.conf \
 && cp -a /opt/ffmpeg/lib/libstdc++.so.6.* /usr/lib/x86_64-linux-gnu/ \
 && ln -sf "$(basename "$(ls /opt/ffmpeg/lib/libstdc++.so.6.0.* | tail -1)")" /usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
 && echo "" | gcc -shared -o /usr/local/lib/libpython3.12.so.1.0 -x c - \
 && ldconfig

# 4. awscli — sagemaker/sm_env.py uses it for the optional pretrained-weights sync.
RUN pip install awscli

# ---------------------------------------------------------------------------
# Repo code -> /opt/ml/code (SageMaker's user-code dir).
# ---------------------------------------------------------------------------
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code
COPY . ${SAGEMAKER_SUBMIT_DIRECTORY}
WORKDIR ${SAGEMAKER_SUBMIT_DIRECTORY}

RUN git config --global --add safe.directory ${SAGEMAKER_SUBMIT_DIRECTORY} \
 && pip install -e . --no-deps --no-build-isolation

# SAGEMAKER_PROGRAM is also set on the estimator environment by launch_sm.py;
# setting it here keeps the image self-describing.
ENV SAGEMAKER_PROGRAM=sagemaker/entry.py
ENV ROLLINGWAM_REPO_DIR=/opt/ml/code
ENV PYTHONPATH=/opt/ml/code:/opt/ml/code/src

RUN python -c "import torch, transformers, accelerate, deepspeed, rollingwam; \
from torchcodec.decoders import VideoDecoder; \
print(f'torch {torch.__version__} | transformers {transformers.__version__} | accelerate {accelerate.__version__} | torchcodec OK')"
