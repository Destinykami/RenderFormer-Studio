# RenderFormer MUSA dependency-only environment image.
#
# Follows the same code/image separation as the CUDA Dockerfile: application
# source is bind-mounted into /workspace/renderformer at container runtime.
ARG BASE_IMAGE=sh-harbor.mthreads.com/mcctest/training-suite:v2.1.7-rc4
FROM ${BASE_IMAGE}

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/renderformer/src \
    PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True

COPY docker/requirements-musa.txt /tmp/requirements-musa.txt

# 1. Install RenderFormer dependencies without touching preinstalled TorchMUSA stack.
RUN python -m pip install --no-deps --no-cache-dir -r /tmp/requirements-musa.txt \
    && rm -f /tmp/requirements-musa.txt

# 2. Install FA3 / MATE from the official MUSA PyPI index (single index per step).
RUN python -m pip uninstall -y mate mate-flash-attention flash_attn_3 || true \
    && pip install flash_attn_3 --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple \
    && pip install mate --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple \
    && pip install mate-mubin --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple

WORKDIR /workspace/renderformer

CMD ["/bin/bash"]
