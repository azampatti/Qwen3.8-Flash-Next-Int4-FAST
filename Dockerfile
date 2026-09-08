# Qwen3.8-Flash-Next A5B INT4-AR — serving image
#
# This adds NOTHING of its own to the model stack. It starts from the image built by
#   Saren-Arterius/qwen3.8-Flash-DGX-AutoRound   (the AutoRound int4 + fp8-hybrid vLLM build)
# and applies two OPTIONAL patches from
#   blazux/qwen3.8-Flash-DGX                      (patch 7, by @Nanetnounou)
#   jschmied/qwen38-flash-next-gb10               (patch 8, deterministic top-k kernel)
# Both are inert unless you turn them on at serve time, so this image behaves exactly like the
# upstream one by default. Every source file is fetched from its own repository at a pinned
# commit and checked against a sha256 — nothing is vendored here.
#
# Build with setup.sh (it builds the upstream image first and passes it in as BASE).
ARG BASE=qwen38-flash-dgx:upstream
FROM ${BASE}
ARG SP=/usr/local/lib/python3.12/dist-packages
ARG DET_ARCH=121a

# --- patch 7: fp8_e4m3 KV cache on the QSA path (@Nanetnounou, blazux issue #6) --------------
# Lets --kv-cache-dtype fp8_e4m3 work: ~1.9x KV tokens for the same budget, at roughly -10%
# decode / -30% prefill and a measurable quality cost. INERT with the default --kv-cache-dtype auto.
ARG BLAZUX_SHA=b76890d5a033dd00166c792393d39cf908f56034
ADD --checksum=sha256:2e37380384de5ff80c17be50637a70a08cbf6ee254bf6df50c910067b98f78e7 \
    https://raw.githubusercontent.com/blazux/qwen3.8-Flash-DGX/${BLAZUX_SHA}/src/patch_qsa_fp8_kv.py /tmp/patch_qsa_fp8_kv.py
RUN python3 /tmp/patch_qsa_fp8_kv.py ${SP} && rm /tmp/patch_qsa_fp8_kv.py

# --- patch 8: deterministic persistent_topk kernel (@jschmied, vllm#55122) -------------------
# Identical greedy output run to run, at full prefill speed. INERT unless VLLM_QSA_DET_TOPK=1.
ARG KDET_SHA=e0ef69d4f5575dad00d34e05479eaf4c6547bace
ARG KDET=https://raw.githubusercontent.com/jschmied/qwen38-flash-next-gb10/${KDET_SHA}
ADD --checksum=sha256:138cacfc5eb117f0922d53c88727e4d0dc26dcfb246c3d401fc280cfc726cc71 ${KDET}/patches/kernel-det/build_det.py        /opt/llm/kernel-det/src/build_det.py
ADD --checksum=sha256:b103fbeaf7589b9468471142ad0b30012a076f93d20ba11fc5ff6dcb1ecd32a6 ${KDET}/patches/kernel-det/bindings_det.cpp   /opt/llm/kernel-det/src/bindings_det.cpp
ADD --checksum=sha256:19e1d53425ea9a839445722fd1dac1c41727128eebdce84508c1bfb8592afecf ${KDET}/patches/kernel-det/topk_det.cu       /opt/llm/kernel-det/src/topk_det.cu
ADD --checksum=sha256:16939700ae389750782ff5c0d5b9caef59aa0ff8b869b64ec94fa72c814910ee ${KDET}/patches/kernel-det/torch_utils.h     /opt/llm/kernel-det/src/torch_utils.h
ADD --checksum=sha256:b4ef9ce298d43d6c0e6db9fcca451df20815b2cfe33791919c1ad9c0e84f0ba7 ${KDET}/patches/kernel-det/persistent_topk.cuh /opt/llm/kernel-det/src/persistent_topk.cuh
ADD --checksum=sha256:70905073fe3fa361030bf1cb469b74610766bdfe361419cd7df50af2561322e3 ${KDET}/tools/determinism/qsadet_patch.py    /tmp/qsadet_patch.py
RUN cd /opt/llm/kernel-det/src && DET_BUILD_DIR=/opt/llm/kernel-det/build DET_ARCH=${DET_ARCH} python3 build_det.py 2>&1 | tail -2 \
 && cp /opt/llm/kernel-det/build/_C_det.so /opt/llm/kernel-det/_C_det.so \
 && VLLM_QSA_PY=${SP}/vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py python3 /tmp/qsadet_patch.py && rm /tmp/qsadet_patch.py \
 && python3 -c "import ast; [ast.parse(open(p).read()) for p in ('${SP}/vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py','${SP}/vllm/models/qwen3_8_flash_next/nvidia/qsa.py')]; print('both patches applied OK')"
