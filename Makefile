PYTHON ?= python
IMAGE ?= frame.jpg
IMG = $(abspath $(IMAGE))

.PHONY: all run bench attn hybrid preprocess

all: run bench

run:
	$(PYTHON) smolvlm_cuda_graphs.py $(IMG)

bench:
	$(PYTHON) bench.py --image $(IMG)

attn:
	$(PYTHON) attn_check.py --image $(IMG)

hybrid:
	$(PYTHON) hybrid_attn.py --image $(IMG)

preprocess:
	$(PYTHON) gpu_preprocess.py --image $(IMG)
