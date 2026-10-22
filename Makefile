PYTHON ?= python
IMAGE ?= frame.jpg
PREPROCESS ?= default
IMG = $(abspath $(IMAGE))

.PHONY: all run bench attn hybrid preprocess eos

all: run bench

run:
	$(PYTHON) smolvlm_cuda_graphs.py $(IMG)

bench:
	$(PYTHON) bench.py --image $(IMG) --preprocess $(PREPROCESS)

attn:
	$(PYTHON) attn_check.py --image $(IMG)

hybrid:
	$(PYTHON) hybrid_attn.py --image $(IMG)

preprocess:
	$(PYTHON) gpu_preprocess.py --image $(IMG)

eos:
	$(PYTHON) eos_stop.py --image $(IMG)
