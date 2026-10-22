PYTHON ?= python
IMAGE ?= frame.jpg
IMG = $(abspath $(IMAGE))

.PHONY: all run bench

all: run bench

run:
	$(PYTHON) smolvlm_cuda_graphs.py $(IMG)

bench:
	$(PYTHON) bench.py --image $(IMG)
