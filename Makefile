PYTHON ?= python
IMAGE ?= frame.jpg
IMG = $(abspath $(IMAGE))

.PHONY: all run bench

all: run bench

run:
	cd scripts && $(PYTHON) smolvlm_cuda_graphs.py $(IMG)

bench:
	cd scripts && $(PYTHON) bench.py --image $(IMG)
