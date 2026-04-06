# Makefile for Zipformer Encoder Engine & Tests (RK3576)
#
# Expects RKNN headers at /home/cat/ or set RKNN_INCLUDE/RKNN_LIB.
# Cross-compile on x86: set CC=aarch64-linux-gnu-gcc

CC ?= gcc
CFLAGS = -O2 -Wall -Wno-unused-result -march=armv8-a+fp16
LDFLAGS = -lrknnrt -lm -lpthread

# RKNN SDK paths (adjust for your system)
RKNN_INCLUDE ?= /home/cat
RKNN_LIB ?= /usr/lib

CFLAGS += -I$(RKNN_INCLUDE)
LDFLAGS += -L$(RKNN_LIB)

.PHONY: all clean test lib fp16 fp16-lib fp16-test

# Default: build everything (INT8 mode)
all: zipformer_layer_test zipformer_encoder_test libzipformer_encoder.so

# Layer precision test (existing)
zipformer_layer_test: zipformer_layer_test.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Full encoder shared library (INT8)
libzipformer_encoder.so: zipformer_encoder.c zipformer_encoder.h
	$(CC) $(CFLAGS) -fPIC -shared -o $@ zipformer_encoder.c $(LDFLAGS)
	@echo "Built $@ (INT8 mode)"

lib: libzipformer_encoder.so

# Full encoder test binary (INT8)
zipformer_encoder_test: zipformer_encoder.c zipformer_encoder.h
	$(CC) $(CFLAGS) -DZIPFORMER_TEST_MAIN -o $@ zipformer_encoder.c $(LDFLAGS)
	@echo "Built $@ (INT8 mode)"

test: zipformer_encoder_test
	./zipformer_encoder_test

# ─── FP16 mode targets ───
# Full encoder shared library (FP16)
libzipformer_encoder_fp16.so: zipformer_encoder.c zipformer_encoder.h
	$(CC) $(CFLAGS) -DMTE_USE_FP16 -fPIC -shared -o $@ zipformer_encoder.c $(LDFLAGS)
	@echo "Built $@ (FP16 mode)"

fp16-lib: libzipformer_encoder_fp16.so

# Full encoder test binary (FP16)
zipformer_encoder_test_fp16: zipformer_encoder.c zipformer_encoder.h
	$(CC) $(CFLAGS) -DMTE_USE_FP16 -DZIPFORMER_TEST_MAIN -o $@ zipformer_encoder.c $(LDFLAGS)
	@echo "Built $@ (FP16 mode)"

fp16: zipformer_encoder_test_fp16

fp16-test: zipformer_encoder_test_fp16
	./zipformer_encoder_test_fp16

# ─── FP16 + Debug dump mode targets ───
libzipformer_encoder_fp16_debug.so: zipformer_encoder.c zipformer_encoder.h
	$(CC) $(CFLAGS) -DMTE_USE_FP16 -DMTE_DEBUG_DUMP -fPIC -shared -o $@ zipformer_encoder.c $(LDFLAGS)
	@echo "Built $@ (FP16 + debug dump mode)"

fp16-debug-lib: libzipformer_encoder_fp16_debug.so

# INT8 + Debug dump
libzipformer_encoder_debug.so: zipformer_encoder.c zipformer_encoder.h
	$(CC) $(CFLAGS) -DMTE_DEBUG_DUMP -fPIC -shared -o $@ zipformer_encoder.c $(LDFLAGS)
	@echo "Built $@ (INT8 + debug dump mode)"

debug-lib: libzipformer_encoder_debug.so

clean:
	rm -f zipformer_layer_test zipformer_encoder_test zipformer_encoder_test_fp16 \
	      libzipformer_encoder.so libzipformer_encoder_fp16.so \
	      libzipformer_encoder_fp16_debug.so libzipformer_encoder_debug.so
