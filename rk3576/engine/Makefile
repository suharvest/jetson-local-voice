CC      ?= gcc
CFLAGS  := -O2 -fPIC -Wall -Wno-unused-result -march=armv8-a+fp16
LDFLAGS := -shared -lrknnrt -lm

# RKNN headers location
RKNN_INC ?= /home/cat

TARGET  := libcp_engine.so
SRC     := cp_engine.c

.PHONY: all clean test w4a16 w8a16

# Default: FP16xFP16 mode (~232ms, bit-exact)
all: $(TARGET)

$(TARGET): $(SRC) cp_engine.h
	$(CC) $(CFLAGS) -I$(RKNN_INC) -o $@ $< $(LDFLAGS)
	@echo "Built $@ (FP16 mode)"

# W4A16 mode: ~73ms, INT4 per-column quantized weights, ~3x speedup
w4a16: $(SRC) cp_engine.h
	$(CC) $(CFLAGS) -DCP_USE_W4A16 -I$(RKNN_INC) -o $(TARGET) $< $(LDFLAGS)
	@echo "Built $(TARGET) (W4A16 mode)"

# W8A16 mode: ~120ms, INT8 per-column quantized weights, ~2x speedup, better accuracy
w8a16: $(SRC) cp_engine.h
	$(CC) $(CFLAGS) -DCP_USE_W8A16 -I$(RKNN_INC) -o $(TARGET) $< $(LDFLAGS)
	@echo "Built $(TARGET) (W8A16 mode)"

clean:
	rm -f $(TARGET)

test: $(TARGET)
	@echo "Running test..."
	python3 test_engine.py --weight-dir /home/cat/cp_weights --cores 2 --reps 10
