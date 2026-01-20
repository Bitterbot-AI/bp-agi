# BP-AGI Makefile - Convenience targets for common operations
# Use CMake for actual build configuration

.PHONY: all build release debug test benchmark clean install help

BUILD_DIR ?= build
CMAKE_FLAGS ?=
NPROC := $(shell nproc 2>/dev/null || echo 4)

all: release

# Release build (optimized)
release:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake -DCMAKE_BUILD_TYPE=Release $(CMAKE_FLAGS) ..
	@cmake --build $(BUILD_DIR) -j$(NPROC)

# Debug build
debug:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake -DCMAKE_BUILD_TYPE=Debug $(CMAKE_FLAGS) ..
	@cmake --build $(BUILD_DIR) -j$(NPROC)

# Build alias
build: release

# Run tests
test: release
	@cd $(BUILD_DIR) && ctest --output-on-failure

# Run tests with verbose output
test-verbose: release
	@cd $(BUILD_DIR) && ./bpagi_tests --gtest_color=yes

# Run benchmark
benchmark: release
	@echo "Running BP-AGI Benchmark..."
	@echo "Target: 1M neurons @ 100Hz (10ms per step)"
	@$(BUILD_DIR)/bpagi_benchmark

# Run demo
demo: release
	@$(BUILD_DIR)/bpagi_demo

# Clean build artifacts
clean:
	@rm -rf $(BUILD_DIR)
	@rm -f python/bpagi/*.so python/bpagi/*.pyd
	@echo "Clean complete."

# Install (requires sudo typically)
install: release
	@cmake --install $(BUILD_DIR)

# Format code (requires clang-format)
format:
	@find include src tests -name '*.hpp' -o -name '*.cpp' | xargs clang-format -i
	@echo "Formatting complete."

# Static analysis (requires clang-tidy)
lint:
	@cd $(BUILD_DIR) && cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
	@find src -name '*.cpp' | xargs clang-tidy -p $(BUILD_DIR)

# Python tests
pytest: release
	@cd python && python -m pytest ../tests/python -v

# Build Python wheel
wheel: release
	@cd python && python setup.py bdist_wheel

# Show help
help:
	@echo "BP-AGI Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all, release  - Build optimized release version (default)"
	@echo "  debug         - Build debug version with symbols"
	@echo "  test          - Run C++ unit tests"
	@echo "  test-verbose  - Run tests with detailed output"
	@echo "  benchmark     - Run performance benchmark"
	@echo "  demo          - Run demo executable"
	@echo "  clean         - Remove build artifacts"
	@echo "  install       - Install library and headers"
	@echo "  format        - Format code with clang-format"
	@echo "  lint          - Run static analysis"
	@echo "  pytest        - Run Python tests"
	@echo "  wheel         - Build Python wheel"
	@echo ""
	@echo "Variables:"
	@echo "  BUILD_DIR     - Build directory (default: build)"
	@echo "  CMAKE_FLAGS   - Extra flags to pass to CMake"
