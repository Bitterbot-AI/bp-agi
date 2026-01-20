cmake_minimum_required(VERSION 3.16)
project(bpagi VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type defaults to Release for performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Optimization flags for performance-critical integer arithmetic
# Blueprint target: 2.5B synapses/sec on consumer CPU
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG -flto")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Wpedantic")

# Enable threading support
find_package(Threads REQUIRED)

# Enable OpenMP for parallel processing (Phase 15: Honeybee scale)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found - enabling parallel processing for honeybee scale")
endif()

# ========================================
# Core Library
# ========================================
add_library(bpagi_core STATIC
    src/synapse.cpp
    src/network.cpp
    src/vision.cpp
    src/uks.cpp
    src/cortical_column.cpp
    src/brain.cpp
    src/motor.cpp
    src/hippocampus.cpp
    src/pong.cpp
)
target_include_directories(bpagi_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(bpagi_core PRIVATE Threads::Threads)
if(OpenMP_CXX_FOUND)
    target_link_libraries(bpagi_core PRIVATE OpenMP::OpenMP_CXX)
endif()

# ========================================
# Demo Executable
# ========================================
add_executable(bpagi_demo src/main.cpp)
target_link_libraries(bpagi_demo PRIVATE bpagi_core)

# ========================================
# Benchmark Executable
# ========================================
add_executable(bpagi_benchmark examples/demo_engine.cpp)
target_link_libraries(bpagi_benchmark PRIVATE bpagi_core)

# ========================================
# Phase 10: Conscious Agent Example
# ========================================
add_executable(conscious_pong examples/conscious_pong.cpp)
target_link_libraries(conscious_pong PRIVATE bpagi_core)

# ========================================
# Phase 13: Real ARC Benchmark
# ========================================
add_executable(run_real_arc examples/run_real_arc.cpp)
target_link_libraries(run_real_arc PRIVATE bpagi_core)

# ========================================
# Phase 14: ARC Verification Tool
# ========================================
add_executable(arc_verify examples/arc_verify.cpp)
target_link_libraries(arc_verify PRIVATE bpagi_core)

# ========================================
# Phase 15: Honeybee Scale (1M neurons)
# ========================================
add_executable(arc_honeybee examples/arc_honeybee.cpp)
target_link_libraries(arc_honeybee PRIVATE bpagi_core)
if(OpenMP_CXX_FOUND)
    target_link_libraries(arc_honeybee PRIVATE OpenMP::OpenMP_CXX)
endif()

# ========================================
# Tests (Google Test)
# ========================================
option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS)
    # Try to find GTest, or fetch it if not available
    find_package(GTest QUIET)

    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.14.0
        )
        # For Windows: Prevent overriding the parent project's compiler/linker settings
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()

    enable_testing()

    # Core engine tests (GTest)
    add_executable(bpagi_tests tests/cpp/test_engine.cpp)
    target_link_libraries(bpagi_tests PRIVATE bpagi_core GTest::gtest_main)

    include(GoogleTest)
    gtest_discover_tests(bpagi_tests)

    # Vision system tests (standalone)
    add_executable(test_vision tests/cpp/test_vision.cpp)
    target_link_libraries(test_vision PRIVATE bpagi_core)

    # UKS tests (standalone)
    add_executable(test_uks tests/cpp/test_uks.cpp)
    target_link_libraries(test_uks PRIVATE bpagi_core)

    # Integration tests (standalone)
    add_executable(test_integration tests/cpp/test_integration.cpp)
    target_link_libraries(test_integration PRIVATE bpagi_core)

    # Phase 5: Shift Test - Translation Invariance (standalone)
    add_executable(test_shift tests/cpp/test_shift.cpp)
    target_link_libraries(test_shift PRIVATE bpagi_core)

    # Phase 12: ARC Reasoning Benchmark (standalone)
    add_executable(test_arc_reasoning tests/cpp/test_arc_reasoning.cpp)
    target_link_libraries(test_arc_reasoning PRIVATE bpagi_core)
endif()

# ========================================
# Python Bindings (pybind11)
# ========================================
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)

if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 CONFIG QUIET)

    if(pybind11_FOUND)
        pybind11_add_module(bpagi_py python/bindings.cpp)
        target_link_libraries(bpagi_py PRIVATE bpagi_core)

        # Install Python module to python/bpagi directory
        set_target_properties(bpagi_py PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/bpagi
        )
    else()
        message(STATUS "pybind11 not found. Python bindings will not be built.")
        message(STATUS "Install with: pip install pybind11")
    endif()
endif()

# ========================================
# Installation
# ========================================
install(TARGETS bpagi_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
install(DIRECTORY include/bpagi DESTINATION include)
