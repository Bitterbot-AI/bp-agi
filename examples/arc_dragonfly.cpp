/**
 * Phase 16: PROJECT DRAGONFLY
 *
 * The "High-Fidelity" Benchmark with 2-Attempt Protocol
 *
 * Scaling to 5M neurons, 50K columns with:
 *   - Attempt 1: Standard inference
 *   - Noise injection if failed (Norepinephrine shake)
 *   - Attempt 2: Second guess after noise perturbation
 *
 * Biologically plausible "second guessing" - when stuck,
 * inject noise to escape local minima.
 */

#include "bpagi/brain.hpp"
#include "bpagi/arc_loader.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace bpagi;

// ========================================
// DRAGONFLY CONFIGURATION
// ========================================

struct DragonflyConfig {
    // Scale: 5x Honeybee
    static constexpr size_t NUM_COLUMNS = 50000;      // 50K columns
    static constexpr size_t BUS_WIDTH = 128;          // 128-bit bus

    // Timing
    static constexpr int PRESENT_TICKS = 10;          // More settling time
    static constexpr int DELAY_TICKS = 5;
    static constexpr int SETTLE_TICKS = 8;
    static constexpr int TEST_TICKS = 20;             // Prediction time

    // Learning
    static constexpr int DOPAMINE_BOOST = 90;
    static constexpr int REWARD_AMOUNT = 100;

    // 2-Attempt Protocol
    static constexpr int NOISE_AMPLITUDE = 15;        // NE injection strength
    static constexpr int NOISE_SETTLE_TICKS = 5;      // Time after noise
    static constexpr float PASS_THRESHOLD = 0.99f;    // 99% match = pass

    // Output
    static constexpr bool VERBOSE = true;
};

// ========================================
// Dragonfly Brain Wrapper
// ========================================

class DragonflyBrain {
public:
    DragonflyBrain() {
        Brain::Config config;
        config.numColumns = DragonflyConfig::NUM_COLUMNS;
        config.busWidth = DragonflyConfig::BUS_WIDTH;
        config.enableLearning = true;

        brain_ = std::make_unique<Brain>(config);

        // Enable operant conditioning
        brain_->getNetwork().setPlasticityEnabled(true);
        brain_->getNetwork().setOperantMode(true);

        std::cout << "========================================" << std::endl;
        std::cout << "   PROJECT DRAGONFLY - ONLINE" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "  Neurons:  " << brain_->getNetwork().getNeuronCount() << std::endl;
        std::cout << "  Synapses: " << brain_->getNetwork().getSynapseCount() << std::endl;
        std::cout << "  Columns:  " << DragonflyConfig::NUM_COLUMNS << std::endl;
        std::cout << "  Bus:      " << DragonflyConfig::BUS_WIDTH << " bits" << std::endl;

        #ifdef _OPENMP
        std::cout << "  Threads:  " << omp_get_max_threads() << std::endl;
        #else
        std::cout << "  Threads:  1 (OpenMP disabled)" << std::endl;
        #endif

        std::cout << "========================================" << std::endl;
        std::cout << std::endl;
    }

    void reset() {
        brain_->reset();
        brain_->getNetwork().setPlasticityEnabled(true);
    }

    void resetShortTerm() {
        brain_->resetShortTermMemory();
    }

    void trainOnExample(const ArcPair& example) {
        // Boost dopamine for learning
        brain_->getNetwork().chemicals().dopamine = DragonflyConfig::DOPAMINE_BOOST;

        // Present input
        brain_->present(example.input);
        for (int t = 0; t < DragonflyConfig::PRESENT_TICKS; t++) {
            brain_->step();
        }

        // Delay (working memory)
        for (int t = 0; t < DragonflyConfig::DELAY_TICKS; t++) {
            brain_->step();
        }

        // Present output (target)
        brain_->present(example.output);
        for (int t = 0; t < DragonflyConfig::PRESENT_TICKS; t++) {
            brain_->step();
        }

        // Reward signal
        brain_->getNetwork().rewardSignal(static_cast<int8_t>(std::min(DragonflyConfig::REWARD_AMOUNT, 100)));
        brain_->getNetwork().injectReward(DragonflyConfig::REWARD_AMOUNT);

        // Settle/consolidate
        for (int t = 0; t < DragonflyConfig::SETTLE_TICKS; t++) {
            brain_->step();
        }
    }

    float predict(const std::vector<uint8_t>& input, const std::vector<uint8_t>& expected) {
        // Present input
        brain_->present(input);
        for (int t = 0; t < DragonflyConfig::PRESENT_TICKS; t++) {
            brain_->step();
        }

        // Wait for prediction to form
        for (int t = 0; t < DragonflyConfig::TEST_TICKS; t++) {
            brain_->step();
        }

        // Compare retina to expected
        return computeSimilarity(expected);
    }

    void injectNoise() {
        // Inject norepinephrine (exploration noise)
        brain_->injectNoise(DragonflyConfig::NOISE_AMPLITUDE);

        // Also boost NE in chemical system
        brain_->getNetwork().chemicals().norepinephrine = 80;

        // Let noise propagate
        for (int t = 0; t < DragonflyConfig::NOISE_SETTLE_TICKS; t++) {
            brain_->step();
        }
    }

    void disableLearning() {
        brain_->getNetwork().setPlasticityEnabled(false);
    }

    void enableLearning() {
        brain_->getNetwork().setPlasticityEnabled(true);
    }

private:
    std::unique_ptr<Brain> brain_;

    float computeSimilarity(const std::vector<uint8_t>& expected) {
        auto& vision = brain_->getVision();
        int matches = 0;
        int total = 0;

        for (size_t y = 0; y < 64; y++) {
            for (size_t x = 0; x < 64; x++) {
                size_t idx = y * 64 + x;
                bool expectedActive = (expected[idx] > 14);  // Non-black
                bool actualActive = vision.isRetinaActive(x, y);

                total++;
                if (expectedActive == actualActive) {
                    matches++;
                }
            }
        }

        return static_cast<float>(matches) / total;
    }
};

// ========================================
// Result Tracking
// ========================================

struct TaskResult {
    std::string taskId;
    size_t numTrain;
    float attempt1Score;
    float attempt2Score;
    bool savedByRetry;
    bool passed;
    double timeMs;
};

// ========================================
// Main: 2-Attempt Protocol
// ========================================

int main(int argc, char* argv[]) {
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  DRAGONFLY ARC-AGI-2 BENCHMARK" << std::endl;
    std::cout << "  2-Attempt Protocol with Noise Injection" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // Load data
    std::string dataFile = "arc_eval.bin";
    if (argc > 1) {
        dataFile = argv[1];
    }

    std::cout << "Loading: " << dataFile << std::endl;
    auto tasks = ArcLoader::load(dataFile);

    if (tasks.empty()) {
        std::cerr << "ERROR: No tasks loaded from " << dataFile << std::endl;
        return 1;
    }

    std::cout << "Loaded " << tasks.size() << " tasks" << std::endl;
    std::cout << std::endl;

    // Initialize Dragonfly brain
    DragonflyBrain brain;

    // Run benchmark
    std::vector<TaskResult> results;
    results.reserve(tasks.size());

    int totalPassed = 0;
    int savedByRetry = 0;

    auto benchStart = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < tasks.size(); i++) {
        const auto& task = tasks[i];
        auto taskStart = std::chrono::high_resolution_clock::now();

        // Reset for fresh learning
        brain.reset();

        // === TRAINING PHASE ===
        for (const auto& example : task.trainExamples) {
            brain.trainOnExample(example);
        }

        // === TEST PHASE (no learning) ===
        brain.disableLearning();

        TaskResult result;
        result.taskId = task.id;
        result.numTrain = task.trainExamples.size();
        result.attempt1Score = 0;
        result.attempt2Score = 0;
        result.savedByRetry = false;
        result.passed = false;

        // Process first test case (benchmark metric)
        if (!task.testExamples.empty()) {
            const auto& test = task.testExamples[0];

            // === ATTEMPT 1: Standard inference ===
            result.attempt1Score = brain.predict(test.input, test.output);

            if (result.attempt1Score >= DragonflyConfig::PASS_THRESHOLD) {
                // Passed on first try!
                result.passed = true;
                result.attempt2Score = result.attempt1Score;
            } else {
                // === INTERVENTION: Noise injection ===
                brain.injectNoise();

                // === ATTEMPT 2: Second guess ===
                result.attempt2Score = brain.predict(test.input, test.output);

                if (result.attempt2Score >= DragonflyConfig::PASS_THRESHOLD) {
                    result.passed = true;
                    result.savedByRetry = true;
                    savedByRetry++;
                }
            }
        }

        auto taskEnd = std::chrono::high_resolution_clock::now();
        result.timeMs = std::chrono::duration<double, std::milli>(taskEnd - taskStart).count();

        if (result.passed) totalPassed++;
        results.push_back(result);

        // === LOGGING ===
        if (DragonflyConfig::VERBOSE) {
            std::cout << "[" << std::setw(3) << (i + 1) << "/" << tasks.size() << "] "
                      << task.id << " (" << task.trainExamples.size() << " train) ";

            if (result.passed && !result.savedByRetry) {
                // First try success
                std::cout << std::fixed << std::setprecision(1)
                          << (result.attempt1Score * 100) << "% [PASS]"
                          << " (" << std::setprecision(0) << result.timeMs << "ms)"
                          << std::endl;
            } else if (result.savedByRetry) {
                // Saved by retry
                std::cout << "Try1: " << std::fixed << std::setprecision(0)
                          << (result.attempt1Score * 100) << "% -> "
                          << "Try2: " << (result.attempt2Score * 100) << "% "
                          << "[PASS - SAVED!]"
                          << " (" << result.timeMs << "ms)"
                          << std::endl;
            } else {
                // Failed both
                std::cout << "Try1: " << std::fixed << std::setprecision(0)
                          << (result.attempt1Score * 100) << "% -> "
                          << "Try2: " << (result.attempt2Score * 100) << "%"
                          << " (" << result.timeMs << "ms)"
                          << std::endl;
            }
        }
    }

    auto benchEnd = std::chrono::high_resolution_clock::now();
    double totalMs = std::chrono::duration<double, std::milli>(benchEnd - benchStart).count();

    // ========================================
    // Final Summary
    // ========================================

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "   DRAGONFLY BENCHMARK RESULTS" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    float avgAttempt1 = std::accumulate(results.begin(), results.end(), 0.0f,
        [](float sum, const TaskResult& r) { return sum + r.attempt1Score; }) / results.size();
    float avgAttempt2 = std::accumulate(results.begin(), results.end(), 0.0f,
        [](float sum, const TaskResult& r) { return sum + r.attempt2Score; }) / results.size();

    std::cout << "Tasks:              " << results.size() << std::endl;
    std::cout << "Total Passed:       " << totalPassed << " ("
              << std::fixed << std::setprecision(1)
              << (100.0f * totalPassed / results.size()) << "%)" << std::endl;
    std::cout << "Saved by Retry:     " << savedByRetry << std::endl;
    std::cout << "First-Try Passes:   " << (totalPassed - savedByRetry) << std::endl;
    std::cout << std::endl;
    std::cout << "Avg Attempt 1:      " << std::setprecision(1) << (avgAttempt1 * 100) << "%" << std::endl;
    std::cout << "Avg Attempt 2:      " << (avgAttempt2 * 100) << "%" << std::endl;
    std::cout << std::endl;
    std::cout << "Total Time:         " << std::setprecision(0) << totalMs << " ms" << std::endl;
    std::cout << "Avg Time/Task:      " << std::setprecision(1) << (totalMs / results.size()) << " ms" << std::endl;
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "   FINAL SCORE: " << totalPassed << "/" << results.size()
              << " (" << std::setprecision(1) << (100.0f * totalPassed / results.size()) << "%)"
              << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
