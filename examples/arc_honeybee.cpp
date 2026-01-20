/**
 * Phase 15: The "Honey Bee" Upgrade
 *
 * Scaling BP-AGI to biologically meaningful capacity:
 * - 1,000,000 neurons (honeybee: ~960,000)
 * - 100,000,000 synapses
 * - 10,000 UKS columns
 * - 128-bit bus width
 *
 * Hypothesis: "Plausible Errors" from the 100k brain will convert to
 * "Passes" when working memory capacity increases.
 *
 * Target: Break 25% on ARC evaluation set.
 */

#include "bpagi/brain.hpp"
#include "bpagi/arc_loader.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <memory>

// OpenMP for parallel synapse processing
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace bpagi;

// ========================================
// Honeybee Configuration
// ========================================

struct HoneybeeConfig {
    // Scale parameters
    static constexpr size_t NUM_COLUMNS = 10000;      // 100x increase
    static constexpr size_t BUS_WIDTH = 128;          // 2x increase

    // Timing (adjusted for larger network settling time)
    static constexpr int PRESENT_TICKS = 8;           // More time to propagate
    static constexpr int DELAY_TICKS = 5;
    static constexpr int SETTLE_TICKS = 5;
    static constexpr int TEST_WAIT_TICKS = 15;        // More time for prediction

    // Learning
    static constexpr int DOPAMINE_BOOST = 80;
    static constexpr int REWARD_AMOUNT = 100;

    // Benchmark
    static constexpr bool VERBOSE = true;
};

// ========================================
// Honeybee Brain Factory
// ========================================

class HoneybeeBrain {
public:
    HoneybeeBrain() {
        // Configure for honeybee scale
        Brain::Config config;
        config.numColumns = HoneybeeConfig::NUM_COLUMNS;
        config.busWidth = HoneybeeConfig::BUS_WIDTH;
        config.enableLearning = true;

        brain_ = std::make_unique<Brain>(config);

        // Enable operant conditioning
        brain_->getNetwork().setPlasticityEnabled(true);
        brain_->getNetwork().setOperantMode(true);

        // Report scale
        std::cout << "Honeybee Brain initialized:" << std::endl;
        std::cout << "  Neurons:  " << brain_->getNetwork().getNeuronCount() << std::endl;
        std::cout << "  Synapses: " << brain_->getNetwork().getSynapseCount() << std::endl;
        std::cout << "  Columns:  " << HoneybeeConfig::NUM_COLUMNS << std::endl;
        std::cout << "  Bus:      " << HoneybeeConfig::BUS_WIDTH << " bits" << std::endl;

        #ifdef _OPENMP
        std::cout << "  OpenMP:   " << omp_get_max_threads() << " threads" << std::endl;
        #else
        std::cout << "  OpenMP:   disabled" << std::endl;
        #endif
    }

    void reset() {
        brain_->reset();
        brain_->getNetwork().setPlasticityEnabled(true);
    }

    void trainOnExample(const ArcPair& example) {
        brain_->getNetwork().chemicals().dopamine = HoneybeeConfig::DOPAMINE_BOOST;

        // Present input
        brain_->present(example.input);
        for (int t = 0; t < HoneybeeConfig::PRESENT_TICKS; t++) {
            brain_->step();
        }

        // Delay
        for (int t = 0; t < HoneybeeConfig::DELAY_TICKS; t++) {
            brain_->step();
        }

        // Present output
        brain_->present(example.output);
        for (int t = 0; t < HoneybeeConfig::PRESENT_TICKS; t++) {
            brain_->step();
        }

        // Reward
        brain_->getNetwork().rewardSignal(static_cast<int8_t>(std::min(HoneybeeConfig::REWARD_AMOUNT, 100)));
        brain_->getNetwork().injectReward(HoneybeeConfig::REWARD_AMOUNT);

        // Settle
        for (int t = 0; t < HoneybeeConfig::SETTLE_TICKS; t++) {
            brain_->step();
        }
    }

    float testPrediction(const std::vector<uint8_t>& input,
                        const std::vector<uint8_t>& expected) {
        // Present input only
        brain_->present(input);
        for (int t = 0; t < HoneybeeConfig::PRESENT_TICKS; t++) {
            brain_->step();
        }

        // Wait for prediction
        for (int t = 0; t < HoneybeeConfig::TEST_WAIT_TICKS; t++) {
            brain_->step();
        }

        // Compare retina activity to expected
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

    void disableLearning() {
        brain_->getNetwork().setPlasticityEnabled(false);
    }

private:
    std::unique_ptr<Brain> brain_;
};

// ========================================
// Result Tracking
// ========================================

struct TaskResult {
    std::string taskId;
    size_t numTrain;
    size_t numTest;
    float similarity;
    bool passed;
    double timeMs;
    std::string category;  // "correct", "input_copy", "plausible_error", "random_noise"
};

std::string classifyResult(float similarity, float inputOutputSim) {
    if (similarity >= 0.99f) return "correct";
    if (std::abs(similarity - inputOutputSim) < 0.05f) return "input_copy";
    if (similarity < 0.20f) return "random_noise";
    return "plausible_error";
}

// ========================================
// Main Benchmark
// ========================================

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "Phase 15: Honeybee ARC Benchmark" << std::endl;
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
        std::cerr << "No tasks loaded." << std::endl;
        return 1;
    }

    std::cout << "Loaded " << tasks.size() << " tasks" << std::endl;
    std::cout << std::endl;

    // Initialize honeybee brain
    std::cout << "Initializing Honeybee Brain..." << std::endl;
    HoneybeeBrain brain;
    std::cout << std::endl;

    // Run benchmark
    std::vector<TaskResult> results;
    results.reserve(tasks.size());

    auto benchStart = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < tasks.size(); i++) {
        const auto& task = tasks[i];

        if (HoneybeeConfig::VERBOSE) {
            std::cout << "[" << std::setw(3) << (i + 1) << "/" << tasks.size() << "] "
                      << task.id << " (" << task.trainExamples.size() << " train)... "
                      << std::flush;
        }

        auto taskStart = std::chrono::high_resolution_clock::now();

        // Reset for fresh learning
        brain.reset();

        // Training phase
        for (const auto& example : task.trainExamples) {
            brain.trainOnExample(example);
        }

        // Test phase (no learning)
        brain.disableLearning();

        float totalSim = 0.0f;
        int passedTests = 0;

        for (const auto& test : task.testExamples) {
            // Calculate input-output baseline
            int inputOutputMatches = 0;
            for (size_t j = 0; j < test.input.size() && j < test.output.size(); j++) {
                if (std::abs(static_cast<int>(test.input[j]) - static_cast<int>(test.output[j])) <= 14) {
                    inputOutputMatches++;
                }
            }
            float inputOutputSim = static_cast<float>(inputOutputMatches) / test.input.size();

            // Get prediction
            float sim = brain.testPrediction(test.input, test.output);
            totalSim += sim;

            if (sim >= 0.99f) {
                passedTests++;
            }
        }

        auto taskEnd = std::chrono::high_resolution_clock::now();
        double taskMs = std::chrono::duration<double, std::milli>(taskEnd - taskStart).count();

        float avgSim = task.testExamples.empty() ? 0.0f : totalSim / task.testExamples.size();
        bool passed = (passedTests == static_cast<int>(task.testExamples.size()) && !task.testExamples.empty());

        TaskResult result;
        result.taskId = task.id;
        result.numTrain = task.trainExamples.size();
        result.numTest = task.testExamples.size();
        result.similarity = avgSim;
        result.passed = passed;
        result.timeMs = taskMs;
        result.category = classifyResult(avgSim, 0.5f);  // Approximate

        results.push_back(result);

        if (HoneybeeConfig::VERBOSE) {
            std::cout << std::fixed << std::setprecision(1)
                      << (avgSim * 100) << "% "
                      << (passed ? "[PASS]" : "")
                      << " (" << std::setprecision(0) << taskMs << "ms)"
                      << std::endl;
        }
    }

    auto benchEnd = std::chrono::high_resolution_clock::now();
    double totalMs = std::chrono::duration<double, std::milli>(benchEnd - benchStart).count();

    // ========================================
    // Summary
    // ========================================

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "HONEYBEE BENCHMARK RESULTS" << std::endl;
    std::cout << "========================================" << std::endl;

    int passedCount = std::count_if(results.begin(), results.end(),
                                   [](const TaskResult& r) { return r.passed; });

    float avgSim = std::accumulate(results.begin(), results.end(), 0.0f,
        [](float sum, const TaskResult& r) { return sum + r.similarity; }) / results.size();

    // Category breakdown
    int correctCount = std::count_if(results.begin(), results.end(),
        [](const TaskResult& r) { return r.category == "correct"; });
    int inputCopyCount = std::count_if(results.begin(), results.end(),
        [](const TaskResult& r) { return r.category == "input_copy"; });
    int plausibleCount = std::count_if(results.begin(), results.end(),
        [](const TaskResult& r) { return r.category == "plausible_error"; });
    int randomCount = std::count_if(results.begin(), results.end(),
        [](const TaskResult& r) { return r.category == "random_noise"; });

    std::cout << "Tasks:          " << results.size() << std::endl;
    std::cout << "Passed:         " << passedCount << " ("
              << std::fixed << std::setprecision(1)
              << (100.0f * passedCount / results.size()) << "%)" << std::endl;
    std::cout << "Avg Similarity: " << (avgSim * 100) << "%" << std::endl;
    std::cout << std::endl;
    std::cout << "Category Breakdown:" << std::endl;
    std::cout << "  Correct:        " << correctCount << std::endl;
    std::cout << "  Input Copy:     " << inputCopyCount << std::endl;
    std::cout << "  Plausible Err:  " << plausibleCount << std::endl;
    std::cout << "  Random Noise:   " << randomCount << std::endl;
    std::cout << std::endl;
    std::cout << "Total Time:     " << std::setprecision(0) << totalMs << " ms" << std::endl;
    std::cout << "Avg Time/Task:  " << std::setprecision(1)
              << (totalMs / results.size()) << " ms" << std::endl;

    // ========================================
    // Save Results for Comparison
    // ========================================

    std::ofstream scalingFile("scaling_results.txt");
    scalingFile << "========================================" << std::endl;
    scalingFile << "BP-AGI SCALING RESULTS" << std::endl;
    scalingFile << "========================================" << std::endl;
    scalingFile << std::endl;
    scalingFile << "BASELINE (100k neurons):" << std::endl;
    scalingFile << "  Score: 16.7% (20/120 tasks)" << std::endl;
    scalingFile << "  Input Copy: 45" << std::endl;
    scalingFile << "  Plausible Errors: 44" << std::endl;
    scalingFile << "  Random Noise: 11" << std::endl;
    scalingFile << std::endl;
    scalingFile << "HONEYBEE (1M neurons, 10K columns):" << std::endl;
    scalingFile << "  Score: " << std::fixed << std::setprecision(1)
                << (100.0f * passedCount / results.size()) << "% ("
                << passedCount << "/" << results.size() << " tasks)" << std::endl;
    scalingFile << "  Correct: " << correctCount << std::endl;
    scalingFile << "  Input Copy: " << inputCopyCount << std::endl;
    scalingFile << "  Plausible Errors: " << plausibleCount << std::endl;
    scalingFile << "  Random Noise: " << randomCount << std::endl;
    scalingFile << std::endl;
    scalingFile << "IMPROVEMENT:" << std::endl;
    float improvement = (100.0f * passedCount / results.size()) - 16.7f;
    scalingFile << "  Delta: " << (improvement >= 0 ? "+" : "") << improvement << "%" << std::endl;
    scalingFile << "  Plausible->Correct conversions: "
                << (correctCount - 20) << " (expected: ~22 of 44)" << std::endl;
    scalingFile << std::endl;
    scalingFile << "TARGET: 25% (need " << (30 - passedCount) << " more passes)" << std::endl;
    scalingFile.close();

    std::cout << std::endl;
    std::cout << "Results saved to: scaling_results.txt" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
