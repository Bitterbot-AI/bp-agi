/**
 * Phase 13: Real ARC Benchmark Runner
 *
 * This program loads the actual ARC-AGI benchmark data and attempts to solve tasks
 * using the BP-AGI Brain's temporal association learning.
 *
 * Approach:
 * 1. For each task, start with a fresh Brain (Tabula Rasa)
 * 2. Training Phase: Show input -> wait -> show output -> inject dopamine
 *    This teaches the brain the causal relationship between patterns
 * 3. Testing Phase: Show test input only -> wait -> measure brain's prediction
 * 4. Score based on how well the internal state matches expected output
 */

#include "bpagi/brain.hpp"
#include "bpagi/arc_loader.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <numeric>

using namespace bpagi;

// ========================================
// Configuration
// ========================================

struct ArcBenchConfig {
    // Timing
    static constexpr int PRESENT_TICKS = 5;      // Ticks to present each image
    static constexpr int DELAY_TICKS = 3;        // Ticks between input and output
    static constexpr int SETTLE_TICKS = 3;       // Ticks after output for learning

    // Learning
    static constexpr int DOPAMINE_BOOST = 80;    // DA level during learning
    static constexpr int REWARD_AMOUNT = 100;    // Reward signal strength

    // Testing
    static constexpr int TEST_WAIT_TICKS = 10;   // Ticks to wait for prediction

    // Benchmark limits
    static constexpr int MAX_TASKS = 100;        // Max tasks to evaluate (0 = all)
    static constexpr bool VERBOSE = true;        // Print per-task results
};

// ========================================
// Result Tracking
// ========================================

struct TaskResult {
    std::string taskId;
    size_t numTrainExamples;
    size_t numTestExamples;
    float trainSimilarity;      // Avg similarity during training
    float testSimilarity;       // Similarity on test
    bool solved;                // Did we get > 95% on test?
    double timeMs;              // Time to process task
};

// ========================================
// ARC Solver Class
// ========================================

class ArcSolver {
public:
    ArcSolver() : brain_{Brain::Config{}} {
        // Configure brain for learning
        brain_.getNetwork().setPlasticityEnabled(true);
        brain_.getNetwork().setOperantMode(true);
    }

    /**
     * Attempt to solve a single ARC task.
     */
    TaskResult solveTask(const ArcTask& task) {
        TaskResult result;
        result.taskId = task.id;
        result.numTrainExamples = task.trainExamples.size();
        result.numTestExamples = task.testExamples.size();

        auto startTime = std::chrono::high_resolution_clock::now();

        // Reset brain for fresh learning
        brain_.reset();
        brain_.getNetwork().setPlasticityEnabled(true);

        // ========================================
        // Training Phase
        // ========================================
        float totalTrainSim = 0.0f;

        for (const auto& example : task.trainExamples) {
            // Learn this input -> output mapping
            trainOnExample(example);

            // Measure how well we learned this example
            float sim = testPrediction(example.input, example.output);
            totalTrainSim += sim;
        }

        result.trainSimilarity = task.trainExamples.empty() ? 0.0f :
                                 totalTrainSim / task.trainExamples.size();

        // ========================================
        // Testing Phase
        // ========================================
        brain_.getNetwork().setPlasticityEnabled(false);  // No learning during test

        float totalTestSim = 0.0f;
        int solvedCount = 0;

        for (const auto& example : task.testExamples) {
            float sim = testPrediction(example.input, example.output);
            totalTestSim += sim;

            if (sim >= 0.95f) {
                solvedCount++;
            }
        }

        result.testSimilarity = task.testExamples.empty() ? 0.0f :
                                totalTestSim / task.testExamples.size();
        result.solved = (solvedCount == static_cast<int>(task.testExamples.size()) &&
                        !task.testExamples.empty());

        auto endTime = std::chrono::high_resolution_clock::now();
        result.timeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

        return result;
    }

private:
    Brain brain_;

    /**
     * Train the brain on a single input -> output example.
     */
    void trainOnExample(const ArcPair& example) {
        // Ensure dopamine is high for learning
        brain_.getNetwork().chemicals().dopamine = ArcBenchConfig::DOPAMINE_BOOST;

        // Present input
        brain_.present(example.input);
        for (int t = 0; t < ArcBenchConfig::PRESENT_TICKS; t++) {
            brain_.step();
        }

        // Brief delay
        for (int t = 0; t < ArcBenchConfig::DELAY_TICKS; t++) {
            brain_.step();
        }

        // Present output (this creates the temporal association)
        brain_.present(example.output);
        for (int t = 0; t < ArcBenchConfig::PRESENT_TICKS; t++) {
            brain_.step();
        }

        // Inject reward to cement the association
        brain_.getNetwork().rewardSignal(static_cast<int8_t>(std::min(ArcBenchConfig::REWARD_AMOUNT, 100)));
        brain_.getNetwork().injectReward(ArcBenchConfig::REWARD_AMOUNT);

        // Let learning settle
        for (int t = 0; t < ArcBenchConfig::SETTLE_TICKS; t++) {
            brain_.step();
        }
    }

    /**
     * Test if the brain predicts the expected output from the input.
     */
    float testPrediction(const std::vector<uint8_t>& input,
                        const std::vector<uint8_t>& expectedOutput) {
        // Present input
        brain_.present(input);
        for (int t = 0; t < ArcBenchConfig::PRESENT_TICKS; t++) {
            brain_.step();
        }

        // Wait for prediction to form
        for (int t = 0; t < ArcBenchConfig::TEST_WAIT_TICKS; t++) {
            brain_.step();
        }

        // Capture the brain's internal state
        // We'll compare retina-level activity to expected output
        auto& vision = brain_.getVision();
        std::vector<uint8_t> predicted(ARC_RETINA_SIZE, 0);

        // Sample activity from retina neurons
        for (size_t y = 0; y < 64; y++) {
            for (size_t x = 0; x < 64; x++) {
                if (vision.isRetinaActive(x, y)) {
                    // Active pixel
                    predicted[y * 64 + x] = 255;
                }
            }
        }

        // Compare with expected
        return ArcLoader::compareImagesWithTolerance(predicted, expectedOutput, 28);
    }
};

// ========================================
// Main Benchmark
// ========================================

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "Phase 13: Real ARC-AGI Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // Determine data file
    std::string dataFile = "arc_training.bin";
    if (argc > 1) {
        dataFile = argv[1];
    }

    // Load tasks
    std::cout << "Loading ARC data from: " << dataFile << std::endl;
    auto tasks = ArcLoader::load(dataFile);

    if (tasks.empty()) {
        std::cerr << "No tasks loaded. Run convert_arc.py first." << std::endl;
        return 1;
    }

    std::cout << "Loaded " << tasks.size() << " tasks" << std::endl;
    std::cout << std::endl;

    // Limit tasks if configured
    size_t maxTasks = ArcBenchConfig::MAX_TASKS > 0 ?
                      std::min(static_cast<size_t>(ArcBenchConfig::MAX_TASKS), tasks.size()) :
                      tasks.size();

    std::cout << "Evaluating " << maxTasks << " tasks..." << std::endl;
    std::cout << std::endl;

    // Run benchmark
    ArcSolver solver;
    std::vector<TaskResult> results;
    results.reserve(maxTasks);

    auto benchStart = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < maxTasks; i++) {
        const auto& task = tasks[i];

        if (ArcBenchConfig::VERBOSE) {
            std::cout << "[" << std::setw(3) << (i + 1) << "/" << maxTasks << "] "
                      << "Task " << task.id << " ("
                      << task.trainExamples.size() << " train, "
                      << task.testExamples.size() << " test)... " << std::flush;
        }

        TaskResult result = solver.solveTask(task);
        results.push_back(result);

        if (ArcBenchConfig::VERBOSE) {
            std::cout << "train=" << std::fixed << std::setprecision(1)
                      << (result.trainSimilarity * 100) << "% "
                      << "test=" << (result.testSimilarity * 100) << "% "
                      << (result.solved ? "[SOLVED]" : "")
                      << " (" << std::setprecision(0) << result.timeMs << "ms)"
                      << std::endl;
        }
    }

    auto benchEnd = std::chrono::high_resolution_clock::now();
    double totalTimeMs = std::chrono::duration<double, std::milli>(benchEnd - benchStart).count();

    // ========================================
    // Summary Statistics
    // ========================================

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "BENCHMARK RESULTS" << std::endl;
    std::cout << "========================================" << std::endl;

    int solvedCount = std::count_if(results.begin(), results.end(),
                                   [](const TaskResult& r) { return r.solved; });

    float avgTrainSim = std::accumulate(results.begin(), results.end(), 0.0f,
        [](float sum, const TaskResult& r) { return sum + r.trainSimilarity; }) / results.size();

    float avgTestSim = std::accumulate(results.begin(), results.end(), 0.0f,
        [](float sum, const TaskResult& r) { return sum + r.testSimilarity; }) / results.size();

    std::cout << "  Tasks Evaluated:    " << results.size() << std::endl;
    std::cout << "  Tasks Solved:       " << solvedCount << " ("
              << std::fixed << std::setprecision(1)
              << (100.0f * solvedCount / results.size()) << "%)" << std::endl;
    std::cout << "  Avg Train Similarity: " << (avgTrainSim * 100) << "%" << std::endl;
    std::cout << "  Avg Test Similarity:  " << (avgTestSim * 100) << "%" << std::endl;
    std::cout << "  Total Time:         " << std::setprecision(0) << totalTimeMs << " ms" << std::endl;
    std::cout << "  Avg Time/Task:      " << std::setprecision(1)
              << (totalTimeMs / results.size()) << " ms" << std::endl;
    std::cout << "========================================" << std::endl;

    // Save detailed results
    std::ofstream csvFile("arc_benchmark_results.csv");
    csvFile << "TaskID,NumTrain,NumTest,TrainSimilarity,TestSimilarity,Solved,TimeMs" << std::endl;
    for (const auto& r : results) {
        csvFile << r.taskId << ","
                << r.numTrainExamples << ","
                << r.numTestExamples << ","
                << std::fixed << std::setprecision(4) << r.trainSimilarity << ","
                << r.testSimilarity << ","
                << (r.solved ? 1 : 0) << ","
                << std::setprecision(1) << r.timeMs << std::endl;
    }
    csvFile.close();
    std::cout << "Detailed results saved to: arc_benchmark_results.csv" << std::endl;

    return 0;
}
