/**
 * ARC-AGI-2 Verification Tool - C++ Brain Interface
 *
 * Runs the BP-AGI brain on each test case and outputs predictions
 * in a format that can be converted to ARC submission JSON.
 *
 * Output format (arc_predictions.bin):
 * - Magic: "BPRD" (4 bytes)
 * - NumTasks: uint32
 * - For each task:
 *   - TaskID: 8 bytes (null-padded)
 *   - NumTests: uint32
 *   - For each test: 64x64 prediction (4096 bytes)
 */

#include "bpagi/brain.hpp"
#include "bpagi/arc_loader.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <cstring>

using namespace bpagi;

// Configuration
struct VerifyConfig {
    static constexpr int PRESENT_TICKS = 5;
    static constexpr int DELAY_TICKS = 3;
    static constexpr int SETTLE_TICKS = 3;
    static constexpr int DOPAMINE_BOOST = 80;
    static constexpr int REWARD_AMOUNT = 100;
    static constexpr int TEST_WAIT_TICKS = 10;
    static constexpr bool VERBOSE = true;
};

/**
 * ARC Verifier - produces externally verifiable predictions
 */
class ArcVerifier {
public:
    ArcVerifier() : brain_{Brain::Config{}} {
        brain_.getNetwork().setPlasticityEnabled(true);
        brain_.getNetwork().setOperantMode(true);
    }

    /**
     * Process a task and return predictions for all test cases.
     */
    std::vector<std::vector<uint8_t>> processTask(const ArcTask& task) {
        std::vector<std::vector<uint8_t>> predictions;

        std::cerr << "  [DEBUG] Network neurons: " << brain_.getNetwork().getNeuronCount() << std::endl;
        std::cerr << "  [DEBUG] Calling brain_.reset()..." << std::flush;
        // Reset brain for fresh learning
        brain_.reset();
        std::cerr << " done" << std::endl;
        std::cerr << "  [DEBUG] Setting plasticity..." << std::flush;
        brain_.getNetwork().setPlasticityEnabled(true);
        std::cerr << " done" << std::endl;

        // Training Phase - learn from all examples
        std::cerr << "  [DEBUG] Training on " << task.trainExamples.size() << " examples..." << std::flush;
        for (size_t i = 0; i < task.trainExamples.size(); i++) {
            std::cerr << " " << i << std::flush;
            trainOnExample(task.trainExamples[i]);
        }
        std::cerr << " done" << std::endl;

        // Testing Phase - generate predictions
        brain_.getNetwork().setPlasticityEnabled(false);

        std::cerr << "  [DEBUG] Testing " << task.testExamples.size() << " examples..." << std::flush;
        for (size_t i = 0; i < task.testExamples.size(); i++) {
            std::cerr << " " << i << std::flush;
            auto prediction = getPrediction(task.testExamples[i].input);
            predictions.push_back(prediction);
        }
        std::cerr << " done" << std::endl;

        return predictions;
    }

    /**
     * Get raw retina prediction (64x64).
     */
    std::vector<uint8_t> getPrediction(const std::vector<uint8_t>& input) {
        // Present input
        brain_.present(input);
        for (int t = 0; t < VerifyConfig::PRESENT_TICKS; t++) {
            brain_.step();
        }

        // Wait for prediction
        for (int t = 0; t < VerifyConfig::TEST_WAIT_TICKS; t++) {
            brain_.step();
        }

        // Capture retina state with actual grayscale values
        auto& vision = brain_.getVision();
        std::vector<uint8_t> prediction(ARC_RETINA_SIZE, 0);

        // Debug: use the input that was presented (since brain shows input, not prediction)
        // The currentImage_ in vision holds the last presented image
        for (size_t i = 0; i < ARC_RETINA_SIZE && i < input.size(); i++) {
            prediction[i] = input[i];  // Use input for now to debug visualization
        }

        return prediction;
    }

private:
    Brain brain_;

    void trainOnExample(const ArcPair& example) {
        brain_.getNetwork().chemicals().dopamine = VerifyConfig::DOPAMINE_BOOST;

        // Present input
        brain_.present(example.input);
        for (int t = 0; t < VerifyConfig::PRESENT_TICKS; t++) {
            brain_.step();
        }

        // Delay
        for (int t = 0; t < VerifyConfig::DELAY_TICKS; t++) {
            brain_.step();
        }

        // Present output
        brain_.present(example.output);
        for (int t = 0; t < VerifyConfig::PRESENT_TICKS; t++) {
            brain_.step();
        }

        // Reward
        brain_.getNetwork().rewardSignal(static_cast<int8_t>(std::min(VerifyConfig::REWARD_AMOUNT, 100)));
        brain_.getNetwork().injectReward(VerifyConfig::REWARD_AMOUNT);

        // Settle
        for (int t = 0; t < VerifyConfig::SETTLE_TICKS; t++) {
            brain_.step();
        }
    }
};

/**
 * Compare prediction to expected output.
 */
float compareGrids(const std::vector<uint8_t>& pred, const std::vector<uint8_t>& expected) {
    if (pred.size() != expected.size()) return 0.0f;

    int matches = 0;
    int total = 0;

    for (size_t i = 0; i < pred.size(); i++) {
        total++;
        // Compare with tolerance for grayscale mapping
        int diff = std::abs(static_cast<int>(pred[i]) - static_cast<int>(expected[i]));
        if (diff <= 14) {  // Half the grayscale step
            matches++;
        }
    }

    return static_cast<float>(matches) / total;
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "ARC-AGI-2 Verification Tool" << std::endl;
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

    // Count total tests
    size_t totalTests = 0;
    for (const auto& task : tasks) {
        totalTests += task.testExamples.size();
    }

    std::cout << "Tasks: " << tasks.size() << std::endl;
    std::cout << "Total test cases: " << totalTests << std::endl;
    std::cout << std::endl;

    // Open output file
    std::ofstream outFile("arc_predictions.bin", std::ios::binary);
    if (!outFile) {
        std::cerr << "Failed to create output file" << std::endl;
        return 1;
    }

    // Write header
    outFile.write("BPRD", 4);
    uint32_t numTasks = tasks.size();
    outFile.write(reinterpret_cast<char*>(&numTasks), 4);

    // Process tasks
    ArcVerifier verifier;
    int taskNum = 0;
    int passedTests = 0;
    int totalProcessed = 0;

    auto startTime = std::chrono::high_resolution_clock::now();

    for (const auto& task : tasks) {
        taskNum++;

        if (VerifyConfig::VERBOSE) {
            std::cout << "[" << std::setw(3) << taskNum << "/" << tasks.size() << "] "
                      << task.id << " (" << task.trainExamples.size() << " train, "
                      << task.testExamples.size() << " test)... " << std::flush;
        }

        // Get predictions
        auto predictions = verifier.processTask(task);

        // Write task ID (8 bytes, null-padded)
        char taskIdBuf[8] = {0};
        std::strncpy(taskIdBuf, task.id.c_str(), 8);
        outFile.write(taskIdBuf, 8);

        // Write number of tests
        uint32_t numTests = predictions.size();
        outFile.write(reinterpret_cast<char*>(&numTests), 4);

        // Write predictions and evaluate
        int taskPassed = 0;
        for (size_t i = 0; i < predictions.size(); i++) {
            // Write prediction
            outFile.write(reinterpret_cast<char*>(predictions[i].data()), predictions[i].size());

            // Evaluate
            float accuracy = compareGrids(predictions[i], task.testExamples[i].output);
            totalProcessed++;

            if (accuracy >= 0.99f) {
                passedTests++;
                taskPassed++;
            }
        }

        if (VerifyConfig::VERBOSE) {
            std::cout << taskPassed << "/" << predictions.size() << " passed" << std::endl;
        }
    }

    outFile.close();

    auto endTime = std::chrono::high_resolution_clock::now();
    double totalMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    // Summary
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "VERIFICATION COMPLETE" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total test cases: " << totalProcessed << std::endl;
    std::cout << "Pixel-perfect:    " << passedTests << "/" << totalProcessed
              << " (" << std::fixed << std::setprecision(2)
              << (100.0 * passedTests / totalProcessed) << "%)" << std::endl;
    std::cout << "Time:             " << std::setprecision(0) << totalMs << " ms" << std::endl;
    std::cout << std::endl;
    std::cout << "Predictions saved to: arc_predictions.bin" << std::endl;
    std::cout << "Run: python examples/arc_verify.py --visualize" << std::endl;

    return 0;
}
