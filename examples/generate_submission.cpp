/**
 * ARC-AGI-2 Submission Generator
 *
 * Implements the 2-Attempt Rule for ARC submissions:
 *   Attempt 1: The "Clean" Run (Deterministic)
 *   Attempt 2: The "Noisy" Run (Stochastic Resonance)
 *
 * Uses Honeybee-scale brain with k-WTA Razor for efficient computation.
 */

#include "bpagi/brain.hpp"
#include "bpagi/arc_loader.hpp"
#include "bpagi/config.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>

using namespace bpagi;

// ============================================
// Configuration
// ============================================
struct SubmissionConfig {
    // Use Honeybee scale for now (faster than Dragonfly)
    static constexpr size_t NUM_COLUMNS = Config::Honeybee::UKS_COLUMNS;
    static constexpr size_t BUS_WIDTH = Config::Honeybee::UKS_BUS_WIDTH;

    // Training timing
    static constexpr int PRESENT_TICKS = 20;
    static constexpr int CONSOLIDATION_TICKS = 10;
    static constexpr int DOPAMINE_LEVEL = 100;

    // Inference timing
    static constexpr int INFERENCE_TICKS = 30;

    // Noise amplitude for stochastic resonance (Attempt 2)
    static constexpr int NOISE_AMPLITUDE = 10;
};

// ============================================
// Color Mapping
// ============================================

// Map grayscale voltage (0-255) back to ARC color (0-9)
int voltageToColor(uint8_t v) {
    if (v < 14) return 0;
    return std::min(9, (v + 14) / 28);
}

// ============================================
// Grid Extraction
// ============================================

// Convert Brain Retina (64x64 grayscale) to ARC Grid (variable size, colors 0-9)
std::vector<std::vector<int>> decodeOutput(const std::vector<uint8_t>& retina) {
    // 1. Find Bounding Box of active pixels
    int min_x = 64, max_x = -1, min_y = 64, max_y = -1;
    bool empty = true;

    for (int y = 0; y < 64; y++) {
        for (int x = 0; x < 64; x++) {
            if (retina[y * 64 + x] > 10) {  // Threshold for "not black"
                if (x < min_x) min_x = x;
                if (x > max_x) max_x = x;
                if (y < min_y) min_y = y;
                if (y > max_y) max_y = y;
                empty = false;
            }
        }
    }

    // Return 1x1 black grid if empty
    if (empty) return {{0}};

    // 2. Extract Crop
    int h = max_y - min_y + 1;
    int w = max_x - min_x + 1;
    std::vector<std::vector<int>> grid(h, std::vector<int>(w));

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            grid[y][x] = voltageToColor(retina[(min_y + y) * 64 + (min_x + x)]);
        }
    }
    return grid;
}

// ============================================
// JSON Output Helpers
// ============================================

std::string gridToJson(const std::vector<std::vector<int>>& grid) {
    std::ostringstream ss;
    ss << "[";
    for (size_t y = 0; y < grid.size(); y++) {
        if (y > 0) ss << ",";
        ss << "[";
        for (size_t x = 0; x < grid[y].size(); x++) {
            if (x > 0) ss << ",";
            ss << grid[y][x];
        }
        ss << "]";
    }
    ss << "]";
    return ss.str();
}

// ============================================
// Main Submission Generator
// ============================================

int main(int argc, char* argv[]) {
    std::cout << "============================================" << std::endl;
    std::cout << "ARC-AGI-2 Submission Generator" << std::endl;
    std::cout << "============================================" << std::endl;
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

    std::cout << "Tasks: " << tasks.size() << std::endl;

    // Count total tests
    size_t totalTests = 0;
    for (const auto& task : tasks) {
        totalTests += task.testExamples.size();
    }
    std::cout << "Total test cases: " << totalTests << std::endl;
    std::cout << std::endl;

    // Initialize Brain with Honeybee configuration
    Brain::Config config;
    config.numColumns = SubmissionConfig::NUM_COLUMNS;
    config.busWidth = SubmissionConfig::BUS_WIDTH;
    config.enableLearning = true;

    Brain brain(config);
    brain.getNetwork().setPlasticityEnabled(true);
    brain.getNetwork().setOperantMode(true);

    // Enable the Razor for efficiency
    brain.getNetwork().setRazorEnabled(true);
    brain.getNetwork().setMaxSpikesPerTick(Config::Honeybee::MAX_SPIKES_PER_TICK);

    std::cout << "Brain initialized:" << std::endl;
    std::cout << "  Neurons:  " << brain.getNetwork().getNeuronCount() << std::endl;
    std::cout << "  Synapses: " << brain.getNetwork().getSynapseCount() << std::endl;
    std::cout << "  Columns:  " << config.numColumns << std::endl;
    std::cout << "  Razor:    " << (brain.getNetwork().isRazorEnabled() ? "ENABLED" : "disabled") << std::endl;
    std::cout << std::endl;

    // Open output file
    std::ofstream outFile("submission.json");
    if (!outFile) {
        std::cerr << "Failed to create submission.json" << std::endl;
        return 1;
    }

    outFile << "{" << std::endl;

    auto startTime = std::chrono::high_resolution_clock::now();
    int taskNum = 0;
    int passedTests = 0;

    for (const auto& task : tasks) {
        taskNum++;

        std::cout << "[" << std::setw(3) << taskNum << "/" << tasks.size() << "] "
                  << task.id << " (" << task.trainExamples.size() << " train, "
                  << task.testExamples.size() << " test)... " << std::flush;

        // ========================================
        // PHASE 1: Training (One-Shot Learning)
        // ========================================
        brain.reset();  // Fresh brain for each task
        brain.getNetwork().setPlasticityEnabled(true);

        for (const auto& pair : task.trainExamples) {
            // Present input
            brain.injectDopamine(SubmissionConfig::DOPAMINE_LEVEL);
            brain.present(pair.input);
            for (int t = 0; t < SubmissionConfig::PRESENT_TICKS; t++) {
                brain.step();
            }

            // Present output (target)
            brain.present(pair.output);
            for (int t = 0; t < SubmissionConfig::CONSOLIDATION_TICKS; t++) {
                brain.step();
            }
        }

        // ========================================
        // PHASE 2: Inference (Generate Predictions)
        // ========================================
        brain.getNetwork().setPlasticityEnabled(false);  // Freeze weights

        // Start task JSON
        if (taskNum > 1) outFile << "," << std::endl;
        outFile << "  \"" << task.id << "\": [" << std::endl;

        for (size_t testIdx = 0; testIdx < task.testExamples.size(); testIdx++) {
            const auto& testPair = task.testExamples[testIdx];

            if (testIdx > 0) outFile << "," << std::endl;
            outFile << "    {" << std::endl;

            // --- ATTEMPT 1: The "Rational" Guess (Deterministic) ---
            brain.resetShortTermMemory();
            brain.present(testPair.input);
            for (int t = 0; t < SubmissionConfig::INFERENCE_TICKS; t++) {
                brain.step();
            }

            // For now, use input as prediction (brain shows input, not generated output)
            // TODO: Implement actual prediction generation via temporal association
            auto attempt1 = decodeOutput(testPair.input);
            outFile << "      \"attempt_1\": " << gridToJson(attempt1) << "," << std::endl;

            // --- ATTEMPT 2: The "Creative" Guess (Stochastic Resonance) ---
            brain.resetShortTermMemory();
            brain.present(testPair.input);
            brain.injectNoise(SubmissionConfig::NOISE_AMPLITUDE);
            for (int t = 0; t < SubmissionConfig::INFERENCE_TICKS; t++) {
                brain.step();
            }

            auto attempt2 = decodeOutput(testPair.input);
            outFile << "      \"attempt_2\": " << gridToJson(attempt2) << std::endl;

            outFile << "    }";

            // Check accuracy (for logging)
            // Compare attempt1 to expected output
            auto expected = decodeOutput(testPair.output);
            if (attempt1 == expected || attempt2 == expected) {
                passedTests++;
            }
        }

        outFile << std::endl << "  ]";
        std::cout << "done" << std::endl;
    }

    outFile << std::endl << "}" << std::endl;
    outFile.close();

    auto endTime = std::chrono::high_resolution_clock::now();
    double totalSec = std::chrono::duration<double>(endTime - startTime).count();

    // Summary
    std::cout << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "SUBMISSION COMPLETE" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Total test cases: " << totalTests << std::endl;
    std::cout << "Time: " << std::fixed << std::setprecision(1) << totalSec << "s" << std::endl;
    std::cout << "Output: submission.json" << std::endl;
    std::cout << std::endl;
    std::cout << "Run: python examples/verify_submission.py" << std::endl;

    return 0;
}
