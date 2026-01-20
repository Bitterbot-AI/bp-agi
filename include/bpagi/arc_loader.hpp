#pragma once
/**
 * Phase 13: ARC Data Loader
 *
 * High-speed binary loader for ARC-AGI benchmark data.
 * Reads the binary format produced by convert_arc.py.
 *
 * Binary Format:
 *   [Magic: 4 bytes "BARC"]
 *   [NumTasks: 4 bytes uint32]
 *   For each Task:
 *     [ID: 8 bytes, null-padded]
 *     [NumTrain: 4 bytes uint32]
 *     [NumTest: 4 bytes uint32]
 *     [Train pairs: Input (4096 bytes) + Output (4096 bytes)]...
 *     [Test pairs: Input (4096 bytes) + Output (4096 bytes)]...
 */

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdint>
#include <algorithm>

namespace bpagi {

// Retina size matches the Vision System
constexpr size_t ARC_RETINA_SIZE = 64 * 64;  // 4096 bytes per image

/**
 * An input/output pair from an ARC task.
 */
struct ArcPair {
    std::vector<uint8_t> input;   // 64x64 grayscale image
    std::vector<uint8_t> output;  // 64x64 grayscale image

    ArcPair() : input(ARC_RETINA_SIZE, 0), output(ARC_RETINA_SIZE, 0) {}
};

/**
 * A complete ARC task with training and test examples.
 */
struct ArcTask {
    std::string id;                        // Task identifier (e.g., "007bbfb7")
    std::vector<ArcPair> trainExamples;    // Training demonstrations
    std::vector<ArcPair> testExamples;     // Test cases to solve

    // Get the total number of examples
    size_t totalExamples() const {
        return trainExamples.size() + testExamples.size();
    }
};

/**
 * ARC dataset loader.
 * Efficiently loads pre-converted binary ARC data.
 */
class ArcLoader {
public:
    /**
     * Load all tasks from a binary ARC file.
     *
     * @param filename Path to the .bin file (e.g., "arc_training.bin")
     * @return Vector of loaded ARC tasks
     */
    static std::vector<ArcTask> load(const std::string& filename) {
        std::vector<ArcTask> tasks;
        std::ifstream file(filename, std::ios::binary);

        if (!file.is_open()) {
            std::cerr << "ArcLoader: Failed to open " << filename << std::endl;
            return tasks;
        }

        // Verify magic number
        char magic[4];
        file.read(magic, 4);
        if (std::string(magic, 4) != "BARC") {
            std::cerr << "ArcLoader: Invalid file format (bad magic)" << std::endl;
            return tasks;
        }

        // Read number of tasks
        uint32_t numTasks;
        file.read(reinterpret_cast<char*>(&numTasks), sizeof(numTasks));

        tasks.reserve(numTasks);

        for (uint32_t i = 0; i < numTasks; i++) {
            ArcTask task;

            // Read task ID (8 bytes, null-padded)
            char idBuffer[8];
            file.read(idBuffer, 8);
            task.id = std::string(idBuffer);
            // Trim null padding
            auto nullPos = task.id.find('\0');
            if (nullPos != std::string::npos) {
                task.id.resize(nullPos);
            }

            // Read counts
            uint32_t numTrain, numTest;
            file.read(reinterpret_cast<char*>(&numTrain), sizeof(numTrain));
            file.read(reinterpret_cast<char*>(&numTest), sizeof(numTest));

            // Load training examples
            task.trainExamples.reserve(numTrain);
            for (uint32_t j = 0; j < numTrain; j++) {
                ArcPair pair;
                pair.input.resize(ARC_RETINA_SIZE);
                pair.output.resize(ARC_RETINA_SIZE);
                file.read(reinterpret_cast<char*>(pair.input.data()), ARC_RETINA_SIZE);
                file.read(reinterpret_cast<char*>(pair.output.data()), ARC_RETINA_SIZE);
                task.trainExamples.push_back(std::move(pair));
            }

            // Load test examples
            task.testExamples.reserve(numTest);
            for (uint32_t j = 0; j < numTest; j++) {
                ArcPair pair;
                pair.input.resize(ARC_RETINA_SIZE);
                pair.output.resize(ARC_RETINA_SIZE);
                file.read(reinterpret_cast<char*>(pair.input.data()), ARC_RETINA_SIZE);
                file.read(reinterpret_cast<char*>(pair.output.data()), ARC_RETINA_SIZE);
                task.testExamples.push_back(std::move(pair));
            }

            tasks.push_back(std::move(task));
        }

        return tasks;
    }

    /**
     * Load a single task by ID.
     *
     * @param filename Path to the .bin file
     * @param taskId The task ID to find
     * @return The task if found, or an empty task
     */
    static ArcTask loadTask(const std::string& filename, const std::string& taskId) {
        auto tasks = load(filename);
        for (auto& task : tasks) {
            if (task.id == taskId) {
                return task;
            }
        }
        return ArcTask{};  // Not found
    }

    /**
     * Compare two images and return similarity score.
     *
     * @param a First image (64x64 grayscale)
     * @param b Second image (64x64 grayscale)
     * @return Similarity score from 0.0 (completely different) to 1.0 (identical)
     */
    static float compareImages(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
        if (a.size() != b.size() || a.empty()) {
            return 0.0f;
        }

        size_t matches = 0;
        for (size_t i = 0; i < a.size(); i++) {
            if (a[i] == b[i]) {
                matches++;
            }
        }

        return static_cast<float>(matches) / static_cast<float>(a.size());
    }

    /**
     * Compare two images with tolerance for similar grayscale values.
     *
     * @param a First image
     * @param b Second image
     * @param tolerance Maximum difference in grayscale value to consider "matching"
     * @return Similarity score
     */
    static float compareImagesWithTolerance(const std::vector<uint8_t>& a,
                                            const std::vector<uint8_t>& b,
                                            int tolerance = 14) {
        if (a.size() != b.size() || a.empty()) {
            return 0.0f;
        }

        size_t matches = 0;
        for (size_t i = 0; i < a.size(); i++) {
            int diff = std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
            if (diff <= tolerance) {
                matches++;
            }
        }

        return static_cast<float>(matches) / static_cast<float>(a.size());
    }
};

}  // namespace bpagi
