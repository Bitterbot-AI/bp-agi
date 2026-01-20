#pragma once
/**
 * Phase 19: Training Protocol (Pinned for Hummingbird)
 *
 * Implements the "Cramming Protocol" - biologically plausible training
 * that mimics how humans study for exams:
 *
 *   1. First exposure to all training puzzles
 *   2. Sleep phase (dream replay of failures)
 *   3. Re-exposure with stronger memories
 *   4. Repeat until plateau
 *   5. Test on unseen puzzles
 *
 * Key Difference from ML Training:
 *   - No gradient descent or backpropagation
 *   - Hebbian/STDP learning during experience
 *   - Hippocampal replay during "sleep"
 *   - Learns PATTERNS, not pixel memorization
 */

#include "brain.hpp"
#include "arc_loader.hpp"
#include <vector>
#include <functional>
#include <iostream>
#include <iomanip>

namespace bpagi {

// ============================================
// Training Statistics
// ============================================
struct RoundStats {
    int round;
    int totalTasks;
    int exactMatches;      // 100% pixel match
    int nearMatches;       // 99%+ match
    int episodesCaptured;  // Failures stored in hippocampus
    double avgAccuracy;    // Mean accuracy across all tasks

    double exactRate() const {
        return totalTasks > 0 ? 100.0 * exactMatches / totalTasks : 0.0;
    }

    double nearRate() const {
        return totalTasks > 0 ? 100.0 * nearMatches / totalTasks : 0.0;
    }
};

// ============================================
// Training Protocol Configuration
// ============================================
struct TrainingConfig {
    // Training rounds
    int maxRounds = 5;              // Maximum training cycles
    double plateauThreshold = 2.0;  // Stop if improvement < this %

    // Per-task timing
    int presentTicks = 20;          // Ticks to present input
    int consolidationTicks = 10;    // Ticks after showing output
    int inferenceTicks = 30;        // Ticks for test inference

    // Neuromodulation
    int learningDopamine = 100;     // Dopamine during learning
    int dreamDopamine = 200;        // Dopamine during replay (higher!)

    // Dream parameters
    int dreamEpisodesPerRound = 5000;  // Replays per sleep phase
    int dreamTicksPerEpisode = 10;     // Ticks per replay

    // Failure capture
    int minSurpriseToCapture = 5;   // Only capture if error > this %

    // Callbacks
    std::function<void(const RoundStats&)> onRoundComplete = nullptr;
    std::function<void(int, const std::string&, double)> onTaskComplete = nullptr;
};

// ============================================
// Training Protocol
// ============================================
class TrainingProtocol {
public:
    explicit TrainingProtocol(Brain& brain, const TrainingConfig& config = TrainingConfig())
        : brain_(brain)
        , config_(config)
    {}

    /**
     * Run the full training protocol on training data.
     * Returns statistics for each round.
     */
    std::vector<RoundStats> train(const std::vector<ArcTask>& tasks) {
        std::vector<RoundStats> allStats;
        double prevExactRate = 0.0;

        for (int round = 1; round <= config_.maxRounds; round++) {
            std::cout << "\n========================================" << std::endl;
            std::cout << "TRAINING ROUND " << round << "/" << config_.maxRounds << std::endl;
            std::cout << "========================================\n" << std::endl;

            // ========================================
            // Phase 1: Exposure (Training Pass)
            // ========================================
            RoundStats stats = runTrainingPass(tasks, round);
            allStats.push_back(stats);

            // Report
            std::cout << "\n--- Round " << round << " Summary ---" << std::endl;
            std::cout << "Exact matches: " << stats.exactMatches << "/" << stats.totalTasks
                      << " (" << std::fixed << std::setprecision(1) << stats.exactRate() << "%)" << std::endl;
            std::cout << "Near matches (99%+): " << stats.nearMatches << std::endl;
            std::cout << "Episodes captured: " << stats.episodesCaptured << std::endl;
            std::cout << "Average accuracy: " << std::setprecision(1) << stats.avgAccuracy << "%" << std::endl;

            if (config_.onRoundComplete) {
                config_.onRoundComplete(stats);
            }

            // Check for plateau
            double improvement = stats.exactRate() - prevExactRate;
            if (round > 1 && improvement < config_.plateauThreshold) {
                std::cout << "\nPlateau detected (improvement: " << improvement << "%)" << std::endl;
                std::cout << "Stopping training early." << std::endl;
                break;
            }
            prevExactRate = stats.exactRate();

            // ========================================
            // Phase 2: Sleep (Dream Replay)
            // ========================================
            if (round < config_.maxRounds && brain_.getEpisodeCount() > 0) {
                std::cout << "\n--- Sleep Phase ---" << std::endl;
                std::cout << "Replaying " << config_.dreamEpisodesPerRound
                          << " episodes from " << brain_.getEpisodeCount() << " stored..." << std::endl;

                brain_.dream(config_.dreamEpisodesPerRound,
                            config_.dreamTicksPerEpisode,
                            config_.dreamDopamine);

                std::cout << "Dream complete. Memories consolidated." << std::endl;
            }
        }

        return allStats;
    }

    /**
     * Evaluate on test data (no learning, no episode capture).
     * This is the true measure of generalization.
     */
    RoundStats evaluate(const std::vector<ArcTask>& tasks) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "EVALUATION (Unseen Puzzles)" << std::endl;
        std::cout << "========================================\n" << std::endl;

        // Disable learning for evaluation
        bool wasLearning = brain_.getNetwork().isPlasticityEnabled();
        brain_.getNetwork().setPlasticityEnabled(false);

        RoundStats stats = runEvaluationPass(tasks);

        // Restore learning state
        brain_.getNetwork().setPlasticityEnabled(wasLearning);

        std::cout << "\n--- Evaluation Summary ---" << std::endl;
        std::cout << "Exact matches: " << stats.exactMatches << "/" << stats.totalTasks
                  << " (" << std::fixed << std::setprecision(1) << stats.exactRate() << "%)" << std::endl;
        std::cout << "Near matches (99%+): " << stats.nearMatches << std::endl;
        std::cout << "Average accuracy: " << std::setprecision(1) << stats.avgAccuracy << "%" << std::endl;

        return stats;
    }

private:
    Brain& brain_;
    TrainingConfig config_;

    /**
     * Single training pass through all tasks.
     * Learning enabled, failures captured to hippocampus.
     */
    RoundStats runTrainingPass(const std::vector<ArcTask>& tasks, int round) {
        RoundStats stats{};
        stats.round = round;
        stats.totalTasks = 0;
        double totalAccuracy = 0.0;

        brain_.getNetwork().setPlasticityEnabled(true);

        int taskNum = 0;
        for (const auto& task : tasks) {
            taskNum++;

            // Reset short-term memory for each task (keep learned weights!)
            brain_.resetShortTermMemory();

            // ========================================
            // Learning Phase: Present training pairs
            // ========================================
            for (const auto& pair : task.trainExamples) {
                // Present input with dopamine (learning signal)
                brain_.injectDopamine(config_.learningDopamine);
                brain_.present(pair.input);
                for (int t = 0; t < config_.presentTicks; t++) {
                    brain_.step();
                }

                // Present output (target association)
                brain_.present(pair.output);
                for (int t = 0; t < config_.consolidationTicks; t++) {
                    brain_.step();
                }
            }

            // ========================================
            // Test Phase: Evaluate on test examples
            // ========================================
            brain_.getNetwork().setPlasticityEnabled(false);  // Freeze for inference

            for (const auto& testPair : task.testExamples) {
                stats.totalTasks++;

                brain_.resetShortTermMemory();
                brain_.present(testPair.input);
                for (int t = 0; t < config_.inferenceTicks; t++) {
                    brain_.step();
                }

                // Calculate accuracy (simplified - would need actual output decoding)
                // For now, use a placeholder accuracy based on column activation
                double accuracy = calculateAccuracy(testPair.input, testPair.output);
                totalAccuracy += accuracy;

                if (accuracy >= 99.95) {
                    stats.exactMatches++;
                } else if (accuracy >= 99.0) {
                    stats.nearMatches++;
                }

                // Capture failures to hippocampus
                if (accuracy < 100.0) {
                    int surprise = static_cast<int>(100.0 - accuracy);
                    if (surprise >= config_.minSurpriseToCapture) {
                        brain_.captureEpisode(testPair.input, testPair.output, surprise);
                        stats.episodesCaptured++;
                    }
                }

                if (config_.onTaskComplete) {
                    config_.onTaskComplete(taskNum, task.id, accuracy);
                }
            }

            brain_.getNetwork().setPlasticityEnabled(true);  // Re-enable for next task

            // Progress indicator
            if (taskNum % 20 == 0) {
                std::cout << "  [" << taskNum << "/" << tasks.size() << "] "
                          << stats.exactMatches << " exact so far..." << std::endl;
            }
        }

        stats.avgAccuracy = stats.totalTasks > 0 ? totalAccuracy / stats.totalTasks : 0.0;
        return stats;
    }

    /**
     * Evaluation pass - no learning, no capture.
     */
    RoundStats runEvaluationPass(const std::vector<ArcTask>& tasks) {
        RoundStats stats{};
        stats.round = 0;  // Evaluation, not training
        stats.totalTasks = 0;
        double totalAccuracy = 0.0;

        int taskNum = 0;
        for (const auto& task : tasks) {
            taskNum++;
            brain_.resetShortTermMemory();

            // Quick learning from training examples (inference mode)
            for (const auto& pair : task.trainExamples) {
                brain_.present(pair.input);
                for (int t = 0; t < config_.presentTicks; t++) {
                    brain_.step();
                }
                brain_.present(pair.output);
                for (int t = 0; t < config_.consolidationTicks / 2; t++) {
                    brain_.step();
                }
            }

            // Test
            for (const auto& testPair : task.testExamples) {
                stats.totalTasks++;

                brain_.resetShortTermMemory();
                brain_.present(testPair.input);
                for (int t = 0; t < config_.inferenceTicks; t++) {
                    brain_.step();
                }

                double accuracy = calculateAccuracy(testPair.input, testPair.output);
                totalAccuracy += accuracy;

                if (accuracy >= 99.95) {
                    stats.exactMatches++;
                } else if (accuracy >= 99.0) {
                    stats.nearMatches++;
                }
            }

            if (taskNum % 20 == 0) {
                std::cout << "  [" << taskNum << "/" << tasks.size() << "] "
                          << stats.exactMatches << " exact so far..." << std::endl;
            }
        }

        stats.avgAccuracy = stats.totalTasks > 0 ? totalAccuracy / stats.totalTasks : 0.0;
        return stats;
    }

    /**
     * Calculate accuracy between predicted and expected output.
     * TODO: This needs to actually decode brain output and compare.
     * For now, returns a placeholder based on pattern matching.
     */
    double calculateAccuracy(const std::vector<uint8_t>& input,
                            const std::vector<uint8_t>& expected) {
        // Placeholder - real implementation would:
        // 1. Get brain's output retina
        // 2. Decode to ARC grid
        // 3. Compare pixel-by-pixel with expected
        //
        // For now, return based on input/output similarity as proxy
        if (input.size() != expected.size()) return 0.0;

        size_t matches = 0;
        for (size_t i = 0; i < input.size(); i++) {
            if (input[i] == expected[i]) matches++;
        }
        return 100.0 * matches / input.size();
    }
};

}  // namespace bpagi
