#pragma once
#include "types.hpp"
#include "network.hpp"
#include "vision.hpp"
#include "uks.hpp"
#include "hippocampus.hpp"
#include <vector>
#include <unordered_set>
#include <optional>

namespace bpagi {

/**
 * Brain: The Integration Layer
 *
 * Connects the Vision System (V2 Boundary/Line Layer) to the UKS (Recognition Bus).
 * This is the "Axon Bundle" that translates visual features into conceptual patterns.
 *
 * Architecture:
 *   Retina (L1) -> Boundary Detectors (V1) -> Line Integrators (V2)
 *                                                      |
 *                                              [Axon Bundle]
 *                                                      |
 *                                                      v
 *                                         UKS Recognition Bus -> Cortical Columns
 *
 * The Axon Bundle hashes the active boundary neuron coordinates to bus indices,
 * creating a compact representation that the UKS can learn and recognize.
 */
class Brain {
public:
    // Configuration
    struct Config {
        size_t numColumns;       // UKS cortical columns
        size_t busWidth;         // UKS bus width (default 64)
        bool enableLearning;     // Enable one-shot learning

        Config()
            : numColumns(100)
            , busWidth(64)
            , enableLearning(true)
        {}
    };

    // Constructor
    explicit Brain(const Config& config = Config());

    // ========================================
    // Main Interface
    // ========================================

    // Present an image to the visual system
    void present(const std::vector<uint8_t>& image);

    // Run one simulation tick
    // Returns the ID of the recognizing column, or nullopt if novel/no recognition
    std::optional<uint32_t> step();

    // Run multiple simulation ticks
    // Returns the final active column (if any)
    std::optional<uint32_t> run(size_t ticks);

    // Reset all systems (clears learned weights too)
    void reset();

    // Reset short-term memory only (clears activations, keeps learned weights)
    // Use between test cases to allow learned patterns to persist
    void resetShortTermMemory();

    // ========================================
    // Neuromodulation Interface
    // ========================================

    // Inject dopamine (The "Save Button" - enables STDP learning)
    void injectDopamine(int amount);

    // Inject random noise to all neurons (Stochastic Resonance)
    // Used for "creative" guessing - shakes system out of local minima
    void injectNoise(int amplitude);

    // ========================================
    // Hippocampus Interface (Phase 18)
    // ========================================

    /**
     * Capture an episode after prediction error (failure).
     * Stores input/output pair in episodic memory for later replay.
     *
     * @param input The input retina (what was presented)
     * @param target The target retina (correct output)
     * @param surprise Prediction error magnitude (0-100)
     */
    void captureEpisode(const std::vector<uint8_t>& input,
                        const std::vector<uint8_t>& target,
                        int surprise);

    /**
     * Dream: Consolidate memories through replay.
     *
     * The Secret Weapon:
     *   1. Disable sensory input (blindfold the brain)
     *   2. Fetch high-surprise episode from Hippocampus
     *   3. Inject input retina into Vision Bus
     *   4. Wait for propagation
     *   5. Force target retina onto Output Bus
     *   6. Inject massive dopamine (force STDP to wire A->B)
     *   7. Repeat
     *
     * This turns 1 failure into 1000 training iterations.
     *
     * @param episodes Number of episodes to replay
     * @param ticksPerEpisode Ticks to run per episode
     * @param dopamineLevel Dopamine intensity for consolidation
     */
    void dream(int episodes, int ticksPerEpisode = 10, int dopamineLevel = 200);

    /**
     * Get number of stored episodes in hippocampus.
     */
    size_t getEpisodeCount() const { return hippocampus_.size(); }

    /**
     * Get total surprise level (for monitoring learning progress).
     */
    int getTotalSurprise() const { return hippocampus_.getTotalSurprise(); }

    // ========================================
    // Query Interface
    // ========================================

    // Get the column that recognized the current input (if any)
    std::optional<uint32_t> getActiveColumn() const;

    // Check if the Request Neuron fired (novel input detected)
    bool didRequestFire() const;

    // Check if a new column was allocated this tick
    bool didAllocate() const;

    // Get the last allocated column ID
    std::optional<uint32_t> getLastAllocatedColumn() const;

    // Get number of allocated columns
    size_t getAllocatedCount() const;

    // ========================================
    // Component Access
    // ========================================

    Network& getNetwork() { return network_; }
    const Network& getNetwork() const { return network_; }

    VisionSystem& getVision() { return vision_; }
    const VisionSystem& getVision() const { return vision_; }

    UKS& getUKS() { return uks_; }
    const UKS& getUKS() const { return uks_; }

    Hippocampus& getHippocampus() { return hippocampus_; }
    const Hippocampus& getHippocampus() const { return hippocampus_; }

    // ========================================
    // Axon Bundle Interface
    // ========================================

    // Get the current visual pattern on the bus (as bus neuron indices)
    const std::vector<NeuronId>& getCurrentBusPattern() const { return currentBusPattern_; }

    // Get the number of active boundary neurons
    size_t getActiveBoundaryCount() const;

private:
    Config config_;

    // Core components
    Network network_;
    VisionSystem vision_;
    UKS uks_;
    Hippocampus hippocampus_;  // Phase 18: Episodic memory & dream replay

    // Current bus pattern (indices 0 to busWidth-1)
    std::vector<NeuronId> currentBusPattern_;

    // Accumulated bus pattern (all indices that have fired since present())
    std::unordered_set<NeuronId> accumulatedBusPattern_;

    // Stored image for continuous presentation
    std::vector<uint8_t> currentImage_;
    bool hasImage_;

    // Tracking state
    size_t prevAllocatedCount_;
    std::optional<uint32_t> lastAllocatedColumn_;
    bool didAllocate_;

    // Pattern stabilization: wait for acute vertices to accumulate
    // before presenting to UKS (fixes timing issue where early partial
    // patterns cause false recognition)
    size_t ticksSincePresent_;
    bool patternPresentedToUKS_;  // Only present once after stabilization
    static constexpr size_t STABILIZATION_TICKS = 8;  // Wait 8 ticks for full feature cascade

    // ========================================
    // Axon Bundle: Vision -> UKS Mapping
    // ========================================

    // Convert active boundary neurons to bus pattern indices
    void updateBusPattern();

    // Hash a boundary position to a bus index
    NeuronId hashToBusIndex(size_t x, size_t y, BoundaryType type) const;
};

}  // namespace bpagi
