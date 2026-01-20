#pragma once
#include "types.hpp"
#include "network.hpp"
#include "cortical_column.hpp"
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <optional>

namespace bpagi {

/**
 * Universal Knowledge Store (UKS)
 *
 * Blueprint: The UKS is a dynamic knowledge graph implemented as a pool of
 * cortical columns. It provides:
 *
 * 1. Recognition Bus: Input pattern is broadcast to all columns
 * 2. Request Neuron: Global "novelty detector" - fires if no column recognizes input
 * 3. Suppressive Gate (WTA): Winner-Take-All circuit ensures only ONE column
 *    activates per input pattern
 * 4. One-Shot Learning: When Request fires, allocate a free column and
 *    immediately wire it to recognize the current input pattern
 *
 * This creates sparse, competitive activation - the "brakes" that prevent
 * runaway excitation (seizure mode).
 */
class UKS {
public:
    // Configuration
    struct Config {
        size_t numColumns;          // Number of cortical columns
        size_t busWidth;            // Width of recognition bus (input dimensions)
        Charge recognitionThreshold; // Threshold for column to claim recognition
        bool enableLearning;        // Allow new concept allocation

        Config()
            : numColumns(100)
            , busWidth(64)
            , recognitionThreshold(12)
            , enableLearning(true)
        {}
    };

    // Constructor
    explicit UKS(Network& network, const Config& config = Config());

    // ========================================
    // Main Interface
    // ========================================

    // Present an input pattern to the Recognition Bus
    // Returns the ID of the recognizing column, or nullopt if novel
    std::optional<uint32_t> present(const std::vector<NeuronId>& inputPattern);

    // Step the UKS (call after network.step())
    void step();

    // Reset all columns to unallocated state
    void reset();

    // ========================================
    // Query Interface
    // ========================================

    // Get the column that fired in the last step (if any)
    std::optional<uint32_t> getActiveColumn() const;

    // Check if the Request Neuron fired (novel input detected)
    bool didRequestFire() const;

    // Get number of allocated columns
    size_t getAllocatedCount() const;

    // Get number of free columns remaining
    size_t getFreeCount() const;

    // Get column by ID
    const CorticalColumn& getColumn(uint32_t id) const;

    // Get all columns
    const std::vector<CorticalColumn>& getColumns() const { return columns_; }

    // ========================================
    // Recognition Bus Interface
    // ========================================

    // Get the current input pattern on the bus
    const std::vector<NeuronId>& getCurrentInput() const { return currentInput_; }

    // Get bus neuron IDs
    const std::vector<NeuronId>& getBusNeurons() const { return busNeurons_; }

    // ========================================
    // Statistics
    // ========================================

    size_t getTotalAllocations() const { return totalAllocations_; }
    size_t getTotalRecognitions() const { return totalRecognitions_; }

    // ========================================
    // Neuromodulated Search (Phase 20)
    // ========================================

    // Get serotonin-modulated search depth for graph traversal
    // High 5-HT = deep search (patient), Low 5-HT = shallow (impulsive)
    // Returns 3-8 hops based on current serotonin level
    int getSearchDepth() const {
        return 3 + (network_.getChemicals().serotonin / 20);  // 3-8 range
    }

private:
    // Reference to underlying neural network
    Network& network_;
    Config config_;

    // Cortical columns pool
    std::vector<CorticalColumn> columns_;

    // Recognition Bus neurons (input layer)
    std::vector<NeuronId> busNeurons_;

    // Request Neuron (novelty detector)
    NeuronId requestNeuron_;

    // Global Inhibitory Neuron (WTA enforcer)
    NeuronId globalInhibitor_;

    // Current state
    std::vector<NeuronId> currentInput_;
    std::optional<uint32_t> activeColumn_;
    bool requestFired_;

    // Statistics
    size_t totalAllocations_;
    size_t totalRecognitions_;

    // ========================================
    // Internal Methods
    // ========================================

    // Build the neural infrastructure
    void buildBus();
    void buildColumns();
    void buildWTACircuit();

    // WTA logic
    void runWTASelection();

    // One-shot learning
    void allocateColumn(uint32_t columnId, const std::vector<NeuronId>& pattern);

    // Find a free column for allocation
    std::optional<uint32_t> findFreeColumn() const;

    // Suppress all columns except the winner
    void suppressOthers(uint32_t winnerId);

    // Check which columns are responding to current input
    std::vector<uint32_t> getRespondingColumns() const;
};

}  // namespace bpagi
