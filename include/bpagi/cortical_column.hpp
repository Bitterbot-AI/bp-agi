#pragma once
#include "types.hpp"
#include <vector>
#include <cstdint>

namespace bpagi {

// Forward declaration
class Network;

/**
 * CorticalColumn: A cognitive module wrapping a cluster of neurons
 *
 * Blueprint: Cortical columns are the basic unit of semantic representation.
 * Each column represents a single "concept" and contains:
 * - Input neurons: Receive signals from the Recognition Bus
 * - Pyramidal neurons: The main processing/output neurons
 * - Inhibitory interneurons: Implement lateral inhibition (WTA)
 *
 * The column operates in two modes:
 * 1. Recognition: If input pattern matches learned weights, column activates
 * 2. Learning: If allocated, one-shot wiring connects input to column
 */
struct CorticalColumn {
    // Column identity
    uint32_t columnId;

    // Neuron membership (~100 neurons per column)
    std::vector<NeuronId> inputNeurons;      // Receive from Recognition Bus
    std::vector<NeuronId> pyramidalNeurons;  // Main processing neurons
    NeuronId outputNeuron;                    // Single output (concept active signal)
    NeuronId inhibitoryNeuron;               // Local inhibition

    // State flags
    bool isAllocated;        // Is this column currently representing a concept?
    bool isActive;           // Did this column fire in the current tick?

    // Priming/attention mechanism
    Charge boostValue;       // Temporary excitatory boost (priming)

    // Learning statistics
    Tick allocatedAtTick;    // When was this column first allocated?
    uint32_t activationCount; // How many times has this column fired?

    // Constructor
    CorticalColumn();

    // Initialize column with neuron IDs from the network
    void initialize(uint32_t id, const std::vector<NeuronId>& inputs,
                   const std::vector<NeuronId>& pyramidals,
                   NeuronId output, NeuronId inhibitory);

    // Reset column to unallocated state
    void reset();

    // Apply boost (priming) to input neurons
    void applyBoost(Network& net);

    // Check if column output is active
    bool checkActive(const Network& net) const;

    // Get the total number of neurons in this column
    size_t getNeuronCount() const;
};

// Column configuration constants
constexpr size_t COLUMN_INPUT_NEURONS = 20;      // Neurons receiving bus input
constexpr size_t COLUMN_PYRAMIDAL_NEURONS = 50;  // Main processing neurons
constexpr size_t COLUMN_TOTAL_NEURONS = COLUMN_INPUT_NEURONS + COLUMN_PYRAMIDAL_NEURONS + 2;

// Column neuron parameters (tuned for reliable recognition)
constexpr Charge COLUMN_INPUT_THRESHOLD = 3;      // Low threshold for input sensitivity
constexpr Charge COLUMN_PYRAMIDAL_THRESHOLD = 5;  // Moderate threshold
constexpr Charge COLUMN_OUTPUT_THRESHOLD = 8;     // Needs multiple pyramidals
constexpr Charge COLUMN_INHIBITORY_THRESHOLD = 3;

}  // namespace bpagi
