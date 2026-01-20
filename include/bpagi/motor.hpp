#pragma once
#include "types.hpp"
#include "network.hpp"
#include <vector>

namespace bpagi {

/**
 * MotorSystem: The Output Interface
 *
 * Phase 7: Embodiment - Connecting Vision to Action
 *
 * This is the "Motor Cortex" that converts high-level visual concepts
 * into motor commands. The key innovation is that the connections
 * from visual concepts to motor neurons are LEARNED via Hebbian/STDP.
 *
 * Architecture:
 *   UKS Recognition Bus → [Plastic Synapses] → Motor Neurons
 *
 * The system starts with Weight=0 (Tabula Rasa) and learns associations
 * through Pavlovian conditioning:
 *   - See "Ball on Left" pattern
 *   - Coach forces "Move Left" action
 *   - STDP strengthens the connection
 *
 * After training, the visual pattern alone triggers the motor response.
 */
class MotorSystem {
public:
    // Motor neuron identifiers
    enum class MotorAction {
        LEFT = 0,
        RIGHT = 1,
        NONE = 2
    };

    // Configuration
    struct Config {
        Charge motorThreshold;    // Threshold for motor neuron to fire
        Charge motorLeak;         // Leak rate for motor neurons
        int32_t motorRefractory;  // Refractory period

        Config()
            : motorThreshold(8)
            , motorLeak(2)
            , motorRefractory(3)
        {}
    };

    // Constructor
    explicit MotorSystem(Network& network, const Config& config = Config());

    // ========================================
    // Connection Setup
    // ========================================

    // Connect the motor system to the UKS Recognition Bus
    // Creates PLASTIC (learnable) synapses from bus neurons to motor neurons
    void connectToBus(const std::vector<NeuronId>& busNeurons);

    // Connect a specific column's output to motor neurons
    // This allows concepts to trigger actions
    void connectColumn(NeuronId columnOutput, MotorAction action, Weight initialWeight = 0);

    // ========================================
    // External Input (Coach/Training)
    // ========================================

    // Inject charge directly into a motor neuron (the "Coach" forcing an action)
    // This is used during Pavlovian conditioning
    void forceAction(MotorAction action, Charge amount = 20);

    // ========================================
    // Exploration (Operant Conditioning)
    // ========================================

    // Inject random exploration activity into motor neurons
    // This is essential for operant learning - the brain must try actions
    // to learn their consequences.
    // explorationRate: probability (0-100) of injecting noise each call
    // amount: charge to inject when exploring
    void injectExploration(int explorationRate = 30, Charge amount = 15);

    // ========================================
    // Output Query
    // ========================================

    // Get the currently active action (based on which motor neuron fired)
    MotorAction getAction() const;

    // Check if a specific motor neuron fired
    bool didFire(MotorAction action) const;

    // Get the current charge of a motor neuron
    Charge getCharge(MotorAction action) const;

    // ========================================
    // Learning Query
    // ========================================

    // Get the neuron ID for a motor action
    NeuronId getMotorNeuron(MotorAction action) const;

    // Get the average synaptic weight from bus to a motor neuron
    // This measures how much the system has "learned" an association
    float getAverageWeight(MotorAction action) const;

    // Get the total incoming weight to a motor neuron
    int getTotalWeight(MotorAction action) const;

private:
    Network& network_;
    Config config_;

    // Motor neurons
    NeuronId motorLeft_;
    NeuronId motorRight_;

    // Track connections for weight analysis
    std::vector<std::pair<NeuronId, NeuronId>> leftConnections_;   // (source, synapse_idx)
    std::vector<std::pair<NeuronId, NeuronId>> rightConnections_;

    // Last action state
    mutable MotorAction lastAction_;
};

}  // namespace bpagi
