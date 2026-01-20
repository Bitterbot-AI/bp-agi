#include "bpagi/motor.hpp"
#include <algorithm>
#include <random>

namespace bpagi {

// Thread-local RNG for exploration
static thread_local std::mt19937 rng(std::random_device{}());

MotorSystem::MotorSystem(Network& network, const Config& config)
    : network_(network)
    , config_(config)
    , lastAction_(MotorAction::NONE)
{
    // Create motor neurons
    // These have moderate threshold and leak to respond to learned patterns
    motorLeft_ = network_.addNeuron(config_.motorThreshold, config_.motorLeak, config_.motorRefractory);
    motorRight_ = network_.addNeuron(config_.motorThreshold, config_.motorLeak, config_.motorRefractory);
}

void MotorSystem::connectToBus(const std::vector<NeuronId>& busNeurons) {
    // Create PLASTIC (learnable) synapses from bus neurons to motor neurons
    // Initial weight = 0 (Tabula Rasa)
    //
    // The magic happens via STDP:
    //   1. Bus neuron fires (visual pattern detected)
    //   2. Coach forces motor neuron to fire
    //   3. STDP detects: pre (bus) fired before post (motor)
    //   4. Connection is STRENGTHENED
    //
    // After many repetitions, the visual pattern alone triggers the motor neuron

    for (NeuronId bus : busNeurons) {
        // Connect each bus neuron to BOTH motor neurons
        // Learning will differentiate which connections strengthen

        // Bus -> Motor Left (plastic, initial weight = 0)
        network_.connectNeurons(bus, motorLeft_, 0, true);  // PLASTIC!
        leftConnections_.push_back({bus, motorLeft_});

        // Bus -> Motor Right (plastic, initial weight = 0)
        network_.connectNeurons(bus, motorRight_, 0, true);  // PLASTIC!
        rightConnections_.push_back({bus, motorRight_});
    }
}

void MotorSystem::connectColumn(NeuronId columnOutput, MotorAction action, Weight initialWeight) {
    // Connect a column's output directly to a motor neuron
    // This allows high-level concepts to trigger actions

    NeuronId target = (action == MotorAction::LEFT) ? motorLeft_ : motorRight_;
    network_.connectNeurons(columnOutput, target, initialWeight, true);  // PLASTIC

    if (action == MotorAction::LEFT) {
        leftConnections_.push_back({columnOutput, motorLeft_});
    } else {
        rightConnections_.push_back({columnOutput, motorRight_});
    }
}

void MotorSystem::forceAction(MotorAction action, Charge amount) {
    // The "Coach" directly stimulates a motor neuron
    // This forces the action and creates the post-synaptic spike
    // that STDP needs to strengthen pre-synaptic connections

    switch (action) {
        case MotorAction::LEFT:
            network_.injectCharge(motorLeft_, amount);
            break;
        case MotorAction::RIGHT:
            network_.injectCharge(motorRight_, amount);
            break;
        case MotorAction::NONE:
            // Do nothing
            break;
    }
}

void MotorSystem::injectExploration(int explorationRate, Charge amount) {
    // Exploration is essential for operant learning
    // Without trying actions, the brain can't learn their consequences
    //
    // This mimics biological spontaneous neural activity:
    // - Motor neurons occasionally fire randomly
    // - This creates actions that may lead to reward/punishment
    // - The eligibility trace captures which actions led to outcomes
    //
    // The exploration rate should decrease as learning progresses
    // (exploitation vs exploration trade-off)

    std::uniform_int_distribution<int> dist(0, 99);

    // Randomly inject charge into left motor
    if (dist(rng) < explorationRate) {
        network_.injectCharge(motorLeft_, amount);
    }

    // Randomly inject charge into right motor
    if (dist(rng) < explorationRate) {
        network_.injectCharge(motorRight_, amount);
    }
}

MotorSystem::MotorAction MotorSystem::getAction() const {
    bool leftFired = network_.didFire(motorLeft_);
    bool rightFired = network_.didFire(motorRight_);

    if (leftFired && !rightFired) {
        lastAction_ = MotorAction::LEFT;
    } else if (rightFired && !leftFired) {
        lastAction_ = MotorAction::RIGHT;
    } else if (leftFired && rightFired) {
        // Both fired - use the one with more charge (tie-breaker)
        lastAction_ = (network_.getCharge(motorLeft_) >= network_.getCharge(motorRight_))
                      ? MotorAction::LEFT : MotorAction::RIGHT;
    } else {
        lastAction_ = MotorAction::NONE;
    }

    return lastAction_;
}

bool MotorSystem::didFire(MotorAction action) const {
    switch (action) {
        case MotorAction::LEFT:  return network_.didFire(motorLeft_);
        case MotorAction::RIGHT: return network_.didFire(motorRight_);
        default: return false;
    }
}

Charge MotorSystem::getCharge(MotorAction action) const {
    switch (action) {
        case MotorAction::LEFT:  return network_.getCharge(motorLeft_);
        case MotorAction::RIGHT: return network_.getCharge(motorRight_);
        default: return 0;
    }
}

NeuronId MotorSystem::getMotorNeuron(MotorAction action) const {
    switch (action) {
        case MotorAction::LEFT:  return motorLeft_;
        case MotorAction::RIGHT: return motorRight_;
        default: return INVALID_NEURON;
    }
}

float MotorSystem::getAverageWeight(MotorAction action) const {
    const auto& connections = (action == MotorAction::LEFT) ? leftConnections_ : rightConnections_;

    if (connections.empty()) return 0.0f;

    int64_t totalWeight = 0;
    int count = 0;

    for (const auto& [src, _] : connections) {
        // Get the weight of the synapse from src to the motor neuron
        // Note: This is simplified - in a full implementation we'd track synapse indices
        totalWeight += network_.getSynapseWeight(src, (action == MotorAction::LEFT) ? motorLeft_ : motorRight_);
        count++;
    }

    return (count > 0) ? static_cast<float>(totalWeight) / count : 0.0f;
}

int MotorSystem::getTotalWeight(MotorAction action) const {
    const auto& connections = (action == MotorAction::LEFT) ? leftConnections_ : rightConnections_;

    int totalWeight = 0;
    NeuronId target = (action == MotorAction::LEFT) ? motorLeft_ : motorRight_;

    for (const auto& [src, _] : connections) {
        totalWeight += network_.getSynapseWeight(src, target);
    }

    return totalWeight;
}

}  // namespace bpagi
