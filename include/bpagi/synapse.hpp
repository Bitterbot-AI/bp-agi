#pragma once
#include "types.hpp"

namespace bpagi {

// Eligibility trace constants for operant conditioning
// The trace bridges the gap between action (T=10) and reward (T=50)
constexpr int8_t ELIGIBILITY_MAX = 100;      // Max trace value when Hebbian event occurs
constexpr int8_t ELIGIBILITY_DECAY = 1;      // Decay per tick
constexpr int32_t REWARD_SCALE_FACTOR = 50;  // Divisor for reward application

// Synapse structure as specified in Blueprint Section 1.1
// Implements discrete weight values and STDP plasticity
// Extended with eligibility traces for operant conditioning
struct Synapse {
    NeuronId targetNeuronIndex;  // ID of post-synaptic neuron
    Weight weight;               // Discrete weight (-16 to +16)
    bool isHebbian;              // Is this synapse plastic?
    int8_t eligibilityTrace;     // Eligibility trace for reward learning

    // Default constructor
    Synapse();

    // Parameterized constructor
    Synapse(NeuronId target, Weight w, bool plastic = true);

    // Mark synapse as eligible for reward (called when Hebbian event occurs)
    // Instead of updating weight immediately, sets eligibility trace
    // deltaT = postFiredTime - preFiredTime
    void markEligible(Tick preFired, Tick postFired);

    // Legacy STDP weight update (immediate, for Pavlovian mode)
    void updateWeight(Tick preFired, Tick postFired);

    // Decay eligibility trace by 1 (called every tick)
    void decayEligibility();

    // Apply reward signal to this synapse
    // weight += (eligibilityTrace * rewardAmount) / REWARD_SCALE_FACTOR
    void applyReward(int rewardAmount);

    // Clamp weight to valid range
    void clampWeight();
};

// STDP helper functions

// Calculate weight change based on timing difference
// Uses integer arithmetic only
Weight calculateSTDPDelta(Tick deltaT);

}  // namespace bpagi
