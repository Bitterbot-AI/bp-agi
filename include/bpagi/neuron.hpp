#pragma once
#include "types.hpp"
#include <algorithm>

namespace bpagi {

// Neuron structure as specified in Blueprint Section 1.1
// Implements Leaky Integrate-and-Fire (LIF) model with discrete timing
struct Neuron {
    Charge currentCharge;       // Membrane potential (V_m)
    Charge leakRate;            // Amount V_m decays per step (L)
    Charge threshold;           // Firing threshold (theta)
    Tick lastFiredStep;         // Timestamp of last spike
    int32_t refractoryDelay;    // Cycles before recharging after spike
    uint32_t synapseListIndex;  // Pointer to sparse synapse array
    uint16_t synapseCount;      // Number of outgoing connections

    // Default constructor - initialize with Blueprint defaults
    Neuron()
        : currentCharge(0)
        , leakRate(DEFAULT_LEAK)
        , threshold(DEFAULT_THRESHOLD)
        , lastFiredStep(-DEFAULT_REFRACTORY - 1)  // Allow immediate firing
        , refractoryDelay(DEFAULT_REFRACTORY)
        , synapseListIndex(0)
        , synapseCount(0)
    {}

    // Parameterized constructor
    Neuron(Charge thresh, Charge leak, int32_t refractory)
        : currentCharge(0)
        , leakRate(leak)
        , threshold(thresh)
        , lastFiredStep(-refractory - 1)  // Allow immediate firing
        , refractoryDelay(refractory)
        , synapseListIndex(0)
        , synapseCount(0)
    {}

    // Check if neuron is in refractory period
    // During refractory, neuron ignores all input (1+1=1 logic)
    inline bool isRefractory(Tick currentTick) const {
        return (currentTick - lastFiredStep) <= refractoryDelay;
    }

    // Apply leak decay: V_m = max(0, V_m - L)
    inline void applyLeak() {
        currentCharge = std::max(CHARGE_MIN, currentCharge - leakRate);
    }

    // Add charge from incoming synapse
    inline void addCharge(Charge amount) {
        currentCharge += amount;
    }

    // Check if neuron should fire and handle firing logic
    // Returns true if neuron fired this tick
    inline bool checkAndFire(Tick currentTick) {
        if (isRefractory(currentTick)) {
            return false;
        }
        if (currentCharge < CHARGE_MIN) {
            currentCharge = CHARGE_MIN;
        }
        if (currentCharge >= threshold) {
            currentCharge = 0;
            lastFiredStep = currentTick;
            return true;
        }
        return false;
    }

    // Reset neuron state
    inline void reset() {
        currentCharge = 0;
        lastFiredStep = -refractoryDelay - 1;
    }
};

}  // namespace bpagi
