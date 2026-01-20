#include "bpagi/synapse.hpp"
#include <algorithm>
#include <cstdlib>

namespace bpagi {

Synapse::Synapse()
    : targetNeuronIndex(INVALID_NEURON)
    , weight(0)
    , isHebbian(true)
    , eligibilityTrace(0)
{
}

Synapse::Synapse(NeuronId target, Weight w, bool plastic)
    : targetNeuronIndex(target)
    , weight(w)
    , isHebbian(plastic)
    , eligibilityTrace(0)
{
    clampWeight();
}

void Synapse::clampWeight() {
    weight = std::clamp(weight, WEIGHT_MIN, WEIGHT_MAX);
}

void Synapse::markEligible(Tick preFired, Tick postFired) {
    // Operant Conditioning: Instead of updating weight immediately,
    // we mark the synapse as "eligible" for reward.
    // The eligibility trace decays over time, creating a temporal
    // credit assignment window.

    if (!isHebbian) {
        return;  // Non-plastic synapse, no eligibility
    }

    Tick deltaT = postFired - preFired;

    // Only mark eligible for causal (pre-before-post) relationships
    // This is the action that CAUSED the outcome
    if (deltaT > 0 && deltaT <= STDP_WINDOW) {
        // Set eligibility to max - the synapse is now "hot"
        // and waiting for a reward signal
        eligibilityTrace = ELIGIBILITY_MAX;
    }
}

void Synapse::updateWeight(Tick preFired, Tick postFired) {
    // Legacy STDP: Immediate weight update (for Pavlovian conditioning)
    if (!isHebbian) {
        return;  // Non-plastic synapse, no update
    }

    Weight delta = calculateSTDPDelta(postFired - preFired);
    weight += delta;
    clampWeight();
}

void Synapse::decayEligibility() {
    // Decay the eligibility trace toward zero
    // This creates a temporal window for reward association
    if (eligibilityTrace > 0) {
        eligibilityTrace -= ELIGIBILITY_DECAY;
        if (eligibilityTrace < 0) {
            eligibilityTrace = 0;
        }
    }
}

void Synapse::applyReward(int rewardAmount) {
    // The Dopamine Flood: Apply reward to eligible synapses
    // Only synapses with active eligibility traces get modified
    //
    // This is the "Bitterbot Secret Sauce":
    // - Action at T=10 sets eligibilityTrace = 100
    // - Reward at T=50 finds eligibilityTrace = 60 (decayed)
    // - weight += (60 * reward) / SCALE_FACTOR
    //
    // Memory cost: 1 byte per synapse
    // Compute cost: 1 integer multiply + divide

    if (!isHebbian || eligibilityTrace <= 0) {
        return;  // Not eligible or not plastic
    }

    // Calculate weight change: trace * reward / scale
    int32_t delta = (static_cast<int32_t>(eligibilityTrace) * rewardAmount) / REWARD_SCALE_FACTOR;

    // Apply the change
    weight += static_cast<Weight>(std::clamp(delta, -16, 16));
    clampWeight();

    // Clear the eligibility trace after reward is applied
    // (prevents double-dipping on the same action)
    eligibilityTrace = 0;
}

Weight calculateSTDPDelta(Tick deltaT) {
    // STDP curve implementation using integer arithmetic
    // Blueprint specifies Hebbian learning with timing-dependent plasticity
    //
    // deltaT = t_post - t_pre
    //   deltaT > 0: pre before post -> LTP (potentiation, strengthen)
    //   deltaT < 0: post before pre -> LTD (depression, weaken)
    //   deltaT = 0: simultaneous -> no change
    //
    // Using discrete approximation of exponential decay:
    // LTP: +A * (1 - |deltaT| / tau) for deltaT in [1, tau]
    // LTD: -A * (1 - |deltaT| / tau) for deltaT in [-tau, -1]

    if (deltaT == 0) {
        return 0;  // No change for simultaneous firing
    }

    // Check if outside STDP window
    Tick absDelta = std::abs(deltaT);
    if (absDelta > STDP_WINDOW) {
        return 0;  // Outside plasticity window
    }

    // Integer approximation of exponential decay
    // Scale factor: max change of +/-2 for closest timing
    // Linear decay to 0 at STDP_WINDOW boundary
    //
    // weight_delta = sign(deltaT) * 2 * (STDP_WINDOW - absDelta) / STDP_WINDOW
    // Using integer arithmetic: multiply before divide to preserve precision

    int32_t magnitude = (2 * (STDP_WINDOW - absDelta)) / STDP_WINDOW;

    if (deltaT > 0) {
        // LTP: pre-then-post, strengthen connection
        return static_cast<Weight>(magnitude);
    } else {
        // LTD: post-then-pre, weaken connection
        return static_cast<Weight>(-magnitude);
    }
}

}  // namespace bpagi
