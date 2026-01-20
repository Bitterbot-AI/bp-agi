#pragma once
#include "neuron.hpp"
#include "synapse.hpp"
#include "spike_queue.hpp"
#include "config.hpp"
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <cstdint>
#include <atomic>

// OpenMP support (optional - gracefully degrades to single-threaded if not available)
#ifdef _OPENMP
#include <omp.h>
#endif

namespace bpagi {

// Main network container implementing the Blueprint Section 1.2 simulation loop
// Manages neurons, synapses, and spike propagation with event-driven timing
class Network {
public:
    // Constructor with initial capacity hints
    Network(size_t numNeurons = 0, size_t maxSynapses = 0);
    ~Network() = default;

    // Non-copyable but moveable
    Network(const Network&) = delete;
    Network& operator=(const Network&) = delete;
    Network(Network&&) = default;
    Network& operator=(Network&&) = default;

    // ========================================
    // Main Simulation Interface
    // ========================================

    // The main simulation step (Blueprint Section 1.2: "Clock" Cycle)
    // Executes: Leakage -> Integration -> Firing -> Plasticity
    void step();

    // Run multiple steps
    void run(size_t steps);

    // Reset network to initial state
    void reset();

    // ========================================
    // Network Construction
    // ========================================

    // Add a neuron with specified parameters
    // Returns the neuron ID
    NeuronId addNeuron(Charge threshold = DEFAULT_THRESHOLD,
                       Charge leak = DEFAULT_LEAK,
                       int32_t refractory = DEFAULT_REFRACTORY);

    // Connect two neurons with a synapse
    // Returns true if connection was successful
    bool connectNeurons(NeuronId from, NeuronId to,
                        Weight weight, bool plastic = true);

    // ========================================
    // External Input
    // ========================================

    // Inject a spike into a neuron (external stimulus)
    void injectSpike(NeuronId neuron);

    // Inject charge directly into a neuron
    void injectCharge(NeuronId neuron, Charge amount);

    // ========================================
    // State Queries
    // ========================================

    // Get current simulation tick
    Tick getCurrentTick() const { return currentTick_; }

    // Get number of neurons
    size_t getNeuronCount() const { return neurons_.size(); }

    // Get number of synapses (including dynamic)
    size_t getSynapseCount() const {
        size_t count = synapses_.size();
        for (const auto& pair : dynamicSynapses_) {
            count += pair.second.size();
        }
        return count;
    }

    // Check if a neuron fired in the current tick
    bool didFire(NeuronId neuron) const;

    // Get the current charge of a neuron
    Charge getCharge(NeuronId neuron) const;

    // Get neuron reference (for advanced access)
    const Neuron& getNeuron(NeuronId id) const { return neurons_[id]; }

    // Get synapse reference
    const Synapse& getSynapse(size_t index) const { return synapses_[index]; }

    // Get synapse count for a specific neuron (contiguous + dynamic)
    size_t getNeuronSynapseCount(NeuronId neuron) const {
        if (neuron >= neurons_.size()) return 0;
        size_t count = neurons_[neuron].synapseCount;
        auto it = dynamicSynapses_.find(neuron);
        if (it != dynamicSynapses_.end()) {
            count += it->second.size();
        }
        return count;
    }

    // Get the dynamic synapses map (for debugging)
    const std::unordered_map<NeuronId, std::vector<Synapse>>& getDynamicSynapses() const {
        return dynamicSynapses_;
    }

    // Get list of neurons that fired in last step
    const std::unordered_set<NeuronId>& getFiredNeurons() const { return firedThisTick_; }

    // Get the weight of a synapse from source to target
    // Returns 0 if no such synapse exists
    Weight getSynapseWeight(NeuronId source, NeuronId target) const;

    // ========================================
    // Configuration
    // ========================================

    // Enable/disable plasticity globally
    void setPlasticityEnabled(bool enabled) { plasticityEnabled_ = enabled; }
    bool isPlasticityEnabled() const { return plasticityEnabled_; }

    // Set operant conditioning mode
    // When true: uses eligibility traces + reward signals (operant)
    // When false: uses immediate STDP updates (Pavlovian)
    void setOperantMode(bool enabled) { operantMode_ = enabled; }
    bool isOperantMode() const { return operantMode_; }

    // ===========================================
    // k-WTA "RAZOR" Configuration (Phase 17)
    // ===========================================
    // The Razor enforces biological sparsity by limiting how many
    // neurons can fire per tick. Only the top-K strongest activations
    // are allowed to spike; all others are silenced.
    //
    // This simulates lateral inhibition in biological cortex.

    // Enable/disable the k-WTA Razor
    void setRazorEnabled(bool enabled) { razorEnabled_ = enabled; }
    bool isRazorEnabled() const { return razorEnabled_; }

    // Set the maximum spikes per tick (k in k-WTA)
    void setMaxSpikesPerTick(size_t k) { maxSpikesPerTick_ = k; }
    size_t getMaxSpikesPerTick() const { return maxSpikesPerTick_; }

    // Get statistics from last tick
    size_t getLastCandidateCount() const { return lastCandidateCount_; }
    size_t getLastSpikeCount() const { return firedThisTick_.size(); }

    // ========================================
    // Reward System (Operant Conditioning)
    // ========================================

    // Inject a global reward signal (The "Dopamine Flood")
    // Positive values strengthen eligible synapses (reward)
    // Negative values weaken eligible synapses (punishment)
    void injectReward(int amount);

    // ========================================
    // Neuromodulation System (The Chemical Layer)
    // ========================================

    // Get current neuromodulator levels (read-only)
    const Neuromodulators& getChemicals() const { return chemicals_; }

    // Get mutable access for external regulation (e.g., UKS homeostatic loop)
    Neuromodulators& chemicals() { return chemicals_; }

    // Convenience methods for common chemical events
    void rewardSignal(int8_t amount = 50);    // Spike DA (learning moment!)
    void surpriseSignal(int8_t amount = 50);  // Spike NE (wake up!)
    void calmSignal(int8_t amount = 10);      // Spike 5-HT (chill out)

    // ========================================
    // Emergency Reset (Phase 20)
    // ========================================

    // Panic reset - "The Interrupt"
    // Clears all working memory when NE spikes extremely high
    // Simulates the biological "startle response" that clears attention
    void panicReset();

private:
    // Simulation state
    Tick currentTick_;
    bool plasticityEnabled_;
    bool operantMode_;  // true = eligibility traces, false = immediate STDP

    // k-WTA "Razor" state (Phase 17)
    bool razorEnabled_;
    size_t maxSpikesPerTick_;
    size_t lastCandidateCount_;  // For diagnostics

    // Neuromodulation state (The Chemical Layer)
    Neuromodulators chemicals_;

    // Network structure
    std::vector<Neuron> neurons_;
    std::vector<Synapse> synapses_;

    // Dynamic synapse storage for neurons with non-contiguous synapses
    // Maps neuron ID to vector of additional synapses
    std::unordered_map<NeuronId, std::vector<Synapse>> dynamicSynapses_;

    // Event queue for spike propagation
    SpikeQueue spikeQueue_;

    // Track which neurons fired this tick (for queries and plasticity)
    std::unordered_set<NeuronId> firedThisTick_;
    std::unordered_set<NeuronId> firedLastTick_;

    // OPTIMIZATION: Refractory bitset for O(1) lookup
    // Avoids fetching full Neuron struct just to check refractory status
    // Each uint64_t covers 64 neurons; bit=1 means refractory
    std::vector<uint64_t> refractoryBits_;

    // Update refractory bitset at start of each tick
    void updateRefractoryBits();

    // Fast O(1) refractory check using bitset
    inline bool isRefractoryFast(NeuronId id) const {
        return (refractoryBits_[id >> 6] & (1ULL << (id & 63))) != 0;
    }

    // ========================================
    // Phase Implementations (Blueprint 1.2)
    // ========================================

    // Phase 1: Leakage - For every active neuron, V_m = V_m - L
    void leakagePhase();

    // Phase 2: Integration - Process spike queue from t-1
    void integrationPhase();

    // Phase 3: Firing - Check thresholds, emit spikes
    void firingPhase();

    // Phase 4: Plasticity - STDP weight updates
    void plasticityPhase();

    // Decay all eligibility traces (called every tick in operant mode)
    void decayEligibilityTraces();
};

}  // namespace bpagi
