#include "bpagi/network.hpp"
#include <algorithm>
#include <mutex>

// OpenMP support
#ifdef _OPENMP
#include <omp.h>
#endif

namespace bpagi {

// ========================================
// Thread-Safe Helpers
// ========================================

// Atomic add for Charge (int32_t) - works on all platforms
inline void atomicAddCharge(Charge& target, Weight amount) {
    #ifdef _OPENMP
    // Use GCC/Clang atomic builtin for thread-safe addition
    __atomic_fetch_add(&target, static_cast<Charge>(amount), __ATOMIC_RELAXED);
    #else
    // Single-threaded: direct add
    target += amount;
    #endif
}

Network::Network(size_t numNeurons, size_t maxSynapses)
    : currentTick_(0)
    , plasticityEnabled_(true)
    , operantMode_(false)  // Default to Pavlovian (immediate STDP)
    , razorEnabled_(Config::ENABLE_RAZOR)  // Phase 17: k-WTA Razor
    , maxSpikesPerTick_(Config::MAX_SPIKES_PER_TICK)
    , lastCandidateCount_(0)
    , spikeQueue_(numNeurons)
{
    neurons_.reserve(numNeurons);
    synapses_.reserve(maxSynapses);
    // Allocate refractory bitset: 1 bit per neuron, rounded up to 64-bit words
    refractoryBits_.resize((numNeurons + 63) / 64, 0);
}

void Network::step() {
    // Blueprint Section 1.2: The "Clock" Cycle
    // Each tick executes four phases in sequence

    // Save last tick's fired neurons for plasticity
    firedLastTick_ = std::move(firedThisTick_);
    firedThisTick_.clear();

    // Advance spike queue to new tick
    spikeQueue_.advanceTick(currentTick_);

    // OPTIMIZATION: Update refractory bitset ONCE at start of tick
    // This allows O(1) refractory checks without fetching neuron structs
    updateRefractoryBits();

    // 1. Leakage Phase: For every active neuron, V_m = V_m - L
    leakagePhase();

    // 2. Integration Phase: Process spike queue from t-1
    integrationPhase();

    // 3. Firing Phase: Check thresholds, emit spikes
    firingPhase();

    // 4. Plasticity Phase: STDP updates
    if (plasticityEnabled_) {
        plasticityPhase();
    }

    // 5. Eligibility Decay (operant mode only)
    // This happens AFTER plasticity so new traces don't immediately decay
    if (operantMode_) {
        decayEligibilityTraces();
    }

    // 6. Chemical Homeostasis - all neuromodulators decay toward baseline
    chemicals_.decay();

    // 7. PHASE 20: Emergency interrupt on extreme arousal
    // If NE spikes to 95+, trigger the "startle response" and clear working memory
    // This prevents the network from getting stuck in runaway activation
    if (chemicals_.norepinephrine >= 95) {
        panicReset();
    }

    currentTick_++;
}

void Network::run(size_t steps) {
    for (size_t i = 0; i < steps; i++) {
        step();
    }
}

void Network::reset() {
    currentTick_ = 0;
    firedThisTick_.clear();
    firedLastTick_.clear();
    spikeQueue_.clear();

    for (auto& neuron : neurons_) {
        neuron.reset();
    }
}

NeuronId Network::addNeuron(Charge threshold, Charge leak, int32_t refractory) {
    NeuronId id = static_cast<NeuronId>(neurons_.size());
    neurons_.emplace_back(threshold, leak, refractory);

    // Ensure refractory bitset has enough capacity
    size_t requiredWords = (neurons_.size() + 63) / 64;
    if (refractoryBits_.size() < requiredWords) {
        refractoryBits_.resize(requiredWords, 0);
    }

    return id;
}

bool Network::connectNeurons(NeuronId from, NeuronId to, Weight weight, bool plastic) {
    // Validate neuron IDs
    if (from >= neurons_.size() || to >= neurons_.size()) {
        return false;
    }

    // Get the source neuron
    Neuron& sourceNeuron = neurons_[from];

    // Check if we can add to the contiguous synapse array
    // Only possible if: no synapses yet, OR this neuron's synapses are at the end
    bool canAddContiguous = (sourceNeuron.synapseCount == 0) ||
        (sourceNeuron.synapseListIndex + sourceNeuron.synapseCount == synapses_.size());

    if (canAddContiguous) {
        // If this is the first synapse for this neuron, record the index
        if (sourceNeuron.synapseCount == 0) {
            sourceNeuron.synapseListIndex = static_cast<uint32_t>(synapses_.size());
        }
        // Add to contiguous array
        synapses_.emplace_back(to, weight, plastic);
        sourceNeuron.synapseCount++;
    } else {
        // Add to dynamic storage (non-contiguous)
        dynamicSynapses_[from].emplace_back(to, weight, plastic);
    }

    return true;
}

void Network::injectSpike(NeuronId neuron) {
    if (neuron < neurons_.size()) {
        // Add spike to queue for immediate processing
        spikeQueue_.addSpike(neuron, currentTick_);
        firedThisTick_.insert(neuron);
        neurons_[neuron].lastFiredStep = currentTick_;
    }
}

void Network::injectCharge(NeuronId neuron, Charge amount) {
    if (neuron < neurons_.size()) {
        neurons_[neuron].addCharge(amount);
    }
}

bool Network::didFire(NeuronId neuron) const {
    return firedThisTick_.count(neuron) > 0;
}

Charge Network::getCharge(NeuronId neuron) const {
    if (neuron < neurons_.size()) {
        return neurons_[neuron].currentCharge;
    }
    return 0;
}

// ========================================
// Phase Implementations
// ========================================

void Network::leakagePhase() {
    // Blueprint: "For every active neuron, V_m = V_m - L"
    // Active means not in refractory period
    //
    // SEROTONIN EFFECT (The "Chill Pill"):
    // High 5-HT increases effective leak rate, making neurons harder to fire.
    // This implements "patience" - the system becomes more stable.
    // effectiveLeak = baseLeak + (serotonin / 10)
    //
    // PARALLELIZATION: Enabled for networks > 100K neurons (honeybee scale)
    // Each neuron's leak is independent, making this embarrassingly parallel.

    const int8_t serotoninBonus = chemicals_.serotonin / 10;  // 0-10 extra leak
    const Tick tick = currentTick_;
    const size_t numNeurons = neurons_.size();

    #ifdef _OPENMP
    // Enable parallel processing for large networks
    if (numNeurons >= 100000) {
        #pragma omp parallel for schedule(static, 4096)
        for (size_t i = 0; i < numNeurons; i++) {
            Neuron& neuron = neurons_[i];
            if (!neuron.isRefractory(tick)) {
                neuron.applyLeak();
                if (serotoninBonus > 0 && neuron.currentCharge > 0) {
                    neuron.currentCharge -= serotoninBonus;
                    if (neuron.currentCharge < 0) {
                        neuron.currentCharge = 0;
                    }
                }
            }
        }
        return;
    }
    #endif

    // Sequential path for smaller networks
    for (auto& neuron : neurons_) {
        if (!neuron.isRefractory(tick)) {
            neuron.applyLeak();
            if (serotoninBonus > 0 && neuron.currentCharge > 0) {
                neuron.currentCharge -= serotoninBonus;
                if (neuron.currentCharge < 0) {
                    neuron.currentCharge = 0;
                }
            }
        }
    }
}

void Network::integrationPhase() {
    // Process spikes from the previous tick
    // Blueprint: "Process spike queue from t-1"
    //
    // NOTE: Integration phase is kept SEQUENTIAL because:
    // 1. Atomic operations for charge updates have high cache contention
    // 2. The spike-to-synapse ratio is typically low, limiting parallelism benefit
    // 3. Sequential code has better cache locality for sparse networks
    //
    // The main parallelization benefit comes from leakage/firing phases which
    // operate on the full neuron array without inter-neuron dependencies.

    std::vector<NeuronId> spikesToProcess = spikeQueue_.getSpikesForTick(currentTick_ - 1);
    const size_t numNeurons = neurons_.size();

    for (NeuronId firedNeuron : spikesToProcess) {
        if (firedNeuron >= numNeurons) continue;

        const Neuron& pre = neurons_[firedNeuron];

        // Process contiguous synapses
        for (uint16_t i = 0; i < pre.synapseCount; i++) {
            const Synapse& syn = synapses_[pre.synapseListIndex + i];
            NeuronId targetId = syn.targetNeuronIndex;

            if (targetId < numNeurons && !isRefractoryFast(targetId)) {
                neurons_[targetId].currentCharge += syn.weight;
            }
        }

        // Process dynamic synapses
        auto dynIt = dynamicSynapses_.find(firedNeuron);
        if (dynIt != dynamicSynapses_.end()) {
            for (const Synapse& syn : dynIt->second) {
                NeuronId targetId = syn.targetNeuronIndex;
                if (targetId < numNeurons && !isRefractoryFast(targetId)) {
                    neurons_[targetId].currentCharge += syn.weight;
                }
            }
        }
    }
}

void Network::updateRefractoryBits() {
    // Clear all bits
    std::fill(refractoryBits_.begin(), refractoryBits_.end(), 0ULL);

    // Set bit for each refractory neuron
    for (NeuronId id = 0; id < neurons_.size(); id++) {
        if (neurons_[id].isRefractory(currentTick_)) {
            refractoryBits_[id >> 6] |= (1ULL << (id & 63));
        }
    }
}

void Network::firingPhase() {
    // Check each neuron for threshold crossing
    // Blueprint: "Check thresholds, emit spikes"
    //
    // ===========================================
    // THE "RAZOR" (k-WTA Lateral Inhibition) - Phase 17
    // ===========================================
    // Instead of letting every above-threshold neuron fire, we collect
    // all candidates, sort by charge, and only allow the top K to spike.
    // This simulates biological lateral inhibition and enforces sparsity.
    //
    // NOREPINEPHRINE EFFECT (The "Panic Button"):
    // High NE lowers effective threshold, making neurons "trigger happy".
    // effectiveThreshold = baseThreshold - (norepinephrine / 5)
    //
    // PHASE 20 ENHANCEMENT: Simulated Annealing via NE Noise
    // When NE > 60, add random noise to threshold decisions.
    // This helps the system escape local minima and explore alternatives.
    // Higher NE = more noise = more exploration (useful for hard problems)

    const int8_t thresholdReduction = chemicals_.norepinephrine / 5;
    const Tick tick = currentTick_;
    const size_t numNeurons = neurons_.size();

    // Calculate noise amplitude based on NE level
    // NE 0-60: no noise (exploitation mode)
    // NE 60-100: 0-10 noise amplitude (exploration mode)
    const int8_t noiseAmplitude = (chemicals_.norepinephrine > 60)
        ? (chemicals_.norepinephrine - 60) / 4  // 0-10 range
        : 0;

    // Use tick-based seed for reproducible noise (different each tick)
    uint32_t noiseSeed = static_cast<uint32_t>(tick * 1103515245 + 12345);

    // ===========================================
    // STEP 1: Collect all candidates that WANT to fire
    // ===========================================
    // Candidate = (charge, neuronId) for sorting by charge
    std::vector<std::pair<Charge, NeuronId>> candidates;
    candidates.reserve(std::min(numNeurons / 10, maxSpikesPerTick_ * 2));

    #ifdef _OPENMP
    // Parallel path for large networks
    if (numNeurons >= 100000) {
        std::vector<std::vector<std::pair<Charge, NeuronId>>> threadCandidates(omp_get_max_threads());

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto& localCandidates = threadCandidates[tid];
            localCandidates.reserve(maxSpikesPerTick_ * 2 / omp_get_num_threads());

            #pragma omp for schedule(static, 4096)
            for (size_t id = 0; id < numNeurons; id++) {
                Neuron& n = neurons_[id];

                if (n.isRefractory(tick)) continue;

                Charge effectiveThreshold = n.threshold - thresholdReduction;

                // PHASE 20: Add NE-driven noise for exploration (simulated annealing)
                if (noiseAmplitude > 0) {
                    // Thread-safe noise using neuron ID for per-neuron randomness
                    uint32_t localSeed = noiseSeed ^ static_cast<uint32_t>(id);
                    localSeed = localSeed * 1103515245 + 12345;
                    int noise = static_cast<int>((localSeed >> 16) & 0xFF) % (2 * noiseAmplitude + 1) - noiseAmplitude;
                    effectiveThreshold += noise;
                }

                if (effectiveThreshold < 1) effectiveThreshold = 1;

                if (n.currentCharge >= effectiveThreshold) {
                    localCandidates.emplace_back(n.currentCharge, static_cast<NeuronId>(id));
                }
            }
        }

        // Merge thread-local candidates
        for (const auto& local : threadCandidates) {
            candidates.insert(candidates.end(), local.begin(), local.end());
        }
    } else
    #endif
    {
        // Sequential path for smaller networks
        for (NeuronId id = 0; id < numNeurons; id++) {
            Neuron& n = neurons_[id];

            if (n.isRefractory(tick)) continue;

            Charge effectiveThreshold = n.threshold - thresholdReduction;

            // PHASE 20: Add NE-driven noise for exploration (simulated annealing)
            if (noiseAmplitude > 0) {
                noiseSeed = noiseSeed * 1103515245 + 12345;
                int noise = static_cast<int>((noiseSeed >> 16) & 0xFF) % (2 * noiseAmplitude + 1) - noiseAmplitude;
                effectiveThreshold += noise;
            }

            if (effectiveThreshold < 1) effectiveThreshold = 1;

            if (n.currentCharge >= effectiveThreshold) {
                candidates.emplace_back(n.currentCharge, id);
            }
        }
    }

    // Store candidate count for diagnostics
    lastCandidateCount_ = candidates.size();

    // ===========================================
    // STEP 2: THE RAZOR - k-Winner-Take-All Selection
    // ===========================================
    // If more candidates than allowed, keep only the top K by charge.
    // Uses nth_element for O(n) selection instead of O(n log n) full sort.

    if (razorEnabled_ && candidates.size() > maxSpikesPerTick_) {
        // Efficiently find the top K (no need to sort the whole list)
        std::nth_element(
            candidates.begin(),
            candidates.begin() + maxSpikesPerTick_,
            candidates.end(),
            [](const auto& a, const auto& b) {
                return a.first > b.first;  // Descending by charge
            }
        );

        // Resize to keep only the winners - the rest are silenced!
        candidates.resize(maxSpikesPerTick_);
    }

    // ===========================================
    // STEP 3: Fire the winners (and silence the losers)
    // ===========================================
    for (const auto& winner : candidates) {
        NeuronId id = winner.second;
        Neuron& n = neurons_[id];

        // Fire!
        n.currentCharge = 0;
        n.lastFiredStep = tick;
        spikeQueue_.addSpike(id, tick);
        firedThisTick_.insert(id);
    }

    // NOTE: Neurons that were candidates but didn't make the cut
    // keep their charge. This creates temporal integration -
    // they may fire on the next tick if still above threshold.
}

void Network::plasticityPhase() {
    // STDP: Update weights based on spike timing
    // Blueprint: "If post fires shortly after pre, strengthen connection"
    //
    // OPTIMIZED VERSION: Forward-lookup only (O(Spikes × Synapses) not O(Spikes × Neurons))
    //
    // DOPAMINE EFFECT (The "Save Button"):
    // DA gates whether learning occurs at all.
    // If DA < 10: Learning is DISABLED (freeze weights)
    // If DA >= 10: Learning strength = delta * (DA / 50)

    // Check if dopamine is too low for learning
    if (chemicals_.dopamine < 10) {
        return;  // No learning when DA is depleted
    }

    // Helper lambda to apply STDP to a synapse
    auto applySTDP = [this](Synapse& syn, NeuronId preId, NeuronId postId) {
        if (!syn.isHebbian) return;

        Tick preFired = neurons_[preId].lastFiredStep;
        Tick postFired = neurons_[postId].lastFiredStep;

        if (operantMode_) {
            // Operant: Mark as eligible, don't update weight yet
            syn.markEligible(preFired, postFired);
        } else {
            // Pavlovian: Immediate weight update
            syn.updateWeight(preFired, postFired);
        }
    };

    // =================================================================
    // LTP (Long-Term Potentiation): Pre fired recently -> Post fires NOW
    // =================================================================
    // OPTIMIZATION: Instead of scanning ALL neurons to find who connects to Post,
    // we iterate Pre-neurons that fired LAST tick and check if their TARGETS fired THIS tick.
    // Complexity: O(LastSpikes × AvgSynapses) instead of O(ThisSpikes × AllNeurons)

    for (NeuronId preId : firedLastTick_) {
        Neuron& pre = neurons_[preId];

        // Check contiguous synapses from this pre-neuron
        for (uint16_t i = 0; i < pre.synapseCount; i++) {
            Synapse& syn = synapses_[pre.synapseListIndex + i];

            // O(1) lookup: Did this synapse's target fire THIS tick?
            if (firedThisTick_.count(syn.targetNeuronIndex)) {
                // Pre fired last tick, Post fired this tick -> LTP
                applySTDP(syn, preId, syn.targetNeuronIndex);
            }
        }

        // Check dynamic synapses
        auto dynIt = dynamicSynapses_.find(preId);
        if (dynIt != dynamicSynapses_.end()) {
            for (Synapse& syn : dynIt->second) {
                if (firedThisTick_.count(syn.targetNeuronIndex)) {
                    applySTDP(syn, preId, syn.targetNeuronIndex);
                }
            }
        }
    }

    // =================================================================
    // LTD (Long-Term Depression): Post fired recently -> Pre fires NOW
    // =================================================================
    // Pavlovian mode only - in operant mode, LTD happens via negative reward

    if (!operantMode_) {
        for (NeuronId preId : firedThisTick_) {
            Neuron& pre = neurons_[preId];

            // Check contiguous synapses
            for (uint16_t i = 0; i < pre.synapseCount; i++) {
                Synapse& syn = synapses_[pre.synapseListIndex + i];

                // Did target fire LAST tick? (Post before Pre = LTD)
                if (syn.isHebbian && firedLastTick_.count(syn.targetNeuronIndex)) {
                    Tick preFired = pre.lastFiredStep;
                    Tick postFired = neurons_[syn.targetNeuronIndex].lastFiredStep;
                    Tick deltaT = postFired - preFired;
                    if (deltaT < 0 && std::abs(deltaT) <= STDP_WINDOW) {
                        syn.updateWeight(preFired, postFired);
                    }
                }
            }

            // Check dynamic synapses
            auto dynIt = dynamicSynapses_.find(preId);
            if (dynIt != dynamicSynapses_.end()) {
                for (Synapse& syn : dynIt->second) {
                    if (syn.isHebbian && firedLastTick_.count(syn.targetNeuronIndex)) {
                        Tick preFired = pre.lastFiredStep;
                        Tick postFired = neurons_[syn.targetNeuronIndex].lastFiredStep;
                        Tick deltaT = postFired - preFired;
                        if (deltaT < 0 && std::abs(deltaT) <= STDP_WINDOW) {
                            syn.updateWeight(preFired, postFired);
                        }
                    }
                }
            }
        }
    }
}

void Network::decayEligibilityTraces() {
    // Decay all eligibility traces by 1
    // This creates a temporal credit assignment window

    // Decay contiguous synapses
    for (Synapse& syn : synapses_) {
        syn.decayEligibility();
    }

    // Decay dynamic synapses
    for (auto& pair : dynamicSynapses_) {
        for (Synapse& syn : pair.second) {
            syn.decayEligibility();
        }
    }
}

void Network::injectReward(int amount) {
    // The "Dopamine Flood"
    // Iterate through ALL plastic synapses and apply reward
    // based on their eligibility traces.
    //
    // This is the core of operant conditioning:
    // - Action at T=10 sets eligibility trace
    // - Reward at T=50 finds decayed trace
    // - Weight updated proportional to trace * reward

    // Apply to contiguous synapses
    for (Synapse& syn : synapses_) {
        syn.applyReward(amount);
    }

    // Apply to dynamic synapses
    for (auto& pair : dynamicSynapses_) {
        for (Synapse& syn : pair.second) {
            syn.applyReward(amount);
        }
    }
}

Weight Network::getSynapseWeight(NeuronId source, NeuronId target) const {
    if (source >= neurons_.size()) return 0;

    const Neuron& pre = neurons_[source];

    // Check contiguous synapses
    for (uint16_t i = 0; i < pre.synapseCount; i++) {
        const Synapse& syn = synapses_[pre.synapseListIndex + i];
        if (syn.targetNeuronIndex == target) {
            return syn.weight;
        }
    }

    // Check dynamic synapses
    auto it = dynamicSynapses_.find(source);
    if (it != dynamicSynapses_.end()) {
        for (const Synapse& syn : it->second) {
            if (syn.targetNeuronIndex == target) {
                return syn.weight;
            }
        }
    }

    return 0;  // No synapse found
}

// ========================================
// Neuromodulation Convenience Methods
// ========================================

void Network::rewardSignal(int8_t amount) {
    // The "Save Button" - spike dopamine to enable/boost learning
    // Called when something good happens (Pong hit, goal achieved)
    chemicals_.spikeDopamine(amount);

    // Also apply reward to eligible synapses (operant mode)
    if (operantMode_) {
        injectReward(amount / 10);  // Scale down for eligibility-based update
    }
}

void Network::surpriseSignal(int8_t amount) {
    // The "Panic Button" - spike norepinephrine to lower thresholds
    // Called when something unexpected happens (novel stimulus, error)
    chemicals_.spikeNorepinephrine(amount);
}

void Network::calmSignal(int8_t amount) {
    // The "Chill Pill" - spike serotonin to increase stability
    // Called during inactivity or when system needs to settle
    chemicals_.spikeSerotonin(amount);
}

// ========================================
// Emergency Reset (Phase 20)
// ========================================

void Network::panicReset() {
    // The "Interrupt" - clears all working memory on extreme surprise
    // This simulates the biological startle response where attention
    // is immediately cleared to deal with a potential threat.
    //
    // Called when NE spikes above 95 - a critical arousal level
    // indicating overwhelming surprise or danger.

    // Clear all membrane potentials (working memory)
    for (auto& neuron : neurons_) {
        neuron.currentCharge = 0;
    }

    // Clear spike queue (pending activations)
    spikeQueue_.clear();

    // Clear firing records
    firedThisTick_.clear();
    firedLastTick_.clear();

    // Bring NE back down but stay alert
    // 70 = still highly aroused, but not in panic territory
    chemicals_.norepinephrine = 70;
}

}  // namespace bpagi
