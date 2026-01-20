#pragma once
#include "types.hpp"
#include <vector>
#include <queue>

namespace bpagi {

// Spike event for the event-driven simulation
struct SpikeEvent {
    NeuronId neuronId;  // Which neuron fired
    Tick tick;          // When it fired

    SpikeEvent() : neuronId(INVALID_NEURON), tick(0) {}
    SpikeEvent(NeuronId id, Tick t) : neuronId(id), tick(t) {}

    // For priority queue ordering (earliest first)
    bool operator>(const SpikeEvent& other) const {
        return tick > other.tick;
    }
};

// Event-driven spike queue for efficient spike propagation
// Uses double-buffering to separate current and next tick events
class SpikeQueue {
public:
    SpikeQueue() : currentTick_(0) {}

    explicit SpikeQueue(size_t reserveSize) : currentTick_(0) {
        currentFired_.reserve(reserveSize);
    }

    // Add a spike event at the given tick
    inline void addSpike(NeuronId neuron, Tick tick) {
        queue_.emplace(neuron, tick);
        if (tick == currentTick_) {
            currentFired_.push_back(neuron);
        }
    }

    // Check if there are spikes for the given tick
    inline bool hasSpikesForTick(Tick tick) const {
        if (queue_.empty()) return false;
        return queue_.top().tick == tick;
    }

    // Get all spikes that occurred at the given tick
    inline std::vector<NeuronId> getSpikesForTick(Tick tick) {
        std::vector<NeuronId> spikes;
        while (!queue_.empty() && queue_.top().tick == tick) {
            spikes.push_back(queue_.top().neuronId);
            queue_.pop();
        }
        return spikes;
    }

    // Pop the next spike for processing (from earliest tick)
    inline NeuronId popSpike() {
        if (queue_.empty()) return INVALID_NEURON;
        NeuronId id = queue_.top().neuronId;
        queue_.pop();
        return id;
    }

    // Get the tick of the next spike without removing it
    inline Tick peekNextTick() const {
        if (queue_.empty()) return -1;
        return queue_.top().tick;
    }

    // Check if queue is empty
    inline bool empty() const { return queue_.empty(); }

    // Clear all spikes
    inline void clear() {
        std::priority_queue<SpikeEvent, std::vector<SpikeEvent>, std::greater<SpikeEvent>> empty;
        std::swap(queue_, empty);
        currentFired_.clear();
        currentTick_ = 0;
    }

    // Get current queue size
    inline size_t size() const { return queue_.size(); }

    // Get spikes fired at current tick
    const std::vector<NeuronId>& getCurrentFired() const { return currentFired_; }

    // Mark the start of a new tick
    inline void advanceTick(Tick newTick) {
        currentTick_ = newTick;
        currentFired_.clear();
    }

private:
    std::priority_queue<SpikeEvent, std::vector<SpikeEvent>, std::greater<SpikeEvent>> queue_;
    std::vector<NeuronId> currentFired_;
    Tick currentTick_;
};

}  // namespace bpagi
