// Spike Recorder: Outputs spike times for raster plot visualization
// Outputs CSV format: tick,neuron_id

#include "bpagi/network.hpp"
#include <iostream>
#include <random>

using namespace bpagi;

int main() {
    const size_t NUM_NEURONS = 200;
    const size_t NUM_TICKS = 1000;
    const double CONNECTIVITY = 0.10;  // 10% sparsity
    const int STIMULUS_INTERVAL = 50;   // Stimulate every 50 ticks

    // Create network
    Network net(NUM_NEURONS, NUM_NEURONS * NUM_NEURONS);

    // Add neurons with parameters tuned for visible wave propagation
    // Low threshold allows activity to spread, no leak preserves charge
    for (size_t i = 0; i < NUM_NEURONS; i++) {
        net.addNeuron(3, 0, 2);  // threshold=3, no leak, refractory=2
    }

    // Random sparse connectivity (10%) with strong excitatory weights
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> connDist(0.0, 1.0);
    std::uniform_int_distribution<int> weightDist(2, 5);  // Strong excitatory weights

    size_t connectionCount = 0;
    for (size_t i = 0; i < NUM_NEURONS; i++) {
        for (size_t j = 0; j < NUM_NEURONS; j++) {
            if (i != j && connDist(rng) < CONNECTIVITY) {
                Weight w = static_cast<Weight>(weightDist(rng));
                net.connectNeurons(static_cast<NeuronId>(i), static_cast<NeuronId>(j), w, false);
                connectionCount++;
            }
        }
    }

    // Output network info to stderr
    std::cerr << "Network: " << NUM_NEURONS << " neurons, " << connectionCount << " synapses" << std::endl;

    net.setPlasticityEnabled(false);

    // Output CSV header
    std::cout << "tick,neuron" << std::endl;

    // Run simulation
    for (size_t tick = 0; tick < NUM_TICKS; tick++) {
        // Stimulate neuron 0 every 50 ticks
        if (tick % STIMULUS_INTERVAL == 0) {
            net.injectSpike(0);
        }

        net.step();

        // Output all spikes this tick
        for (NeuronId id : net.getFiredNeurons()) {
            std::cout << tick << "," << id << std::endl;
        }
    }

    return 0;
}
