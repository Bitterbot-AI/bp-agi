// UKS Raster Plot: Shows sparse, controlled activation with Winner-Take-All
// Contrast this with the "epileptic" raw network raster

#include "bpagi/network.hpp"
#include "bpagi/uks.hpp"
#include <iostream>
#include <random>

using namespace bpagi;

int main() {
    const size_t NUM_TICKS = 500;

    // Create network and UKS
    Network net(50000, 500000);

    UKS::Config config;
    config.numColumns = 50;    // 50 cortical columns
    config.busWidth = 64;      // 64-dimensional input space
    config.enableLearning = true;

    UKS uks(net, config);

    // Define 5 distinct input patterns (concepts to learn)
    std::vector<std::vector<NeuronId>> patterns = {
        {0, 1, 2, 3, 4, 5, 6, 7},       // Pattern A
        {10, 11, 12, 13, 14, 15, 16},   // Pattern B
        {20, 21, 22, 23, 24, 25},       // Pattern C
        {30, 31, 32, 33, 34, 35, 36},   // Pattern D
        {40, 41, 42, 43, 44, 45, 46, 47} // Pattern E
    };

    // Output CSV header
    std::cout << "tick,neuron_type,neuron_id" << std::endl;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> patternDist(0, patterns.size() - 1);

    // Run simulation
    for (size_t tick = 0; tick < NUM_TICKS; tick++) {
        // Present a pattern every 20 ticks
        if (tick % 20 == 0) {
            // Cycle through patterns deterministically for first 100 ticks
            // Then random patterns
            int patternIdx;
            if (tick < 100) {
                patternIdx = (tick / 20) % patterns.size();
            } else {
                patternIdx = patternDist(rng);
            }

            uks.present(patterns[patternIdx]);
        }

        net.step();
        uks.step();

        // Output bus activity (input layer)
        for (size_t i = 0; i < uks.getBusNeurons().size(); i++) {
            if (net.didFire(uks.getBusNeurons()[i])) {
                std::cout << tick << ",bus," << i << std::endl;
            }
        }

        // Output column activity (concept layer)
        for (size_t col = 0; col < config.numColumns; col++) {
            const auto& column = uks.getColumn(col);
            if (column.isAllocated && net.didFire(column.outputNeuron)) {
                // Column output = concept recognized
                std::cout << tick << ",concept," << col << std::endl;
            }
        }

        // Output Request neuron (novelty detector)
        if (uks.didRequestFire()) {
            std::cout << tick << ",request,0" << std::endl;
        }
    }

    // Summary to stderr
    std::cerr << "=== UKS Raster Complete ===" << std::endl;
    std::cerr << "Columns allocated: " << uks.getAllocatedCount() << std::endl;
    std::cerr << "Total recognitions: " << uks.getTotalRecognitions() << std::endl;

    return 0;
}
