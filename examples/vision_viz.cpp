// Vision System Visualization Tool
// Outputs CSV data for Python visualization

#include "bpagi/network.hpp"
#include "bpagi/vision.hpp"
#include <iostream>

using namespace bpagi;

int main() {
    Network net(100000, 1000000);
    VisionSystem vision(net);

    // Create a white rectangle on black background
    std::vector<uint8_t> image(RETINA_SIZE, 0);

    // Draw rectangle at (16, 16) with size 32x32
    for (size_t y = 16; y < 48; y++) {
        for (size_t x = 16; x < 48; x++) {
            image[y * RETINA_WIDTH + x] = 255;
        }
    }

    // Present image
    vision.present(image);

    // Output CSV header
    std::cout << "layer,x,y" << std::endl;

    // Output input image pixels
    for (size_t y = 0; y < RETINA_HEIGHT; y++) {
        for (size_t x = 0; x < RETINA_WIDTH; x++) {
            if (image[y * RETINA_WIDTH + x] > 0) {
                std::cout << "input," << x << "," << y << std::endl;
            }
        }
    }

    // Output retina activations
    auto activeRetina = vision.getActiveRetina();
    for (const auto& [x, y] : activeRetina) {
        std::cout << "retina," << x << "," << y << std::endl;
    }

    // Run simulation for boundary detection
    for (int tick = 0; tick < 5; tick++) {
        net.step();
        vision.step();

        // Track boundaries that fire on any tick
        auto boundaries = vision.getActiveBoundaries();
        for (const auto& [x, y, type] : boundaries) {
            std::cout << "boundary," << x << "," << y << std::endl;
        }
    }

    return 0;
}
