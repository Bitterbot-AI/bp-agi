#include "bpagi/vision.hpp"
#include <cmath>
#include <algorithm>

namespace bpagi {

VisionSystem::VisionSystem(Network& network)
    : network_(network)
{
    // Initialize state vectors - NOW 10x LARGER for color channels
    retinaState_.resize(RETINA_SIZE * NUM_COLORS, false);
    currentImage_.resize(RETINA_SIZE, 0);

    // Build the three-layer hierarchy
    buildRetina();
    buildBoundaryDetectors();
    buildLineIntegrators();
    buildCornerDetectors();
    buildAcuteVertexDetectors();
    buildDimensionSensors();  // Parietal Patch: spatial awareness

    // Wire the layers
    wireRetinaToBoundary();
    wireBoundaryToLine();
    wireBoundaryToCorner();
    wireBoundaryToAcuteVertex();
}

// ========================================
// Layer 1: Color Retina (10 Channels per Pixel)
// ========================================

void VisionSystem::buildRetina() {
    // 10 neurons per pixel - one for each ARC color (0-9)
    // Total: 64x64x10 = 40,960 neurons
    retinaNeurons_.reserve(RETINA_SIZE * NUM_COLORS);

    for (size_t i = 0; i < RETINA_SIZE; i++) {
        for (size_t c = 0; c < NUM_COLORS; c++) {
            // Retina neurons: low threshold (respond to external input)
            NeuronId n = network_.addNeuron(2, 0, 1);
            retinaNeurons_.push_back(n);
        }
    }
}

// ========================================
// Layer 2: Boundary Detectors (V1) - Shape Detection
// ========================================

void VisionSystem::buildBoundaryDetectors() {
    // 4 boundary types per pixel position
    size_t totalBoundary = RETINA_SIZE * NUM_BOUNDARY_TYPES;
    boundaryNeurons_.reserve(totalBoundary);

    for (size_t i = 0; i < totalBoundary; i++) {
        // Boundary neurons: threshold=2 for edge detection
        NeuronId n = network_.addNeuron(2, 0, 2);
        boundaryNeurons_.push_back(n);
    }
}

void VisionSystem::wireRetinaToBoundary() {
    // CORE LOGIC: Boundary neurons detect SHAPE (foreground vs background).
    // They receive input from ALL non-black color channels (colors 1-9).
    // This preserves shape accuracy while the color info flows in parallel.

    // Helper lambda to connect all non-black color channels at a pixel to a boundary neuron
    auto connectPixelToBoundary = [&](size_t px_idx, NeuronId boundaryId, int weight) {
        // Connect colors 1-9 (skip color 0 = black/background)
        for (size_t c = 1; c < NUM_COLORS; c++) {
            NeuronId colorNeuron = retinaNeurons_[px_idx * NUM_COLORS + c];
            network_.connectNeurons(colorNeuron, boundaryId, weight, false);
        }
    };

    for (size_t y = 0; y < RETINA_HEIGHT; y++) {
        for (size_t x = 0; x < RETINA_WIDTH; x++) {
            NeuronId vertBoundary = boundaryNeurons_[boundaryIdx(x, y, BoundaryType::VERTICAL)];
            NeuronId horizBoundary = boundaryNeurons_[boundaryIdx(x, y, BoundaryType::HORIZONTAL)];
            NeuronId diagBoundary = boundaryNeurons_[boundaryIdx(x, y, BoundaryType::DIAGONAL)];
            NeuronId antiDiagBoundary = boundaryNeurons_[boundaryIdx(x, y, BoundaryType::ANTI_DIAGONAL)];

            // VERTICAL boundary (|): fires if left differs from right
            if (x > 0 && x < RETINA_WIDTH - 1) {
                connectPixelToBoundary(idx(x, y), vertBoundary, 4);      // Center excitation
                connectPixelToBoundary(idx(x - 1, y), vertBoundary, -2); // Left inhibition
                connectPixelToBoundary(idx(x + 1, y), vertBoundary, -2); // Right inhibition
            }

            // HORIZONTAL boundary (-): fires if top differs from bottom
            if (y > 0 && y < RETINA_HEIGHT - 1) {
                connectPixelToBoundary(idx(x, y), horizBoundary, 4);
                connectPixelToBoundary(idx(x, y - 1), horizBoundary, -2);
                connectPixelToBoundary(idx(x, y + 1), horizBoundary, -2);
            }

            // DIAGONAL boundary (/): fires on diagonal edge
            if (x > 0 && y > 0 && x < RETINA_WIDTH - 1 && y < RETINA_HEIGHT - 1) {
                connectPixelToBoundary(idx(x, y), diagBoundary, 4);
                connectPixelToBoundary(idx(x - 1, y - 1), diagBoundary, -2);
                connectPixelToBoundary(idx(x + 1, y + 1), diagBoundary, -2);
            }

            // ANTI-DIAGONAL boundary (\): fires on anti-diagonal edge
            if (x > 0 && y > 0 && x < RETINA_WIDTH - 1 && y < RETINA_HEIGHT - 1) {
                connectPixelToBoundary(idx(x, y), antiDiagBoundary, 4);
                connectPixelToBoundary(idx(x + 1, y - 1), antiDiagBoundary, -2);
                connectPixelToBoundary(idx(x - 1, y + 1), antiDiagBoundary, -2);
            }
        }
    }
}

// ========================================
// Layer 3: Line Integrators (V2)
// ========================================

void VisionSystem::buildLineIntegrators() {
    size_t lineWidth = RETINA_WIDTH / LINE_POOL_SIZE;
    size_t lineHeight = RETINA_HEIGHT / LINE_POOL_SIZE;
    size_t totalLines = lineWidth * lineHeight * NUM_BOUNDARY_TYPES * 2;

    lineNeurons_.reserve(totalLines);

    for (size_t i = 0; i < totalLines; i++) {
        NeuronId n = network_.addNeuron(static_cast<Charge>(LINE_MIN_ACTIVE), 0, 3);
        lineNeurons_.push_back(n);
    }
}

void VisionSystem::wireBoundaryToLine() {
    size_t lineIdx = 0;

    // Horizontal line segments
    for (size_t y = 0; y < RETINA_HEIGHT; y++) {
        for (size_t x = 0; x + LINE_POOL_SIZE <= RETINA_WIDTH; x += LINE_POOL_SIZE) {
            for (size_t type = 0; type < NUM_BOUNDARY_TYPES; type++) {
                if (lineIdx >= lineNeurons_.size()) break;
                NeuronId lineNeuron = lineNeurons_[lineIdx++];
                for (size_t dx = 0; dx < LINE_POOL_SIZE; dx++) {
                    NeuronId boundary = boundaryNeurons_[boundaryIdx(x + dx, y, static_cast<BoundaryType>(type))];
                    network_.connectNeurons(boundary, lineNeuron, 1, false);
                }
            }
        }
    }

    // Vertical line segments
    for (size_t x = 0; x < RETINA_WIDTH; x++) {
        for (size_t y = 0; y + LINE_POOL_SIZE <= RETINA_HEIGHT; y += LINE_POOL_SIZE) {
            for (size_t type = 0; type < NUM_BOUNDARY_TYPES; type++) {
                if (lineIdx >= lineNeurons_.size()) break;
                NeuronId lineNeuron = lineNeurons_[lineIdx++];
                for (size_t dy = 0; dy < LINE_POOL_SIZE; dy++) {
                    NeuronId boundary = boundaryNeurons_[boundaryIdx(x, y + dy, static_cast<BoundaryType>(type))];
                    network_.connectNeurons(boundary, lineNeuron, 1, false);
                }
            }
        }
    }
}

// ========================================
// Layer 3: Corner Detectors
// ========================================

void VisionSystem::buildCornerDetectors() {
    size_t totalCorners = RETINA_SIZE * NUM_CORNER_TYPES;
    cornerNeurons_.reserve(totalCorners);

    for (size_t i = 0; i < totalCorners; i++) {
        NeuronId n = network_.addNeuron(2, 0, 2);
        cornerNeurons_.push_back(n);
    }
}

void VisionSystem::wireBoundaryToCorner() {
    for (size_t y = 1; y < RETINA_HEIGHT - 1; y++) {
        for (size_t x = 1; x < RETINA_WIDTH - 1; x++) {
            NeuronId topLeft = cornerNeurons_[cornerIdx(x, y, CornerType::TOP_LEFT)];
            NeuronId topRight = cornerNeurons_[cornerIdx(x, y, CornerType::TOP_RIGHT)];
            NeuronId bottomLeft = cornerNeurons_[cornerIdx(x, y, CornerType::BOTTOM_LEFT)];
            NeuronId bottomRight = cornerNeurons_[cornerIdx(x, y, CornerType::BOTTOM_RIGHT)];

            NeuronId hBoundary = boundaryNeurons_[boundaryIdx(x, y, BoundaryType::HORIZONTAL)];
            NeuronId vBoundary = boundaryNeurons_[boundaryIdx(x, y, BoundaryType::VERTICAL)];
            NeuronId hBoundaryAbove = boundaryNeurons_[boundaryIdx(x, y - 1, BoundaryType::HORIZONTAL)];
            NeuronId vBoundaryLeft = boundaryNeurons_[boundaryIdx(x - 1, y, BoundaryType::VERTICAL)];

            network_.connectNeurons(hBoundary, topLeft, 1, false);
            network_.connectNeurons(vBoundary, topLeft, 1, false);

            network_.connectNeurons(hBoundary, topRight, 1, false);
            network_.connectNeurons(vBoundaryLeft, topRight, 1, false);

            network_.connectNeurons(hBoundaryAbove, bottomLeft, 1, false);
            network_.connectNeurons(vBoundary, bottomLeft, 1, false);

            network_.connectNeurons(hBoundaryAbove, bottomRight, 1, false);
            network_.connectNeurons(vBoundaryLeft, bottomRight, 1, false);
        }
    }
}

// ========================================
// Layer 3: Acute Vertex Detectors
// ========================================

void VisionSystem::buildAcuteVertexDetectors() {
    size_t totalAcuteVertices = RETINA_SIZE * NUM_ACUTE_VERTEX_TYPES;
    acuteVertexNeurons_.reserve(totalAcuteVertices);

    for (size_t i = 0; i < totalAcuteVertices; i++) {
        NeuronId n = network_.addNeuron(4, 0, 2);
        acuteVertexNeurons_.push_back(n);
    }
}

void VisionSystem::wireBoundaryToAcuteVertex() {
    for (size_t y = 1; y < RETINA_HEIGHT - 1; y++) {
        for (size_t x = 1; x < RETINA_WIDTH - 1; x++) {
            NeuronId peakVertex = acuteVertexNeurons_[acuteVertexIdx(x, y, AcuteVertexType::PEAK)];
            NeuronId valleyVertex = acuteVertexNeurons_[acuteVertexIdx(x, y, AcuteVertexType::VALLEY)];

            NeuronId diagHere = boundaryNeurons_[boundaryIdx(x, y, BoundaryType::DIAGONAL)];
            NeuronId antiDiagHere = boundaryNeurons_[boundaryIdx(x, y, BoundaryType::ANTI_DIAGONAL)];
            NeuronId vertHere = boundaryNeurons_[boundaryIdx(x, y, BoundaryType::VERTICAL)];
            NeuronId horizHere = boundaryNeurons_[boundaryIdx(x, y, BoundaryType::HORIZONTAL)];

            network_.connectNeurons(diagHere, peakVertex, 2, false);
            network_.connectNeurons(antiDiagHere, peakVertex, 2, false);
            network_.connectNeurons(diagHere, valleyVertex, 2, false);
            network_.connectNeurons(antiDiagHere, valleyVertex, 2, false);

            network_.connectNeurons(vertHere, peakVertex, -3, false);
            network_.connectNeurons(horizHere, peakVertex, -3, false);
            network_.connectNeurons(vertHere, valleyVertex, -3, false);
            network_.connectNeurons(horizHere, valleyVertex, -3, false);
        }
    }
}

// ========================================
// Parietal Patch: Spatial Awareness ("Ruler Neurons")
// ========================================

void VisionSystem::buildDimensionSensors() {
    // Input Dimensions (Sensory) - low threshold, responds to external input
    // These encode "what size is the input grid?"
    inputWidthNeurons_.reserve(MAX_GRID_DIM);
    inputHeightNeurons_.reserve(MAX_GRID_DIM);
    for (size_t i = 0; i < MAX_GRID_DIM; i++) {
        inputWidthNeurons_.push_back(network_.addNeuron(2, 0, 1));
        inputHeightNeurons_.push_back(network_.addNeuron(2, 0, 1));
    }

    // Output Dimensions (Motor/Prediction) - higher threshold
    // These encode "what size should the output grid be?"
    // The brain learns to activate these based on input dimensions
    outputWidthNeurons_.reserve(MAX_GRID_DIM);
    outputHeightNeurons_.reserve(MAX_GRID_DIM);
    for (size_t i = 0; i < MAX_GRID_DIM; i++) {
        outputWidthNeurons_.push_back(network_.addNeuron(5, 0, 1));
        outputHeightNeurons_.push_back(network_.addNeuron(5, 0, 1));
    }

    // Wire input->output dimension neurons with plastic synapses
    // This allows the brain to learn dimension relationships
    // e.g., "when input is 5, output is 7" (add-border pattern)
    for (size_t i = 0; i < MAX_GRID_DIM; i++) {
        for (size_t j = 0; j < MAX_GRID_DIM; j++) {
            // Width -> Width (plastic, starts weak)
            network_.connectNeurons(inputWidthNeurons_[i], outputWidthNeurons_[j], 1, true);
            // Height -> Height (plastic, starts weak)
            network_.connectNeurons(inputHeightNeurons_[i], outputHeightNeurons_[j], 1, true);
        }
    }
}

void VisionSystem::setInputDimensions(int w, int h) {
    // Activate the "ruler neuron" for the input dimensions
    // Strong signal: "The input width is w, height is h"
    if (w > 0 && w < static_cast<int>(MAX_GRID_DIM)) {
        network_.injectCharge(inputWidthNeurons_[w], 20);
    }
    if (h > 0 && h < static_cast<int>(MAX_GRID_DIM)) {
        network_.injectCharge(inputHeightNeurons_[h], 20);
    }
}

void VisionSystem::setOutputDimensions(int w, int h) {
    // During training: activate the TARGET output dimensions
    // This allows STDP to strengthen input->output associations
    if (w > 0 && w < static_cast<int>(MAX_GRID_DIM)) {
        network_.injectCharge(outputWidthNeurons_[w], 20);
    }
    if (h > 0 && h < static_cast<int>(MAX_GRID_DIM)) {
        network_.injectCharge(outputHeightNeurons_[h], 20);
    }
}

std::pair<int, int> VisionSystem::getPredictedDimensions() const {
    // Winner-take-all readout of output dimension neurons
    // Returns the dimension with highest charge
    int bestW = 0;
    Charge maxW = 0;
    int bestH = 0;
    Charge maxH = 0;

    for (size_t i = 1; i < MAX_GRID_DIM; i++) {  // Start at 1, 0 is invalid
        Charge chargeW = network_.getCharge(outputWidthNeurons_[i]);
        if (chargeW > maxW) {
            maxW = chargeW;
            bestW = static_cast<int>(i);
        }

        Charge chargeH = network_.getCharge(outputHeightNeurons_[i]);
        if (chargeH > maxH) {
            maxH = chargeH;
            bestH = static_cast<int>(i);
        }
    }

    // Fallback if brain is silent: default to small grid
    if (bestW == 0) bestW = 3;
    if (bestH == 0) bestH = 3;

    return {bestW, bestH};
}

// ========================================
// Main Interface
// ========================================

void VisionSystem::present(const std::vector<uint8_t>& image) {
    if (image.size() != RETINA_SIZE) {
        return;
    }

    currentImage_ = image;
    processRetina();
}

void VisionSystem::processRetina() {
    // Clear previous state
    std::fill(retinaState_.begin(), retinaState_.end(), false);

    // Activate the SPECIFIC color channel for each pixel
    for (size_t i = 0; i < RETINA_SIZE; i++) {
        uint8_t pixelValue = currentImage_[i];

        // Convert grayscale voltage back to ARC color index
        // Based on convert_arc.py: 0, 28, 56, 84, 112, 140, 168, 196, 224, 252
        uint8_t colorIdx = 0;
        if (pixelValue >= 240) colorIdx = 9;       // 252 -> Maroon
        else if (pixelValue >= 210) colorIdx = 8;  // 224 -> Cyan
        else if (pixelValue >= 182) colorIdx = 7;  // 196 -> Orange
        else if (pixelValue >= 154) colorIdx = 6;  // 168 -> Magenta
        else if (pixelValue >= 126) colorIdx = 5;  // 140 -> Gray
        else if (pixelValue >= 98) colorIdx = 4;   // 112 -> Yellow
        else if (pixelValue >= 70) colorIdx = 3;   // 84 -> Green
        else if (pixelValue >= 42) colorIdx = 2;   // 56 -> Red
        else if (pixelValue >= 14) colorIdx = 1;   // 28 -> Blue
        // else colorIdx = 0 (Black/Background)

        // Activate the specific color channel neuron
        size_t neuronIdx = i * NUM_COLORS + colorIdx;
        retinaState_[neuronIdx] = true;
        network_.injectCharge(retinaNeurons_[neuronIdx], 10);
    }
}

void VisionSystem::step() {
    // Vision neurons fire through main network step
}

void VisionSystem::reset() {
    std::fill(retinaState_.begin(), retinaState_.end(), false);
    std::fill(currentImage_.begin(), currentImage_.end(), 0);
}

// ========================================
// Query Interface
// ========================================

bool VisionSystem::isRetinaActive(size_t x, size_t y) const {
    // Return true if ANY non-black color is active (for shape detection)
    if (x >= RETINA_WIDTH || y >= RETINA_HEIGHT) return false;

    size_t baseIdx = idx(x, y) * NUM_COLORS;
    for (size_t c = 1; c < NUM_COLORS; c++) {  // Skip color 0 (black)
        if (retinaState_[baseIdx + c]) return true;
    }
    return false;
}

uint8_t VisionSystem::getRetinaColor(size_t x, size_t y) const {
    // Return which color is active at this position (0 if none/black)
    if (x >= RETINA_WIDTH || y >= RETINA_HEIGHT) return 0;

    size_t baseIdx = idx(x, y) * NUM_COLORS;

    // Return the first active non-black color found
    for (size_t c = 1; c < NUM_COLORS; c++) {
        if (retinaState_[baseIdx + c]) return static_cast<uint8_t>(c);
    }

    // Check if black is explicitly active
    if (retinaState_[baseIdx]) return 0;

    return 0;  // Nothing active
}

uint8_t VisionSystem::getPixelValue(size_t x, size_t y) const {
    if (x >= RETINA_WIDTH || y >= RETINA_HEIGHT) return 0;
    return currentImage_[y * RETINA_WIDTH + x];
}

bool VisionSystem::isBoundaryActive(size_t x, size_t y, BoundaryType type) const {
    if (x >= RETINA_WIDTH || y >= RETINA_HEIGHT) return false;
    return network_.didFire(boundaryNeurons_[boundaryIdx(x, y, type)]);
}

bool VisionSystem::isLineActive(size_t x, size_t y, BoundaryType type) const {
    return false;  // TODO: implement proper line indexing
}

std::vector<std::pair<size_t, size_t>> VisionSystem::getActiveRetina() const {
    std::vector<std::pair<size_t, size_t>> active;
    for (size_t y = 0; y < RETINA_HEIGHT; y++) {
        for (size_t x = 0; x < RETINA_WIDTH; x++) {
            if (isRetinaActive(x, y)) {
                active.emplace_back(x, y);
            }
        }
    }
    return active;
}

std::vector<std::tuple<size_t, size_t, BoundaryType>> VisionSystem::getActiveBoundaries() const {
    std::vector<std::tuple<size_t, size_t, BoundaryType>> active;

    for (size_t y = 0; y < RETINA_HEIGHT; y++) {
        for (size_t x = 0; x < RETINA_WIDTH; x++) {
            for (size_t t = 0; t < NUM_BOUNDARY_TYPES; t++) {
                BoundaryType type = static_cast<BoundaryType>(t);
                if (network_.didFire(boundaryNeurons_[boundaryIdx(x, y, type)])) {
                    active.emplace_back(x, y, type);
                }
            }
        }
    }

    return active;
}

NeuronId VisionSystem::getRetinaNeuron(size_t x, size_t y) const {
    // Returns the BLACK channel neuron (for backwards compatibility)
    // Use getRetinaNeurons() for full access
    if (x >= RETINA_WIDTH || y >= RETINA_HEIGHT) return INVALID_NEURON;
    return retinaNeurons_[idx(x, y) * NUM_COLORS];
}

NeuronId VisionSystem::getBoundaryNeuron(size_t x, size_t y, BoundaryType type) const {
    if (x >= RETINA_WIDTH || y >= RETINA_HEIGHT) return INVALID_NEURON;
    return boundaryNeurons_[boundaryIdx(x, y, type)];
}

NeuronId VisionSystem::getLineNeuron(size_t x, size_t y, BoundaryType type) const {
    return INVALID_NEURON;
}

size_t VisionSystem::getActiveRetinaCount() const {
    size_t count = 0;
    // Count active PIXELS (not channels)
    for (size_t i = 0; i < RETINA_SIZE; i++) {
        for (size_t c = 1; c < NUM_COLORS; c++) {
            if (retinaState_[i * NUM_COLORS + c]) {
                count++;
                break;  // Count pixel once even if multiple colors
            }
        }
    }
    return count;
}

size_t VisionSystem::getActiveBoundaryCount() const {
    size_t count = 0;
    for (NeuronId n : boundaryNeurons_) {
        if (network_.didFire(n)) count++;
    }
    return count;
}

size_t VisionSystem::getActiveLineCount() const {
    size_t count = 0;
    for (NeuronId n : lineNeurons_) {
        if (network_.didFire(n)) count++;
    }
    return count;
}

// ========================================
// Corner Query Methods
// ========================================

std::vector<std::tuple<size_t, size_t, CornerType>> VisionSystem::getActiveCorners() const {
    std::vector<std::tuple<size_t, size_t, CornerType>> active;

    for (size_t y = 0; y < RETINA_HEIGHT; y++) {
        for (size_t x = 0; x < RETINA_WIDTH; x++) {
            for (size_t t = 0; t < NUM_CORNER_TYPES; t++) {
                CornerType type = static_cast<CornerType>(t);
                size_t index = cornerIdx(x, y, type);
                if (index < cornerNeurons_.size() && network_.didFire(cornerNeurons_[index])) {
                    active.emplace_back(x, y, type);
                }
            }
        }
    }

    return active;
}

bool VisionSystem::isCornerActive(size_t x, size_t y, CornerType type) const {
    if (x >= RETINA_WIDTH || y >= RETINA_HEIGHT) return false;
    size_t index = cornerIdx(x, y, type);
    if (index >= cornerNeurons_.size()) return false;
    return network_.didFire(cornerNeurons_[index]);
}

NeuronId VisionSystem::getCornerNeuron(size_t x, size_t y, CornerType type) const {
    if (x >= RETINA_WIDTH || y >= RETINA_HEIGHT) return INVALID_NEURON;
    size_t index = cornerIdx(x, y, type);
    if (index >= cornerNeurons_.size()) return INVALID_NEURON;
    return cornerNeurons_[index];
}

size_t VisionSystem::getActiveCornerCount() const {
    size_t count = 0;
    for (NeuronId n : cornerNeurons_) {
        if (network_.didFire(n)) count++;
    }
    return count;
}

// ========================================
// Feature Counting
// ========================================

size_t VisionSystem::countCornersByType(CornerType type) const {
    size_t count = 0;

    for (size_t y = 0; y < RETINA_HEIGHT; y++) {
        for (size_t x = 0; x < RETINA_WIDTH; x++) {
            size_t index = cornerIdx(x, y, type);
            if (index < cornerNeurons_.size() && network_.didFire(cornerNeurons_[index])) {
                count++;
            }
        }
    }

    return count;
}

size_t VisionSystem::countBoundariesByType(BoundaryType type) const {
    size_t count = 0;

    for (size_t y = 0; y < RETINA_HEIGHT; y++) {
        for (size_t x = 0; x < RETINA_WIDTH; x++) {
            if (network_.didFire(boundaryNeurons_[boundaryIdx(x, y, type)])) {
                count++;
            }
        }
    }

    return count;
}

// ========================================
// Acute Vertex Query Methods
// ========================================

std::vector<std::tuple<size_t, size_t, AcuteVertexType>> VisionSystem::getActiveAcuteVertices() const {
    std::vector<std::tuple<size_t, size_t, AcuteVertexType>> active;

    for (size_t y = 0; y < RETINA_HEIGHT; y++) {
        for (size_t x = 0; x < RETINA_WIDTH; x++) {
            for (size_t t = 0; t < NUM_ACUTE_VERTEX_TYPES; t++) {
                AcuteVertexType type = static_cast<AcuteVertexType>(t);
                size_t index = acuteVertexIdx(x, y, type);
                if (index < acuteVertexNeurons_.size() && network_.didFire(acuteVertexNeurons_[index])) {
                    active.emplace_back(x, y, type);
                }
            }
        }
    }

    return active;
}

bool VisionSystem::isAcuteVertexActive(size_t x, size_t y, AcuteVertexType type) const {
    if (x >= RETINA_WIDTH || y >= RETINA_HEIGHT) return false;
    size_t index = acuteVertexIdx(x, y, type);
    if (index >= acuteVertexNeurons_.size()) return false;
    return network_.didFire(acuteVertexNeurons_[index]);
}

NeuronId VisionSystem::getAcuteVertexNeuron(size_t x, size_t y, AcuteVertexType type) const {
    if (x >= RETINA_WIDTH || y >= RETINA_HEIGHT) return INVALID_NEURON;
    size_t index = acuteVertexIdx(x, y, type);
    if (index >= acuteVertexNeurons_.size()) return INVALID_NEURON;
    return acuteVertexNeurons_[index];
}

size_t VisionSystem::getActiveAcuteVertexCount() const {
    size_t count = 0;
    for (NeuronId n : acuteVertexNeurons_) {
        if (network_.didFire(n)) count++;
    }
    return count;
}

size_t VisionSystem::countAcuteVerticesByType(AcuteVertexType type) const {
    size_t count = 0;

    for (size_t y = 0; y < RETINA_HEIGHT; y++) {
        for (size_t x = 0; x < RETINA_WIDTH; x++) {
            size_t index = acuteVertexIdx(x, y, type);
            if (index < acuteVertexNeurons_.size() && network_.didFire(acuteVertexNeurons_[index])) {
                count++;
            }
        }
    }

    return count;
}

size_t VisionSystem::countTotalAcuteVertices() const {
    return countAcuteVerticesByType(AcuteVertexType::PEAK) +
           countAcuteVerticesByType(AcuteVertexType::VALLEY);
}

}  // namespace bpagi
