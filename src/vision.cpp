#include "bpagi/vision.hpp"
#include <cmath>

namespace bpagi {

VisionSystem::VisionSystem(Network& network)
    : network_(network)
{
    // Initialize state vectors
    retinaState_.resize(RETINA_SIZE, false);
    currentImage_.resize(RETINA_SIZE, 0);

    // Build the three-layer hierarchy
    buildRetina();
    buildBoundaryDetectors();
    buildLineIntegrators();
    buildCornerDetectors();
    buildAcuteVertexDetectors();

    // Wire the layers
    wireRetinaToBoundary();
    wireBoundaryToLine();
    wireBoundaryToCorner();
    wireBoundaryToAcuteVertex();
}

// ========================================
// Layer 1: Retina (Photoreceptors)
// ========================================

void VisionSystem::buildRetina() {
    retinaNeurons_.reserve(RETINA_SIZE);

    for (size_t i = 0; i < RETINA_SIZE; i++) {
        // Retina neurons: low threshold (respond to external input)
        // No leak, short refractory
        NeuronId n = network_.addNeuron(2, 0, 1);
        retinaNeurons_.push_back(n);
    }
}

// ========================================
// Layer 2: Boundary Detectors (V1)
// ========================================

void VisionSystem::buildBoundaryDetectors() {
    // 4 boundary types per pixel position
    size_t totalBoundary = RETINA_SIZE * NUM_BOUNDARY_TYPES;
    boundaryNeurons_.reserve(totalBoundary);

    for (size_t i = 0; i < totalBoundary; i++) {
        // Boundary neurons: threshold=2 for edge detection
        // Edge (one neighbor OFF): +4 - 2 = +2 >= 2 (fires)
        // Interior (both neighbors ON): +4 - 2 - 2 = 0 < 2 (no fire)
        NeuronId n = network_.addNeuron(2, 0, 2);
        boundaryNeurons_.push_back(n);
    }
}

void VisionSystem::wireRetinaToBoundary() {
    // Wire boundary detectors based on their orientation
    // Each boundary neuron checks specific retina positions for contrast

    for (size_t y = 0; y < RETINA_HEIGHT; y++) {
        for (size_t x = 0; x < RETINA_WIDTH; x++) {
            // Get boundary neuron IDs for this position
            NeuronId vertBoundary = boundaryNeurons_[boundaryIdx(x, y, BoundaryType::VERTICAL)];
            NeuronId horizBoundary = boundaryNeurons_[boundaryIdx(x, y, BoundaryType::HORIZONTAL)];
            NeuronId diagBoundary = boundaryNeurons_[boundaryIdx(x, y, BoundaryType::DIAGONAL)];
            NeuronId antiDiagBoundary = boundaryNeurons_[boundaryIdx(x, y, BoundaryType::ANTI_DIAGONAL)];

            // VERTICAL boundary (|): fires if left differs from right
            // Excitation from current pixel, inhibition from neighbors
            if (x > 0 && x < RETINA_WIDTH - 1) {
                NeuronId left = retinaNeurons_[idx(x - 1, y)];
                NeuronId center = retinaNeurons_[idx(x, y)];
                NeuronId right = retinaNeurons_[idx(x + 1, y)];

                // Edge detection using center-surround inhibition:
                // +4 from center, -2 from each neighbor, threshold=2
                // Edge (one neighbor OFF): +4 - 2 = +2 >= 2 (fires)
                // Interior (both neighbors ON): +4 - 2 - 2 = 0 < 2 (no fire)
                network_.connectNeurons(center, vertBoundary, 4, false);
                network_.connectNeurons(left, vertBoundary, -2, false);
                network_.connectNeurons(right, vertBoundary, -2, false);
            }

            // HORIZONTAL boundary (-): fires if top differs from bottom
            if (y > 0 && y < RETINA_HEIGHT - 1) {
                NeuronId top = retinaNeurons_[idx(x, y - 1)];
                NeuronId center = retinaNeurons_[idx(x, y)];
                NeuronId bottom = retinaNeurons_[idx(x, y + 1)];

                network_.connectNeurons(center, horizBoundary, 4, false);
                network_.connectNeurons(top, horizBoundary, -2, false);
                network_.connectNeurons(bottom, horizBoundary, -2, false);
            }

            // DIAGONAL boundary (/): fires on diagonal edge
            if (x > 0 && y > 0 && x < RETINA_WIDTH - 1 && y < RETINA_HEIGHT - 1) {
                NeuronId topLeft = retinaNeurons_[idx(x - 1, y - 1)];
                NeuronId center = retinaNeurons_[idx(x, y)];
                NeuronId bottomRight = retinaNeurons_[idx(x + 1, y + 1)];

                network_.connectNeurons(center, diagBoundary, 4, false);
                network_.connectNeurons(topLeft, diagBoundary, -2, false);
                network_.connectNeurons(bottomRight, diagBoundary, -2, false);
            }

            // ANTI-DIAGONAL boundary (\): fires on anti-diagonal edge
            if (x > 0 && y > 0 && x < RETINA_WIDTH - 1 && y < RETINA_HEIGHT - 1) {
                NeuronId topRight = retinaNeurons_[idx(x + 1, y - 1)];
                NeuronId center = retinaNeurons_[idx(x, y)];
                NeuronId bottomLeft = retinaNeurons_[idx(x - 1, y + 1)];

                network_.connectNeurons(center, antiDiagBoundary, 4, false);
                network_.connectNeurons(topRight, antiDiagBoundary, -2, false);
                network_.connectNeurons(bottomLeft, antiDiagBoundary, -2, false);
            }
        }
    }
}

// ========================================
// Layer 3: Line Integrators (V2)
// ========================================

void VisionSystem::buildLineIntegrators() {
    // Line integrators pool 4x1 patches of boundary neurons
    // Create one for each 4-pixel segment in each orientation

    size_t lineWidth = RETINA_WIDTH / LINE_POOL_SIZE;
    size_t lineHeight = RETINA_HEIGHT / LINE_POOL_SIZE;
    size_t totalLines = lineWidth * lineHeight * NUM_BOUNDARY_TYPES * 2;  // horizontal + vertical lines

    lineNeurons_.reserve(totalLines);

    for (size_t i = 0; i < totalLines; i++) {
        // Line neurons: high threshold (need multiple boundary inputs)
        NeuronId n = network_.addNeuron(static_cast<Charge>(LINE_MIN_ACTIVE), 0, 3);
        lineNeurons_.push_back(n);
    }
}

void VisionSystem::wireBoundaryToLine() {
    // Wire boundary detectors to line integrators
    // Each line integrator covers a 4x1 patch of boundary neurons

    size_t lineIdx = 0;

    // Horizontal line segments (pool 4 horizontal pixels)
    for (size_t y = 0; y < RETINA_HEIGHT; y++) {
        for (size_t x = 0; x + LINE_POOL_SIZE <= RETINA_WIDTH; x += LINE_POOL_SIZE) {
            for (size_t type = 0; type < NUM_BOUNDARY_TYPES; type++) {
                if (lineIdx >= lineNeurons_.size()) break;

                NeuronId lineNeuron = lineNeurons_[lineIdx++];

                // Connect 4 consecutive boundary neurons to this line integrator
                for (size_t dx = 0; dx < LINE_POOL_SIZE; dx++) {
                    NeuronId boundary = boundaryNeurons_[boundaryIdx(x + dx, y, static_cast<BoundaryType>(type))];
                    network_.connectNeurons(boundary, lineNeuron, 1, false);
                }
            }
        }
    }

    // Vertical line segments (pool 4 vertical pixels)
    for (size_t x = 0; x < RETINA_WIDTH; x++) {
        for (size_t y = 0; y + LINE_POOL_SIZE <= RETINA_HEIGHT; y += LINE_POOL_SIZE) {
            for (size_t type = 0; type < NUM_BOUNDARY_TYPES; type++) {
                if (lineIdx >= lineNeurons_.size()) break;

                NeuronId lineNeuron = lineNeurons_[lineIdx++];

                // Connect 4 consecutive boundary neurons to this line integrator
                for (size_t dy = 0; dy < LINE_POOL_SIZE; dy++) {
                    NeuronId boundary = boundaryNeurons_[boundaryIdx(x, y + dy, static_cast<BoundaryType>(type))];
                    network_.connectNeurons(boundary, lineNeuron, 1, false);
                }
            }
        }
    }
}

// ========================================
// Layer 3: Corner Detectors (Geometric Features)
// ========================================

void VisionSystem::buildCornerDetectors() {
    // 4 corner types per pixel position
    // Corners detect where perpendicular edges meet
    size_t totalCorners = RETINA_SIZE * NUM_CORNER_TYPES;
    cornerNeurons_.reserve(totalCorners);

    for (size_t i = 0; i < totalCorners; i++) {
        // Corner neurons: threshold=2 (need BOTH perpendicular edges)
        // This is coincidence detection - AND logic
        NeuronId n = network_.addNeuron(2, 0, 2);
        cornerNeurons_.push_back(n);
    }
}

void VisionSystem::wireBoundaryToCorner() {
    // Wire corner detectors to detect where perpendicular edges meet
    //
    // Corner types based on which edges meet:
    // TOP_LEFT (┌): horizontal edge going right + vertical edge going down
    // TOP_RIGHT (┐): horizontal edge going left + vertical edge going down
    // BOTTOM_LEFT (└): horizontal edge going right + vertical edge going up
    // BOTTOM_RIGHT (┘): horizontal edge going left + vertical edge going up
    //
    // We detect this by looking at boundary neurons at adjacent positions

    for (size_t y = 1; y < RETINA_HEIGHT - 1; y++) {
        for (size_t x = 1; x < RETINA_WIDTH - 1; x++) {
            // Get corner neuron IDs for this position
            NeuronId topLeft = cornerNeurons_[cornerIdx(x, y, CornerType::TOP_LEFT)];
            NeuronId topRight = cornerNeurons_[cornerIdx(x, y, CornerType::TOP_RIGHT)];
            NeuronId bottomLeft = cornerNeurons_[cornerIdx(x, y, CornerType::BOTTOM_LEFT)];
            NeuronId bottomRight = cornerNeurons_[cornerIdx(x, y, CornerType::BOTTOM_RIGHT)];

            // Get boundary neurons at this position
            NeuronId hBoundary = boundaryNeurons_[boundaryIdx(x, y, BoundaryType::HORIZONTAL)];
            NeuronId vBoundary = boundaryNeurons_[boundaryIdx(x, y, BoundaryType::VERTICAL)];

            // Also get boundaries at adjacent positions for better corner detection
            NeuronId hBoundaryBelow = boundaryNeurons_[boundaryIdx(x, y + 1, BoundaryType::HORIZONTAL)];
            NeuronId hBoundaryAbove = boundaryNeurons_[boundaryIdx(x, y - 1, BoundaryType::HORIZONTAL)];
            NeuronId vBoundaryRight = boundaryNeurons_[boundaryIdx(x + 1, y, BoundaryType::VERTICAL)];
            NeuronId vBoundaryLeft = boundaryNeurons_[boundaryIdx(x - 1, y, BoundaryType::VERTICAL)];

            // TOP_LEFT corner: horizontal edge here/below + vertical edge here/right
            // Fires when there's a top-left corner of a shape
            network_.connectNeurons(hBoundary, topLeft, 1, false);
            network_.connectNeurons(vBoundary, topLeft, 1, false);

            // TOP_RIGHT corner: horizontal edge here/below + vertical edge here/left
            network_.connectNeurons(hBoundary, topRight, 1, false);
            network_.connectNeurons(vBoundaryLeft, topRight, 1, false);

            // BOTTOM_LEFT corner: horizontal edge here/above + vertical edge here/right
            network_.connectNeurons(hBoundaryAbove, bottomLeft, 1, false);
            network_.connectNeurons(vBoundary, bottomLeft, 1, false);

            // BOTTOM_RIGHT corner: horizontal edge here/above + vertical edge here/left
            network_.connectNeurons(hBoundaryAbove, bottomRight, 1, false);
            network_.connectNeurons(vBoundaryLeft, bottomRight, 1, false);
        }
    }
}

// ========================================
// Layer 3: Acute Vertex Detectors (Triangle-like angles)
// ========================================

void VisionSystem::buildAcuteVertexDetectors() {
    // 2 acute vertex types per pixel position
    // These detect where DIAGONAL (/) meets ANTI-DIAGONAL (\)
    // Critical for distinguishing triangles from squares!
    //
    // PEAK (^): Triangle apex pointing UP
    // VALLEY (v): Triangle apex pointing DOWN
    size_t totalAcuteVertices = RETINA_SIZE * NUM_ACUTE_VERTEX_TYPES;
    acuteVertexNeurons_.reserve(totalAcuteVertices);

    for (size_t i = 0; i < totalAcuteVertices; i++) {
        // Acute vertex neurons: threshold=4 (need BOTH diagonal types)
        // Excitation: diag(+2) + antiDiag(+2) = +4 >= threshold ✓
        // Inhibition: vert(-3) or horiz(-3) reduces total below threshold
        // Triangle apex: +2+2 = +4 >= 4 ✓ (no orthogonal edges)
        // Square corner: +2+2-3-3 = -2 < 4 ✗ (orthogonal edges present)
        NeuronId n = network_.addNeuron(4, 0, 2);
        acuteVertexNeurons_.push_back(n);
    }
}

void VisionSystem::wireBoundaryToAcuteVertex() {
    // Wire acute vertex detectors to detect where diagonal edges meet
    // AND where orthogonal edges are ABSENT
    //
    // KEY INSIGHT for Shape Discrimination:
    //   Triangle apex: Has diagonal + anti-diagonal, NO orthogonal edges
    //   Square corner: Has diagonal + anti-diagonal, PLUS orthogonal edges
    //
    // By inhibiting acute vertices when orthogonal edges are present,
    // we make this detector specific to triangle-like angles.
    //
    // PEAK (^): Diagonal (/) meets Anti-Diagonal (\), apex pointing UP
    // VALLEY (v): Diagonal (/) meets Anti-Diagonal (\), apex pointing DOWN

    for (size_t y = 1; y < RETINA_HEIGHT - 1; y++) {
        for (size_t x = 1; x < RETINA_WIDTH - 1; x++) {
            // Get acute vertex neuron IDs for this position
            NeuronId peakVertex = acuteVertexNeurons_[acuteVertexIdx(x, y, AcuteVertexType::PEAK)];
            NeuronId valleyVertex = acuteVertexNeurons_[acuteVertexIdx(x, y, AcuteVertexType::VALLEY)];

            // Get boundary neurons at this position
            NeuronId diagHere = boundaryNeurons_[boundaryIdx(x, y, BoundaryType::DIAGONAL)];
            NeuronId antiDiagHere = boundaryNeurons_[boundaryIdx(x, y, BoundaryType::ANTI_DIAGONAL)];
            NeuronId vertHere = boundaryNeurons_[boundaryIdx(x, y, BoundaryType::VERTICAL)];
            NeuronId horizHere = boundaryNeurons_[boundaryIdx(x, y, BoundaryType::HORIZONTAL)];

            // EXCITATION: Both diagonal types must be present
            // Weight=2 each, threshold=4 means BOTH must fire
            network_.connectNeurons(diagHere, peakVertex, 2, false);
            network_.connectNeurons(antiDiagHere, peakVertex, 2, false);

            network_.connectNeurons(diagHere, valleyVertex, 2, false);
            network_.connectNeurons(antiDiagHere, valleyVertex, 2, false);

            // INHIBITION: Suppress if orthogonal edges are present
            // This discriminates triangle angles from square corners
            // Weight=-3: If vertical OR horizontal fires, it suppresses the vertex
            network_.connectNeurons(vertHere, peakVertex, -3, false);
            network_.connectNeurons(horizHere, peakVertex, -3, false);

            network_.connectNeurons(vertHere, valleyVertex, -3, false);
            network_.connectNeurons(horizHere, valleyVertex, -3, false);
        }
    }
}

// ========================================
// Main Interface
// ========================================

void VisionSystem::present(const std::vector<uint8_t>& image) {
    if (image.size() != RETINA_SIZE) {
        return;  // Invalid image size
    }

    currentImage_ = image;
    processRetina();
}

void VisionSystem::processRetina() {
    // Convert image pixels to retina activation
    // Inject charge into retina neurons based on contrast

    for (size_t i = 0; i < RETINA_SIZE; i++) {
        uint8_t pixel = currentImage_[i];

        // Check if pixel has significant contrast from background
        int contrast = std::abs(static_cast<int>(pixel) - static_cast<int>(BACKGROUND_LEVEL));

        if (contrast > CONTRAST_THRESHOLD) {
            // This photoreceptor is active - inject charge
            retinaState_[i] = true;
            network_.injectCharge(retinaNeurons_[i], 10);  // Strong activation
        } else {
            retinaState_[i] = false;
        }
    }
}

void VisionSystem::step() {
    // The vision system's neurons are part of the main network
    // They fire automatically through the network's step()
    // This method can be used for any vision-specific post-processing
}

void VisionSystem::reset() {
    std::fill(retinaState_.begin(), retinaState_.end(), false);
    std::fill(currentImage_.begin(), currentImage_.end(), 0);
}

// ========================================
// Query Interface
// ========================================

bool VisionSystem::isRetinaActive(size_t x, size_t y) const {
    if (x >= RETINA_WIDTH || y >= RETINA_HEIGHT) return false;
    return retinaState_[idx(x, y)];
}

uint8_t VisionSystem::getPixelValue(size_t x, size_t y) const {
    if (x >= RETINA_WIDTH || y >= RETINA_HEIGHT) return 0;
    size_t index = y * RETINA_WIDTH + x;
    if (index >= currentImage_.size()) return 0;
    return currentImage_[index];
}

bool VisionSystem::isBoundaryActive(size_t x, size_t y, BoundaryType type) const {
    if (x >= RETINA_WIDTH || y >= RETINA_HEIGHT) return false;
    return network_.didFire(boundaryNeurons_[boundaryIdx(x, y, type)]);
}

bool VisionSystem::isLineActive(size_t x, size_t y, BoundaryType type) const {
    // Line neurons are indexed differently
    // This is a simplified check
    return false;  // TODO: implement proper line indexing
}

std::vector<std::pair<size_t, size_t>> VisionSystem::getActiveRetina() const {
    std::vector<std::pair<size_t, size_t>> active;
    for (size_t y = 0; y < RETINA_HEIGHT; y++) {
        for (size_t x = 0; x < RETINA_WIDTH; x++) {
            if (retinaState_[idx(x, y)]) {
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
    if (x >= RETINA_WIDTH || y >= RETINA_HEIGHT) return INVALID_NEURON;
    return retinaNeurons_[idx(x, y)];
}

NeuronId VisionSystem::getBoundaryNeuron(size_t x, size_t y, BoundaryType type) const {
    if (x >= RETINA_WIDTH || y >= RETINA_HEIGHT) return INVALID_NEURON;
    return boundaryNeurons_[boundaryIdx(x, y, type)];
}

NeuronId VisionSystem::getLineNeuron(size_t x, size_t y, BoundaryType type) const {
    // Simplified - return invalid for now
    return INVALID_NEURON;
}

size_t VisionSystem::getActiveRetinaCount() const {
    size_t count = 0;
    for (bool active : retinaState_) {
        if (active) count++;
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
// Feature Counting (Position-Invariant)
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
