#pragma once
#include "types.hpp"
#include "network.hpp"
#include <vector>
#include <cstdint>
#include <array>

namespace bpagi {

/**
 * Relative Vision System
 *
 * Blueprint: Biologically plausible visual processing using edge detection
 * rather than pixel-by-pixel analysis. This is dramatically more efficient:
 * - A 64x64 white square in a CNN: processes 4,096 pixels
 * - In boundary detection: only ~250 edge neurons fire (perimeter)
 *
 * Three-layer hierarchy:
 * 1. Retina (L1): 64x64 photoreceptors - detect contrast from background
 * 2. V1 Boundary (L2): Edge detectors - 4 orientations per pixel
 * 3. V2 Line Integrators (L3): Spatial pooling - detect continuous lines
 */

// Vision system dimensions
constexpr size_t RETINA_WIDTH = 64;
constexpr size_t RETINA_HEIGHT = 64;
constexpr size_t RETINA_SIZE = RETINA_WIDTH * RETINA_HEIGHT;

// Contrast threshold for receptor activation
constexpr uint8_t CONTRAST_THRESHOLD = 30;

// Background level (for contrast comparison)
constexpr uint8_t BACKGROUND_LEVEL = 0;  // Black background

// Boundary detector orientations
enum class BoundaryType : uint8_t {
    VERTICAL = 0,      // |  - detects left/right edges
    HORIZONTAL = 1,    // -  - detects top/bottom edges
    DIAGONAL = 2,      // /  - detects diagonal edges
    ANTI_DIAGONAL = 3  // \  - detects anti-diagonal edges
};
constexpr size_t NUM_BOUNDARY_TYPES = 4;

// Line integrator parameters
constexpr size_t LINE_POOL_SIZE = 4;      // 4x1 patch for line detection
constexpr size_t LINE_MIN_ACTIVE = 3;     // Need 3/4 to fire

// Corner detector types (Layer 3 - Geometric Features)
// Corners form where two perpendicular edges meet (90-degree corners)
enum class CornerType : uint8_t {
    TOP_LEFT = 0,      // Vertical + Horizontal meeting at top-left
    TOP_RIGHT = 1,     // Vertical + Horizontal meeting at top-right
    BOTTOM_LEFT = 2,   // Vertical + Horizontal meeting at bottom-left
    BOTTOM_RIGHT = 3   // Vertical + Horizontal meeting at bottom-right
};
constexpr size_t NUM_CORNER_TYPES = 4;

// Acute Vertex detector types (Layer 3 - Geometric Features)
// Acute vertices form where DIAGONAL (/) meets ANTI-DIAGONAL (\)
// These detect triangle-like acute angles (~60 degrees)
// Critical for distinguishing triangles from squares!
enum class AcuteVertexType : uint8_t {
    PEAK = 0,    // ^  - Diagonal and Anti-Diagonal meet at TOP (triangle apex)
    VALLEY = 1   // v  - Diagonal and Anti-Diagonal meet at BOTTOM (inverted apex)
};
constexpr size_t NUM_ACUTE_VERTEX_TYPES = 2;

/**
 * VisionSystem: Hardwired visual cortex simulation
 */
class VisionSystem {
public:
    // Constructor - creates all vision neurons in the network
    explicit VisionSystem(Network& network);

    // ========================================
    // Main Interface
    // ========================================

    // Present an image to the retina
    // Input: 64x64 grayscale image as row-major uint8_t vector
    void present(const std::vector<uint8_t>& image);

    // Step the vision system (call after network.step())
    void step();

    // Reset vision state
    void reset();

    // ========================================
    // Query Interface
    // ========================================

    // Get retina activation state
    bool isRetinaActive(size_t x, size_t y) const;

    // Get pixel value at position (grayscale 0-255)
    uint8_t getPixelValue(size_t x, size_t y) const;

    // Get boundary neuron firing state
    bool isBoundaryActive(size_t x, size_t y, BoundaryType type) const;

    // Get line integrator firing state
    bool isLineActive(size_t x, size_t y, BoundaryType type) const;

    // Get all active retina positions
    std::vector<std::pair<size_t, size_t>> getActiveRetina() const;

    // Get all active boundary positions with types
    std::vector<std::tuple<size_t, size_t, BoundaryType>> getActiveBoundaries() const;

    // Get all active corner positions with types
    std::vector<std::tuple<size_t, size_t, CornerType>> getActiveCorners() const;

    // Check if a corner is active
    bool isCornerActive(size_t x, size_t y, CornerType type) const;

    // Get all active acute vertex positions with types
    std::vector<std::tuple<size_t, size_t, AcuteVertexType>> getActiveAcuteVertices() const;

    // Check if an acute vertex is active
    bool isAcuteVertexActive(size_t x, size_t y, AcuteVertexType type) const;

    // ========================================
    // Neuron ID Access (for UKS integration)
    // ========================================

    // Get neuron ID for a specific retina position
    NeuronId getRetinaNeuron(size_t x, size_t y) const;

    // Get neuron ID for a boundary detector
    NeuronId getBoundaryNeuron(size_t x, size_t y, BoundaryType type) const;

    // Get neuron ID for a line integrator
    NeuronId getLineNeuron(size_t x, size_t y, BoundaryType type) const;

    // Get all boundary neuron IDs (for connecting to UKS bus)
    const std::vector<NeuronId>& getAllBoundaryNeurons() const { return boundaryNeurons_; }

    // Get neuron ID for a corner detector
    NeuronId getCornerNeuron(size_t x, size_t y, CornerType type) const;

    // Get neuron ID for an acute vertex detector
    NeuronId getAcuteVertexNeuron(size_t x, size_t y, AcuteVertexType type) const;

    // ========================================
    // Statistics
    // ========================================

    size_t getActiveRetinaCount() const;
    size_t getActiveBoundaryCount() const;
    size_t getActiveLineCount() const;
    size_t getActiveCornerCount() const;
    size_t getActiveAcuteVertexCount() const;

    // ========================================
    // Feature Counts (for Geometric Invariance)
    // ========================================

    // Count active features by type (position-invariant)
    size_t countCornersByType(CornerType type) const;
    size_t countBoundariesByType(BoundaryType type) const;
    size_t countAcuteVerticesByType(AcuteVertexType type) const;

    // Total acute vertices (all types combined)
    size_t countTotalAcuteVertices() const;

private:
    Network& network_;

    // Layer 1: Retina neurons (64x64 = 4096)
    std::vector<NeuronId> retinaNeurons_;
    std::vector<bool> retinaState_;  // Current activation state

    // Layer 2: Boundary detector neurons (64x64x4 = 16384)
    std::vector<NeuronId> boundaryNeurons_;

    // Layer 3: Line integrator neurons
    std::vector<NeuronId> lineNeurons_;

    // Layer 3: Corner detector neurons (geometric features)
    std::vector<NeuronId> cornerNeurons_;

    // Layer 3: Acute vertex detector neurons (triangle-like angles)
    std::vector<NeuronId> acuteVertexNeurons_;

    // Current image buffer
    std::vector<uint8_t> currentImage_;

    // ========================================
    // Internal Methods
    // ========================================

    // Build the neural layers
    void buildRetina();
    void buildBoundaryDetectors();
    void buildLineIntegrators();
    void buildCornerDetectors();
    void buildAcuteVertexDetectors();

    // Wire the layers together
    void wireRetinaToBoundary();
    void wireBoundaryToLine();
    void wireBoundaryToCorner();
    void wireBoundaryToAcuteVertex();

    // Process retina input (convert image to spikes)
    void processRetina();

    // Helper: Convert 2D coordinates to 1D index
    static size_t idx(size_t x, size_t y) { return y * RETINA_WIDTH + x; }
    static size_t boundaryIdx(size_t x, size_t y, BoundaryType type) {
        return (y * RETINA_WIDTH + x) * NUM_BOUNDARY_TYPES + static_cast<size_t>(type);
    }
    static size_t cornerIdx(size_t x, size_t y, CornerType type) {
        return (y * RETINA_WIDTH + x) * NUM_CORNER_TYPES + static_cast<size_t>(type);
    }
    static size_t acuteVertexIdx(size_t x, size_t y, AcuteVertexType type) {
        return (y * RETINA_WIDTH + x) * NUM_ACUTE_VERTEX_TYPES + static_cast<size_t>(type);
    }

    // Check if coordinates are valid
    static bool isValid(int x, int y) {
        return x >= 0 && x < static_cast<int>(RETINA_WIDTH) &&
               y >= 0 && y < static_cast<int>(RETINA_HEIGHT);
    }
};

}  // namespace bpagi
