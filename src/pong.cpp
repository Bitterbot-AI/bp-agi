#include "bpagi/pong.hpp"
#include <cstdlib>
#include <ctime>
#include <algorithm>

namespace bpagi {

PongGame::PongGame()
    : ballX_(WIDTH / 2)
    , ballY_(HEIGHT / 4)
    , ballVelX_(1)
    , ballVelY_(1)
    , paddleX_((WIDTH - PADDLE_WIDTH) / 2)
    , paddleVel_(0)
    , hits_(0)
    , misses_(0)
{
    // Seed random for ball reset variation
    static bool seeded = false;
    if (!seeded) {
        srand(static_cast<unsigned>(time(nullptr)));
        seeded = true;
    }
}

void PongGame::reset() {
    resetBall();
    paddleX_ = (WIDTH - PADDLE_WIDTH) / 2;
    paddleVel_ = 0;
    hits_ = 0;
    misses_ = 0;
}

void PongGame::resetBall() {
    // Start ball at top center with random horizontal direction
    ballX_ = WIDTH / 4 + (rand() % (WIDTH / 2));
    ballY_ = 5;

    // Random horizontal velocity (-1 or +1)
    ballVelX_ = (rand() % 2) ? 1 : -1;

    // Always moving down initially
    ballVelY_ = 1;
}

bool PongGame::step() {
    bool scored = false;

    // Move paddle
    paddleX_ += paddleVel_;

    // Clamp paddle to screen bounds
    if (paddleX_ < 0) paddleX_ = 0;
    if (paddleX_ > WIDTH - PADDLE_WIDTH) paddleX_ = WIDTH - PADDLE_WIDTH;

    // Move ball
    ballX_ += ballVelX_;
    ballY_ += ballVelY_;

    // Handle wall collisions (top, left, right)
    handleWallCollisions();

    // Check for paddle collision or miss
    if (ballY_ >= PADDLE_Y - BALL_SIZE) {
        if (checkPaddleCollision()) {
            // Hit! Bounce back up
            ballVelY_ = -ballVelY_;
            ballY_ = PADDLE_Y - BALL_SIZE - 1;

            // Add some spin based on where ball hit paddle
            int hitPos = ballX_ - paddleX_;
            if (hitPos < PADDLE_WIDTH / 3) {
                ballVelX_ = -1;  // Hit left side, go left
            } else if (hitPos > 2 * PADDLE_WIDTH / 3) {
                ballVelX_ = 1;   // Hit right side, go right
            }
            // Middle hit keeps current direction

            hits_++;
            scored = true;
        } else if (ballY_ >= HEIGHT) {
            // Miss! Reset ball
            misses_++;
            resetBall();
            scored = false;
        }
    }

    // Reset paddle velocity (requires continuous input)
    paddleVel_ = 0;

    return scored;
}

void PongGame::handleWallCollisions() {
    // Left wall
    if (ballX_ <= 0) {
        ballX_ = 0;
        ballVelX_ = -ballVelX_;
    }

    // Right wall
    if (ballX_ >= WIDTH - BALL_SIZE) {
        ballX_ = WIDTH - BALL_SIZE;
        ballVelX_ = -ballVelX_;
    }

    // Top wall
    if (ballY_ <= 0) {
        ballY_ = 0;
        ballVelY_ = -ballVelY_;
    }
}

bool PongGame::checkPaddleCollision() {
    // Check if ball overlaps with paddle
    return (ballX_ + BALL_SIZE > paddleX_ &&
            ballX_ < paddleX_ + PADDLE_WIDTH &&
            ballY_ + BALL_SIZE >= PADDLE_Y);
}

void PongGame::moveLeft() {
    paddleVel_ = -2;  // Move 2 pixels per tick
}

void PongGame::moveRight() {
    paddleVel_ = 2;
}

void PongGame::setPaddleVelocity(int vel) {
    paddleVel_ = std::max(-3, std::min(3, vel));
}

float PongGame::getHitRate() const {
    int total = hits_ + misses_;
    if (total == 0) return 0.0f;
    return static_cast<float>(hits_) / total;
}

std::vector<uint8_t> PongGame::getRetinaImage() const {
    std::vector<uint8_t> image(WIDTH * HEIGHT, 0);

    // Draw paddle (white bar at y=60)
    for (int x = paddleX_; x < paddleX_ + PADDLE_WIDTH && x < WIDTH; x++) {
        if (x >= 0) {
            image[PADDLE_Y * WIDTH + x] = 255;
            // Make paddle 2 pixels tall for visibility
            if (PADDLE_Y + 1 < HEIGHT) {
                image[(PADDLE_Y + 1) * WIDTH + x] = 255;
            }
        }
    }

    // Draw ball (2x2 white dot)
    for (int dy = 0; dy < BALL_SIZE; dy++) {
        for (int dx = 0; dx < BALL_SIZE; dx++) {
            int px = ballX_ + dx;
            int py = ballY_ + dy;
            if (px >= 0 && px < WIDTH && py >= 0 && py < HEIGHT) {
                image[py * WIDTH + px] = 255;
            }
        }
    }

    return image;
}

}  // namespace bpagi
