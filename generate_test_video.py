"""
Generate a synthetic test video for testing the Smartbin pipeline.

Creates a short video with:
- 2 seconds of static background (trigger should stay idle)
- 3 seconds with a moving coloured rectangle (trigger should fire, YOLO detects)
- 2 seconds of static background (idle timeout → finalize)

Usage:
    python generate_test_video.py
    python main.py --source test_video.avi --show
"""

import cv2
import numpy as np

OUTPUT = "test_video.avi"
WIDTH, HEIGHT = 640, 480
FPS = 15

# Total: 2s static + 3s motion + 2s static = 7 seconds
STATIC_FRAMES = int(2 * FPS)
MOTION_FRAMES = int(3 * FPS)
TAIL_FRAMES = int(2 * FPS)


def main():
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(OUTPUT, fourcc, FPS, (WIDTH, HEIGHT))

    bg_color = (40, 40, 40)  # Dark grey background

    # Phase 1: Static background
    print(f"Writing {STATIC_FRAMES} static frames...")
    for _ in range(STATIC_FRAMES):
        frame = np.full((HEIGHT, WIDTH, 3), bg_color, dtype=np.uint8)
        writer.write(frame)

    # Phase 2: Moving hand + item (simulates a person holding an item approaching the bin)
    print(f"Writing {MOTION_FRAMES} motion frames...")
    obj_w, obj_h = 120, 100
    hand_w, hand_h = 90, 90
    for i in range(MOTION_FRAMES):
        frame = np.full((HEIGHT, WIDTH, 3), bg_color, dtype=np.uint8)

        # Object & hand slide from left to center
        progress = i / MOTION_FRAMES
        x = int(50 + progress * (WIDTH // 2 - 50))
        y = HEIGHT // 2 - obj_h // 2

        # Draw hand (skin tone: BGR 120, 150, 200) holding the item
        cv2.rectangle(frame, (x - 40, y + 20), (x - 40 + hand_w, y + 20 + hand_h), (120, 150, 200), -1)
        cv2.putText(
            frame, "HAND", (x - 30, y + 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
        )

        # Draw a colourful "item" (bottle-like shape)
        cv2.rectangle(frame, (x, y), (x + obj_w, y + obj_h), (0, 180, 255), -1)
        cv2.rectangle(frame, (x + 40, y - 30), (x + 80, y), (0, 140, 200), -1)

        # Add text label
        cv2.putText(
            frame, "ITEM", (x + 15, y + 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )

        writer.write(frame)

    # Phase 3: Static background again (item removed)
    print(f"Writing {TAIL_FRAMES} tail static frames...")
    for _ in range(TAIL_FRAMES):
        frame = np.full((HEIGHT, WIDTH, 3), bg_color, dtype=np.uint8)
        writer.write(frame)

    writer.release()
    total = STATIC_FRAMES + MOTION_FRAMES + TAIL_FRAMES
    print(f"Done! Wrote {total} frames ({total/FPS:.1f}s) to {OUTPUT}")
    print(f"\nTest with:  python main.py --source {OUTPUT} --show")


if __name__ == "__main__":
    main()
