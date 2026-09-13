# SmartBin AI v2 — Raspberry Pi 5 & Camera Module 3 Setup

## 1. Operating System Preparation
- Install **Raspberry Pi OS (64-bit, Bookworm)** using Raspberry Pi Imager.
- Ensure 64-bit kernel is running:
  ```bash
  uname -m  # Should output 'aarch64'
  ```

---

## 2. Raspberry Pi Camera Module 3 Configuration

The Camera Module 3 uses the Sony IMX708 sensor and relies on the `libcamera` subsystem.

### Verify Camera Ribbon Connection
Connect the CSI ribbon cable to `CAM/DISP0` or `CAM/DISP1` on Raspberry Pi 5. Verify camera detection:
```bash
rpicam-hello -t 2000
```

### Install Picamera2 and Hardware Acceleration Libraries
```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-libcamera python3-opencv libopenblas-dev
```

### Camera Tuning for Bin Hopper
In `smartbin-v2/configs/deployment_config.yaml`:
- Set `exposure_mode: sports` to lock short shutter duration (1/500s) preventing motion blur when objects are tossed in.
- Enable `awb_mode: auto` to adapt to fluorescent or LED lighting inside the bin canopy.

---

## 3. Power and Thermal Management
- Use the official **Raspberry Pi 27W USB-C Power Supply** (5.1V / 5A) to prevent undervoltage throttling during concurrent AI inference and camera streaming.
- Install the official **Raspberry Pi Active Cooler** (heatsink + PWM fan).
- Verify temperatures remain under 65°C:
  ```bash
  vcgencmd measure_temp
  ```
