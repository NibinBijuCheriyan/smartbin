# SmartBin AI v2 — Arduino & Servo Actuation Wiring Guide

## 1. Electrical Schematic Overview

```
+-------------------------------------------------------------+
|                      POWER DISTRIBUTION                      |
|                                                             |
|   +5V External Power (2A+) -------> Servo VCC (Red Wire)    |
|   GND External Power       -------> Servo GND (Black Wire)  |
|                                  |                          |
|                                  +-> Arduino GND Pin        |
|                                                             |
+-------------------------------------------------------------+

+-------------------------------------------------------------+
|                     ARDUINO CONTROL SIGNALS                 |
|                                                             |
|   Arduino Pin 9 (PWM)   ----------> Servo Signal (Orange)   |
|   Arduino Pin 13 (LED)  ----------> Green Status LED (Power)|
|   Arduino Pin 8 (LED)   ----------> Amber Busy LED (Sorting)|
|                                                             |
+-------------------------------------------------------------+

+-------------------------------------------------------------+
|                    HOST COMMUNICATIONS                      |
|                                                             |
|   Raspberry Pi USB Port <=========> Arduino USB Port (Serial)|
|                                     (/dev/ttyACM0 @ 115200) |
+-------------------------------------------------------------+
```

> [!CAUTION]
> **Servo Power Isolation**:
> Never power high-torque servos directly from the Arduino 5V pin or Raspberry Pi GPIO header. Servos draw up to 1.5A peak stall current, which will trigger brownout resets on the Raspberry Pi. Always use an external regulated 5V/3A DC power supply with a common ground shared with the Arduino.

---

## 2. Flap Angles Calibration

The physical sorting divider or rotating chute aligns with 4 compartments:

| Compartment ID | Waste Stream | Target Servo Angle | Function |
|:---:|:---|:---:|:---|
| **Home** | Neutral Standby | **90°** | Closed hopper position |
| **1** | **Recyclable** | **30°** | Deflects dry recyclables (cans, bottles, cardboard) |
| **2** | **Compost** | **90°** (Open) | Opens compost trapdoor for food/organic waste |
| **3** | **Landfill** | **150°** | Deflects non-recyclable inert trash |
| **4** | **Reject** | **0°** | Diverts contaminated items or unconfident predictions |

---

## 3. Serial Packet Protocol

The Python `ArduinoSerialBridge` communicates via ASCII lines at **115200 baud**:
- **Host to Arduino**:
  - `SORT:<id>\n` -> Trigger sorting compartment (e.g. `SORT:1\n`)
  - `PING\n` -> Heartbeat query
  - `RESET\n` -> Immediately return servo to 90° home position
- **Arduino to Host**:
  - `{"status":"ACTUATING","compartment":1,"angle":30}\n`
  - `{"status":"SORT_COMPLETED","compartment":1}\n`
  - `{"status":"PONG","busy":false}\n`
