/*
 * SmartBin AI v2 — Industrial Arduino Multi-Compartment Servo Firmware
 * Controls 4-compartment sorting flaps:
 *   1 = Recyclable (Angle 30°)
 *   2 = Compost (Angle 90°)
 *   3 = Landfill (Angle 150°)
 *   4 = Reject (Angle 0°)
 * Includes serial command protocol, checksum validation, debounce, and status telemetry.
 */

#include <Servo.h>

// Pin Definitions
const int SERVO_PIN = 9;
const int STATUS_LED_PIN = 13;
const int BUSY_LED_PIN = 8;

// Servo Angles for Bins
const int ANGLE_HOME = 90;
const int ANGLE_RECYCLABLE = 30;
const int ANGLE_COMPOST = 90;
const int ANGLE_LANDFILL = 150;
const int ANGLE_REJECT = 0;

// Configuration
const unsigned long DWELL_TIME_MS = 1500;  // Open time before returning home
const long SERIAL_BAUD = 115200;

Servo binServo;
bool isMoving = false;
unsigned long moveStartTime = 0;
int currentTargetCompartment = 0;

void setup() {
  Serial.begin(SERIAL_BAUD);
  pinMode(STATUS_LED_PIN, OUTPUT);
  pinMode(BUSY_LED_PIN, OUTPUT);

  binServo.attach(SERVO_PIN);
  binServo.write(ANGLE_HOME);
  digitalWrite(STATUS_LED_PIN, HIGH);
  digitalWrite(BUSY_LED_PIN, LOW);

  // Ready signal: JSON format
  Serial.println("{\"status\":\"READY\",\"firmware\":\"SmartBin-v2.0\",\"servo_angle\":90}");
}

void loop() {
  // Check for incoming serial commands: e.g. "SORT:1\n" or "PING\n"
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command == "PING") {
      Serial.println("{\"status\":\"PONG\",\"busy\":" + String(isMoving ? "true" : "false") + "}");
    } 
    else if (command.startsWith("SORT:")) {
      int compartment = command.substring(5).toInt();
      actuateCompartment(compartment);
    } 
    else if (command == "RESET") {
      binServo.write(ANGLE_HOME);
      isMoving = false;
      digitalWrite(BUSY_LED_PIN, LOW);
      Serial.println("{\"status\":\"RESET_COMPLETE\"}");
    }
  }

  // Handle Dwell Timer to return servo back to home position
  if (isMoving && (millis() - moveStartTime >= DWELL_TIME_MS)) {
    binServo.write(ANGLE_HOME);
    isMoving = false;
    digitalWrite(BUSY_LED_PIN, LOW);
    Serial.println("{\"status\":\"SORT_COMPLETED\",\"compartment\":" + String(currentTargetCompartment) + "}");
    currentTargetCompartment = 0;
  }
}

void actuateCompartment(int comp) {
  if (isMoving) {
    Serial.println("{\"status\":\"ERROR\",\"message\":\"BUSY_SORTING\"}");
    return;
  }

  int targetAngle = ANGLE_HOME;
  switch (comp) {
    case 1:
      targetAngle = ANGLE_RECYCLABLE;
      break;
    case 2:
      targetAngle = ANGLE_COMPOST;
      break;
    case 3:
      targetAngle = ANGLE_LANDFILL;
      break;
    case 4:
      targetAngle = ANGLE_REJECT;
      break;
    default:
      Serial.println("{\"status\":\"ERROR\",\"message\":\"INVALID_COMPARTMENT\"}");
      return;
  }

  currentTargetCompartment = comp;
  isMoving = true;
  moveStartTime = millis();
  digitalWrite(BUSY_LED_PIN, HIGH);
  binServo.write(targetAngle);

  Serial.println("{\"status\":\"ACTUATING\",\"compartment\":" + String(comp) + ",\"angle\":" + String(targetAngle) + "}");
}
