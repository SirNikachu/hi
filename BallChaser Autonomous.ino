/*
 * BallChaser_Autonomous.ino
 * -------------------------------------------------------
 * Autonomous ball-tracking boat
 *   - PixyCam CMUcam5 (SPI via ICSP header) finds the ball
 *   - Proportional steering drives two BM70 thrusters
 *     through an L293D H-bridge (differential / tank style)
 *   - JSN-SR04T waterproof ultrasonic confirms how close
 *     the ball is and triggers the "arrival" behavior
 *
 * BEFORE UPLOADING:
 *   1. Install the "Pixy" library (Arduino IDE > Library Manager,
 *      or from pixycam.com). This is for the ORIGINAL Pixy/CMUcam5.
 *   2. Teach Pixy your ball colors using PixyMon (signatures 1-7).
 *   3. Check the pin numbers below against your breadboard.
 *
 * WIRING (matches your MITRE diagram, adjust if needed):
 *   L293D:
 *     ENA (pin 1)  -> D5  (PWM, left thruster speed)
 *     IN1 (pin 2)  -> D4
 *     IN2 (pin 7)  -> D7
 *     ENB (pin 9)  -> D6  (PWM, right thruster speed)
 *     IN3 (pin 10) -> D8
 *     IN4 (pin 15) -> A0 (used as digital output)
 *     Vcc2 (pin 8) -> 7.4 V battery +
 *     Vcc1 (pin 16)-> 5 V
 *     GNDs         -> common ground (battery AND Arduino!)
 *   JSN-SR04T:
 *     TRIG -> A1
 *     ECHO -> A2
 *     5V / GND -> Arduino 5V / GND
 *   PixyCam:
 *     Ribbon cable to the Uno ICSP header (uses SPI: D11-D13)
 *     -> that's why D11 is NOT used for the L293D.
 */

#include <SPI.h>
#include <Pixy.h>

Pixy pixy;

// ---------------- Pin definitions ----------------
const int ENA = 5;   // Left thruster PWM
const int IN1 = 4;
const int IN2 = 7;
const int ENB = 6;   // Right thruster PWM
const int IN3 = 8;
const int IN4 = A0;

const int TRIG = A1;
const int ECHO = A2;

// ---------------- Tuning knobs --------------------
const int  BASE_SPEED    = 140;  // cruise PWM (0-255)
const int  MAX_SPEED     = 220;  // cap so thrusters don't cavitate
const int  SEARCH_SPEED  = 110;  // spin speed while looking for a ball
const float KP           = 0.9;  // steering gain (raise = snappier turns)
const int  ARRIVE_CM     = 25;   // "we reached the ball" distance
const int  SLOW_CM       = 80;   // start slowing down inside this range
const unsigned long LOST_TIMEOUT = 1500; // ms without ball before searching

// Pixy image is 320 px wide; center x = 160
const int PIXY_CENTER_X = 160;

unsigned long lastSeen = 0;
int lastErrorSign = 1;   // remember which way the ball went off-screen

void setup() {
  Serial.begin(115200);
  pinMode(ENA, OUTPUT); pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(ENB, OUTPUT); pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  pinMode(TRIG, OUTPUT); pinMode(ECHO, INPUT);
  stopMotors();
  pixy.init();
  Serial.println("BallChaser ready.");
}

void loop() {
  int blocks = pixy.getBlocks();

  if (blocks) {
    // ---- Pick the biggest block (closest ball) ----
    int best = 0;
    long bestArea = 0;
    for (int i = 0; i < blocks; i++) {
      long area = (long)pixy.blocks[i].width * pixy.blocks[i].height;
      if (area > bestArea) { bestArea = area; best = i; }
    }

    int x = pixy.blocks[best].x;          // 0..319
    int error = x - PIXY_CENTER_X;        // negative = ball is left
    lastErrorSign = (error >= 0) ? 1 : -1;
    lastSeen = millis();

    // ---- Ultrasonic range check ----
    int dist = readDistanceCM();

    if (dist > 0 && dist <= ARRIVE_CM) {
      arrivedAtBall();
      return;
    }

    // Slow down as we approach so we don't plow the ball under
    int speed = BASE_SPEED;
    if (dist > 0 && dist < SLOW_CM) {
      speed = map(dist, ARRIVE_CM, SLOW_CM, 70, BASE_SPEED);
    }

    // ---- Proportional differential steering ----
    int correction = (int)(KP * error);
    int leftSpeed  = constrain(speed + correction, -MAX_SPEED, MAX_SPEED);
    int rightSpeed = constrain(speed - correction, -MAX_SPEED, MAX_SPEED);
    driveTank(leftSpeed, rightSpeed);

    Serial.print("x="); Serial.print(x);
    Serial.print(" err="); Serial.print(error);
    Serial.print(" dist="); Serial.println(dist);
  }
  else if (millis() - lastSeen > LOST_TIMEOUT) {
    // ---- Lost the ball: rotate in place toward last known side ----
    driveTank(SEARCH_SPEED * lastErrorSign, -SEARCH_SPEED * lastErrorSign);
  }
  // else: brief dropout — keep last command for smoother motion
}

// Called when the ultrasonic says the ball is right in front of us.
// Default: nudge forward to bump/collect it, then stop briefly.
void arrivedAtBall() {
  Serial.println(">>> Ball reached!");
  driveTank(90, 90);
  delay(600);
  stopMotors();
  delay(400);
  lastSeen = 0;  // force search mode so we go find the next one
}

// ------------- Motor helpers -------------
// Positive = forward, negative = reverse, per side.
void driveTank(int left, int right) {
  setMotor(IN1, IN2, ENA, left);
  setMotor(IN3, IN4, ENB, right);
}

void setMotor(int inA, int inB, int en, int speed) {
  if (speed >= 0) { digitalWrite(inA, HIGH); digitalWrite(inB, LOW); }
  else            { digitalWrite(inA, LOW);  digitalWrite(inB, HIGH); speed = -speed; }
  analogWrite(en, constrain(speed, 0, 255));
}

void stopMotors() {
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);
}

// ------------- Ultrasonic (JSN-SR04T) -------------
// Returns distance in cm, or -1 if no echo.
// Note: the SR04T has a ~20 cm minimum range blind zone.
int readDistanceCM() {
  digitalWrite(TRIG, LOW);  delayMicroseconds(2);
  digitalWrite(TRIG, HIGH); delayMicroseconds(10);
  digitalWrite(TRIG, LOW);
  long duration = pulseIn(ECHO, HIGH, 25000UL); // 25 ms timeout (~4 m)
  if (duration == 0) return -1;
  return (int)(duration / 58);
}
