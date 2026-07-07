/*
 * BallChaser_TankDrive_RC.ino
 * -------------------------------------------------------
 * Manual TANK DRIVE using the RadioLink T8S transmitter
 * and R8EF receiver (PWM output mode).
 *
 *   LEFT stick up/down  (CH3 on Mode 2) -> LEFT thruster
 *   RIGHT stick up/down (CH2)           -> RIGHT thruster
 *
 * Push both sticks up = forward, both down = reverse,
 * opposite sticks = spin in place. Classic tank drive.
 *
 * SETUP NOTES:
 *   - Make sure the R8EF is in PWM mode (the sketch reads
 *     standard 1000-2000 us servo pulses). Long-press the
 *     receiver button to toggle S.BUS/PPM/PWM — LED should
 *     indicate PWM mode (see RadioLink manual).
 *   - Power the receiver from the Arduino 5V pin.
 *   - Receiver signal wires:
 *       CH2 signal -> D3
 *       CH3 signal -> D2
 *   - If a motor runs backwards, swap its two motor wires
 *     (BM70 thrusters don't care about polarity) or flip
 *     the INVERT flag below.
 *
 * L293D wiring (same as the autonomous sketch):
 *     ENA -> D5 (left PWM),  IN1 -> D4, IN2 -> D7
 *     ENB -> D6 (right PWM), IN3 -> D8, IN4 -> A0
 *     Vcc2 -> 7.4 V battery, Vcc1 -> 5 V, common GND!
 */

// ---------------- Pin definitions ----------------
const int CH_LEFT  = 2;   // receiver CH3 (left stick vertical, Mode 2)
const int CH_RIGHT = 3;   // receiver CH2 (right stick vertical)

const int ENA = 5;   // Left thruster PWM
const int IN1 = 4;
const int IN2 = 7;
const int ENB = 6;   // Right thruster PWM
const int IN3 = 8;
const int IN4 = A0;

// ---------------- Tuning knobs --------------------
const int PULSE_MIN   = 1000;  // us at full down
const int PULSE_MID   = 1500;  // us at stick center
const int PULSE_MAX   = 2000;  // us at full up
const int DEADBAND    = 40;    // us around center ignored (kills jitter)
const int MAX_SPEED   = 255;   // full throttle PWM
const bool INVERT_LEFT  = false; // flip if a side drives backwards
const bool INVERT_RIGHT = false;

// Failsafe: if no valid pulses for this long, kill motors
const unsigned long SIGNAL_TIMEOUT = 500; // ms
unsigned long lastGoodSignal = 0;

void setup() {
  Serial.begin(115200);
  pinMode(CH_LEFT, INPUT);
  pinMode(CH_RIGHT, INPUT);
  pinMode(ENA, OUTPUT); pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(ENB, OUTPUT); pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  stopMotors();
  Serial.println("Tank drive ready. Sticks centered = stop.");
}

void loop() {
  // pulseIn with 25 ms timeout — R8EF frames repeat every ~20 ms
  unsigned long pLeft  = pulseIn(CH_LEFT,  HIGH, 25000UL);
  unsigned long pRight = pulseIn(CH_RIGHT, HIGH, 25000UL);

  bool leftOK  = (pLeft  > 900 && pLeft  < 2100);
  bool rightOK = (pRight > 900 && pRight < 2100);

  if (leftOK && rightOK) {
    lastGoodSignal = millis();

    int leftSpeed  = pulseToSpeed(pLeft);
    int rightSpeed = pulseToSpeed(pRight);
    if (INVERT_LEFT)  leftSpeed  = -leftSpeed;
    if (INVERT_RIGHT) rightSpeed = -rightSpeed;

    driveTank(leftSpeed, rightSpeed);

    Serial.print("L="); Serial.print(pLeft);
    Serial.print("us R="); Serial.print(pRight);
    Serial.print("us -> "); Serial.print(leftSpeed);
    Serial.print(" / "); Serial.println(rightSpeed);
  }
  else if (millis() - lastGoodSignal > SIGNAL_TIMEOUT) {
    // FAILSAFE — transmitter off or out of range
    stopMotors();
  }
}

// Convert a 1000-2000 us pulse to -255..+255 with a center deadband
int pulseToSpeed(unsigned long pulse) {
  int offset = (int)pulse - PULSE_MID;
  if (abs(offset) < DEADBAND) return 0;
  // remove the deadband edge so speed ramps smoothly from 0
  if (offset > 0) offset -= DEADBAND; else offset += DEADBAND;
  long range = (PULSE_MAX - PULSE_MID) - DEADBAND; // ~460
  int speed = (int)((long)offset * MAX_SPEED / range);
  return constrain(speed, -MAX_SPEED, MAX_SPEED);
}

// ------------- Motor helpers -------------
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
