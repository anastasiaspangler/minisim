/*
  Standalone battery test:
  UNO R4 + DRV8833 + Adafruit 4639 encoder motor + SG90 micro servo
  Wiring:
    DRV8833 AIN1 -> Arduino D5
    DRV8833 AIN2 -> Arduino D6
    DRV8833 SLP  -> Arduino 5V
    Encoder yellow -> Arduino D2
    Encoder green  -> Arduino D3
    Encoder black  -> Arduino 5V
    Encoder blue   -> Arduino GND
    Motor red/white -> DRV8833 AOUT1/AOUT2
    SG90 signal (orange) -> Arduino D9
    SG90 power  (red)    -> Arduino 5V
    SG90 ground (brown)  -> Arduino GND
  LED behavior:
    Solid ON      = motor running forward
    Blinking      = motor running reverse
    OFF           = paused
  Servo behavior:
    Continuously sweeps 70°–110° (±20° wave around center)
    regardless of motor state
*/
#include <Servo.h>

Servo myservo;

const int MOTOR_IN1   = 5;
const int MOTOR_IN2   = 6;
const int ENCODER_A   = 2;
const int ENCODER_B   = 3;
const int STATUS_LED  = LED_BUILTIN;
const int SERVO_PIN   = 9;

const int MOTOR_SPEED        = 250;   // 0–255
const unsigned long STARTUP_DELAY_MS   = 1500;
const unsigned long RUN_TIME_MS        = 5000;
const unsigned long PAUSE_TIME_MS      = 500;
const unsigned long RAMP_TIME_MS       = 500;
const unsigned long BLINK_TIME_MS      = 200;

// Servo wave parameters
const int   SERVO_CENTER     = 90;   // degrees — neutral position
const int   SERVO_AMPLITUDE  = 20;   // degrees each side (waves 70°–110°)
const unsigned long SERVO_PERIOD_MS = 1200; // full back-and-forth cycle time

volatile long encoderCount = 0;

enum MotorState {
  STARTUP,
  FORWARD,
  PAUSE_AFTER_FORWARD,
  REVERSE,
  PAUSE_AFTER_REVERSE
};

MotorState state        = STARTUP;
unsigned long stateStartTime = 0;
unsigned long lastBlinkTime  = 0;
bool ledState = false;

// ── Encoder ISR ──────────────────────────────────────────────────────────────
void encoderISR() {
  if (digitalRead(ENCODER_A) == digitalRead(ENCODER_B)) {
    encoderCount++;
  } else {
    encoderCount--;
  }
}

// ── Motor helpers ─────────────────────────────────────────────────────────────
void motorStop() {
  analogWrite(MOTOR_IN1, 0);
  analogWrite(MOTOR_IN2, 0);
}
void motorForward(int speedValue) {
  analogWrite(MOTOR_IN1, speedValue);
  analogWrite(MOTOR_IN2, 0);
}
void motorReverse(int speedValue) {
  analogWrite(MOTOR_IN1, 0);
  analogWrite(MOTOR_IN2, speedValue);
}
int rampedSpeed(unsigned long elapsedTime) {
  if (elapsedTime >= RAMP_TIME_MS) return MOTOR_SPEED;
  return (int)((elapsedTime * MOTOR_SPEED) / RAMP_TIME_MS);
}

// ── State transition ──────────────────────────────────────────────────────────
void changeState(MotorState newState) {
  state          = newState;
  stateStartTime = millis();
  lastBlinkTime  = millis();
  motorStop();
  digitalWrite(STATUS_LED, LOW);
  ledState = false;
}

// ── Servo wave (call every loop iteration) ────────────────────────────────────
// Uses a sine approximation via millis() so it never blocks.
void updateServo() {
  // Map current time into a 0–period phase, then to –1..+1 triangle wave.
  // Triangle wave is gentler on a small servo than a hard back-and-forth.
  unsigned long phase = millis() % SERVO_PERIOD_MS;          // 0 … PERIOD-1
  int halfPeriod = SERVO_PERIOD_MS / 2;

  int offset;
  if (phase < (unsigned long)halfPeriod) {
    // Rising half: 0 → +AMPLITUDE
    offset = (int)(((long)phase * SERVO_AMPLITUDE) / halfPeriod);
  } else {
    // Falling half: +AMPLITUDE → -AMPLITUDE
    offset = (int)(((long)(SERVO_PERIOD_MS - phase) * SERVO_AMPLITUDE) / halfPeriod)
             - SERVO_AMPLITUDE;
  }

  myservo.write(SERVO_CENTER + offset);  // stays within [70°, 110°]
}

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
  pinMode(MOTOR_IN1,  OUTPUT);
  pinMode(MOTOR_IN2,  OUTPUT);
  pinMode(ENCODER_A,  INPUT_PULLUP);
  pinMode(ENCODER_B,  INPUT_PULLUP);
  pinMode(STATUS_LED, OUTPUT);

  motorStop();
  digitalWrite(STATUS_LED, LOW);

  myservo.attach(SERVO_PIN);
  myservo.write(SERVO_CENTER);   // park at center before loop starts

  attachInterrupt(digitalPinToInterrupt(ENCODER_A), encoderISR, CHANGE);

  stateStartTime = millis();
}

// ── Loop ──────────────────────────────────────────────────────────────────────
void loop() {
  // Servo runs continuously, every iteration, regardless of motor state
  updateServo();

  unsigned long now     = millis();
  unsigned long elapsed = now - stateStartTime;

  switch (state) {
    case STARTUP:
      motorStop();
      digitalWrite(STATUS_LED, LOW);
      if (elapsed >= STARTUP_DELAY_MS) {
        encoderCount = 0;
        changeState(FORWARD);
      }
      break;

    case FORWARD:
      motorForward(rampedSpeed(elapsed));
      digitalWrite(STATUS_LED, HIGH);
      if (elapsed >= RUN_TIME_MS) {
        changeState(PAUSE_AFTER_FORWARD);
      }
      break;

    case PAUSE_AFTER_FORWARD:
      motorStop();
      digitalWrite(STATUS_LED, LOW);
      if (elapsed >= PAUSE_TIME_MS) {
        changeState(REVERSE);
      }
      break;

    case REVERSE:
      motorReverse(rampedSpeed(elapsed));
      if (now - lastBlinkTime >= BLINK_TIME_MS) {
        ledState = !ledState;
        digitalWrite(STATUS_LED, ledState);
        lastBlinkTime = now;
      }
      if (elapsed >= RUN_TIME_MS) {
        changeState(PAUSE_AFTER_REVERSE);
      }
      break;

    case PAUSE_AFTER_REVERSE:
      motorStop();
      digitalWrite(STATUS_LED, LOW);
      if (elapsed >= PAUSE_TIME_MS) {
        changeState(FORWARD);
      }
      break;
  }
}