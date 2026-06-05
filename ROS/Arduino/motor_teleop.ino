#include <Servo.h>

Servo servo;

const int MOTOR_IN1 = 5;
const int MOTOR_IN2 = 6;
const int ENCODER_A = 2;
const int ENCODER_B = 3;
const int SERVO_PIN = 9;

const int SPEED = 250;
const int VACUUM_ON_ANGLE = 70;
const int VACUUM_OFF_ANGLE = 90;
const unsigned long TELEMETRY_PERIOD_MS = 50;

volatile long encoderCount = 0;
bool vacuumOn = false;
unsigned long lastTelemetryAt = 0;

void encoderISR() {
  if (digitalRead(ENCODER_A) == digitalRead(ENCODER_B)) {
    encoderCount++;
  } else {
    encoderCount--;
  }
}

void stopMotor() {
  analogWrite(MOTOR_IN1, 0);
  analogWrite(MOTOR_IN2, 0);
}

void forwardMotor() {
  analogWrite(MOTOR_IN1, SPEED);
  analogWrite(MOTOR_IN2, 0);
}

void reverseMotor() {
  analogWrite(MOTOR_IN1, 0);
  analogWrite(MOTOR_IN2, SPEED);
}

void setVacuum(bool on) {
  vacuumOn = on;
  servo.write(vacuumOn ? VACUUM_ON_ANGLE : VACUUM_OFF_ANGLE);
}

void toggleVacuum() {
  setVacuum(!vacuumOn);
}

void homeRail() {
  stopMotor();
  noInterrupts();
  encoderCount = 0;
  interrupts();
}

void printTelemetry() {
  noInterrupts();
  long count = encoderCount;
  interrupts();

  Serial.print("ENC:");
  Serial.print(count);
  Serial.print(",VAC:");
  Serial.print(vacuumOn ? 1 : 0);
  Serial.print(",ANG:");
  Serial.println(vacuumOn ? VACUUM_ON_ANGLE : VACUUM_OFF_ANGLE);
}

void setup() {
  pinMode(MOTOR_IN1, OUTPUT);
  pinMode(MOTOR_IN2, OUTPUT);
  pinMode(ENCODER_A, INPUT_PULLUP);
  pinMode(ENCODER_B, INPUT_PULLUP);

  stopMotor();
  servo.attach(SERVO_PIN);
  setVacuum(false);

  attachInterrupt(digitalPinToInterrupt(ENCODER_A), encoderISR, CHANGE);

  Serial.begin(115200);
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == 'F') forwardMotor();
    if (c == 'R') reverseMotor();
    if (c == 'S') stopMotor();
    if (c == 'V') toggleVacuum();
    if (c == 'H') homeRail();
  }

  unsigned long now = millis();
  if (now - lastTelemetryAt >= TELEMETRY_PERIOD_MS) {
    lastTelemetryAt = now;
    printTelemetry();
  }
}
