#include <Wire.h>                   // Include Wire library for I2C
#include <LiquidCrystal_I2C.h>      // Include the I2C LCD library

// Set the I2C address of the LCD (usually 0x27 or 0x3F)
LiquidCrystal_I2C lcd(0x27, 16, 2); // Change 0x27 to your LCD's I2C address if needed

#include <SoftwareSerial.h>
SoftwareSerial bluetooth(10, 11);   // RX, TX pins for Bluetooth

// Define motor control pins for L298N
const int motorPin1 = 2;            // IN1 on the L298N
const int motorPin2 = 3;            // IN2 on the L298N
const int enablePin = 9;            // ENA on the L298N for speed control (optional)

void setup() {
  Serial.begin(9600);
  lcd.begin(16,2);                      // Initialize the I2C LCD
  lcd.backlight();                  // Turn on the backlight
  bluetooth.begin(9600);            // Start Bluetooth communication
  lcd.print("Waiting for msg");     // Initial message

  // Set motor pins as outputs
  pinMode(motorPin1, OUTPUT);
  pinMode(motorPin2, OUTPUT);
  pinMode(enablePin, OUTPUT);

  // Ensure the motor is off initially
  digitalWrite(motorPin1, LOW);
  digitalWrite(motorPin2, LOW);
  analogWrite(enablePin, 0);       // Set speed to 0 (stop)
}

void loop() {
  // Check if data is available from Bluetooth
  if (bluetooth.available()) {
    String message = bluetooth.readStringUntil('\n'); // Read message until newline
    displayMessage(message);       // Call function to display message
    controlMotor(message);         // Control motor based on message
  }
}

void displayMessage(String msg) {
  lcd.clear();                     // Clear the LCD
  lcd.print(msg);                  // Print the message on the LCD
  Serial.println(msg);
}

bool startsWith(String str, String prefix) {
  return str.substring(0, prefix.length()) == prefix;
}

void controlMotor(String msg) {
  if (startsWith(msg,"OK")) {
    // Rotate the motor (forward direction)
    digitalWrite(motorPin1, HIGH);
    digitalWrite(motorPin2, LOW);
    analogWrite(enablePin, 200);   // Set speed (0-255, adjust if needed)
  } else {
    // Stop the motor
    digitalWrite(motorPin1, LOW);
    digitalWrite(motorPin2, LOW);
    analogWrite(enablePin, 0);     // Set speed to 0
  }
}
