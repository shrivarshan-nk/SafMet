#include <SoftwareSerial.h>

SoftwareSerial BTSerial(10, 11);  // TX, RX for HC-05 Bluetooth

// Pin definitions
int forcePin = A1;       // FSR connected to A1
int mqAnalogPin = A0;    // MQ-3 analog pin to A0
const int threshold = 610;

// Variables
int forceValue = 0;      // Stores the force sensor value
int mqAnalogValue = 3;   // Stores the MQ-3 analog value
int forceThreshold = 100;  // Adjust this based on FSR sensitivity

void setup() {
  // Initialize Bluetooth
  BTSerial.begin(9600);  // Start Bluetooth communication via HC-05

  // Initialize Serial for debugging
  Serial.begin(9600);
}

void loop() {
  // Read force sensor
  forceValue = analogRead(forcePin);

  // Read MQ-3 sensor
  mqAnalogValue = analogRead(mqAnalogPin);

  // Check conditions: FSR must be pressed and MQ-3 must not detect alcohol
  if (forceValue > forceThreshold && mqAnalogValue <= threshold) {
    // Force is detected and no alcohol is detected
    BTSerial.println("OK");   
    Serial.println(mqAnalogValue); // Send OK via Bluetooth
    Serial.println("OK");     // Print OK to serial monitor
  } else {
    if (forceValue < forceThreshold) {
      BTSerial.println("Not OK, Not wearing helmet"); 
      Serial.println(mqAnalogValue);   // Send Not OK via Bluetooth
      Serial.println("Not OK, not wearing helmet");  
    } else {
      BTSerial.println("Not OK, Alcohol detected"); 
      Serial.println(mqAnalogValue);   // Send MQ-3 value via Bluetooth
      Serial.println("Not OK, Alcohol Detected");  
    }
  }

  // Delay before next reading
  delay(1000);
}
