#include <LiquidCrystal.h>

LiquidCrystal lcd(12, 11, 5, 4, 3, 2);
const int analogPin = A0;
const int greenLED = 8;
const int redLED = 9;
const int FULL_THRESHOLD = 900;

void setup() {
  pinMode(greenLED, OUTPUT);
  pinMode(redLED, OUTPUT);
  lcd.begin(16, 2);
  lcd.print("Power Saving...");
}

void loop() {
  int sensorValue = analogRead(analogPin);

  if (sensorValue < FULL_THRESHOLD) {
    digitalWrite(redLED, HIGH);
    digitalWrite(greenLED, LOW);
    lcd.setCursor(0, 0);
    lcd.print("Power Saving... ");
    lcd.setCursor(0, 1);
    lcd.print("Charging        ");
  } else {
    digitalWrite(redLED, LOW);
    digitalWrite(greenLED, HIGH);
    lcd.setCursor(0, 0);
    lcd.print("Power Saving    ");
    lcd.setCursor(0, 1);
    lcd.print("Complete!       ");
  }

  delay(300);
}
 