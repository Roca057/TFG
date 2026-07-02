const int pinNicla = D2;   // Input: Connected to Nicla Vision PG1 pin
const int pinRele = D5;  // Output: Connected to Relay IN

void setup() {
  Serial.begin(115200);
  
  pinMode(pinNicla, INPUT);      
  pinMode(pinRele, OUTPUT);    
  pinMode(LED_BUILTIN, OUTPUT);  
  
  // Safe initial state: Everything OFF at boot
  digitalWrite(pinRele, LOW);
  digitalWrite(LED_BUILTIN, HIGH); 
}

void loop() {
  // Check for trigger pulse from Nicla Vision
  if (digitalRead(pinNicla) == HIGH) {
    
    digitalWrite(pinRele, HIGH);    
    digitalWrite(LED_BUILTIN, LOW); 
    
    delay(2800);
    
  } 

// Keep everything OFF while waiting for a pulse
digitalWrite(pinRele, LOW);     
digitalWrite(LED_BUILTIN, HIGH);  

  
  delay(10); 
}
