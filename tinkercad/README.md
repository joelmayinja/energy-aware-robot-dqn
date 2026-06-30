# Tinkercad Physical Demo

This folder holds notes and code for the physical demo companion to the project.

**Important context:** Tinkercad Circuits simulates Arduino-style boards — it does not run
PyTorch or the actual trained DQN/Student model. This folder is for a simplified demo that
*mimics* the controller's behavior using basic rules, for example:

- A servo motor standing in for the robot arm's joint movement
- An LED for grid status (green = stable, red = unstable/brownout)
- A photoresistor (LDR) simulating "solar surplus available"
- A potentiometer simulating "battery reserve"

Once built in Tinkercad, paste the public share link here:

- Tinkercad project link: _(https://www.tinkercad.com/things/1190rCJ2dVj/editel?returnTo=%2Fdashboard%2Fdesigns%2Fcircuits)_
- Arduino sketch: `demo_sketch.ino` _(<OTIMIZING ENERGY THROUGH DEEP Q LEARNING.png>)_
