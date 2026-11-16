# VLA-BOT: Modular Vision-Language-Action Mobile Manipulator

A real-time **Physical AI Digital Twin** built in **MuJoCo 4.0** and **OpenCV**, translating natural-language commands into closed-loop vision localization, task planning, damped-least-squares 3D inverse kinematics, and predictive human safety control.

## 5-Stage System Architecture

```text
[Natural Language / Multi-Step Queue]  ──►  Semantic Parser (JSON Goal + Confidence)
                                                       │
[Overhead RGB-D Cam / Live Webcam]     ──►  OpenCV HSV + Pinhole 3D Localization
                                                       │
[2D Worker Motion Stream]              ──►  Constant-Velocity Kalman Filter (0.7s / 1.4s)
                                                       ▼
                                     Hierarchical Task & Recovery Planner
                                       ├── AMR Base Waypoint Controller + SSM Safety
                                       └── 4-DOF Arm Jacobian IK (MuJoCo) + Inspection
                                                       │
                                                       ▼
                                     Hardware Control Bridge (Raspberry Pi / ESP32 Ready)