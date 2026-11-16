"""
VLA-BOT | Vision-Language-Action Mobile Manipulator | MuJoCo Industrial Warehouse Digital Twin

Run:   python src\\vla_warehouse_twin.py
Keys:  type a command + ENTER | 1-4 presets (when box is empty) | TAB shift cube | ` toggle worker | V camera view | W webcam toggle | ESC quit
"""
import json
import math
import random
import re
import textwrap
import time
import traceback

import cv2
import mujoco
import numpy as np

# ----------------------------------------------------------------------------------------------
# 1. EXPANDED INDUSTRIAL WAREHOUSE SCENE (6m x 6m Floor, Racks, Gantry, 3 Stations, AMR + Arm)
# ----------------------------------------------------------------------------------------------
MJCF_XML = """
<mujoco model="industrial_vla_warehouse">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="1024" offheight="768"/>
    <headlight diffuse="0.82 0.82 0.82" ambient="0.42 0.42 0.42" specular="0.20 0.20 0.20"/>
  </visual>
  <asset>
    <texture name="concrete" type="2d" builtin="checker" rgb1="0.16 0.18 0.21" rgb2="0.19 0.21 0.25" width="512" height="512"/>
    <material name="floor_mat" texture="concrete" texrepeat="20 20" reflectance="0.10"/>
    <material name="steel_dark" rgba="0.14 0.17 0.22 1" specular="0.6" shininess="0.7"/>
    <material name="steel_light" rgba="0.72 0.76 0.80 1" specular="0.4" shininess="0.5"/>
    <material name="table_top" rgba="0.88 0.90 0.93 1" specular="0.2" shininess="0.3"/>
    <material name="safety_yellow" rgba="0.95 0.76 0.06 1" specular="0.2"/>
    <material name="amr_chassis" rgba="0.10 0.14 0.20 1" specular="0.6" shininess="0.8"/>
    <material name="amr_cyan" rgba="0.0 0.82 0.92 1" specular="0.7" shininess="0.8"/>
    <material name="wheel_rubber" rgba="0.06 0.06 0.07 1" specular="0.1"/>
    <material name="red_mat" rgba="0.92 0.16 0.16 1" specular="0.2"/>
    <material name="blue_mat" rgba="0.10 0.35 0.96 1" specular="0.2"/>
    <material name="green_mat" rgba="0.14 0.82 0.32 1" specular="0.2"/>
    <material name="hi_vis_vest" rgba="0.98 0.42 0.05 1" specular="0.3"/>
    <material name="denim_pants" rgba="0.15 0.24 0.42 1"/>
    <material name="skin_tone" rgba="0.92 0.76 0.64 1"/>
  </asset>

  <worldbody>
    <light pos="0.1 -0.4 4.2" dir="0 0.12 -1" directional="true"/>
    <geom name="floor" type="plane" size="6 6 0.1" material="floor_mat"/>

    <!-- Overview & Pickup Gantry Cameras -->
    <camera name="main_3d_cam" pos="0.15 -3.05 2.45" xyaxes="1 0 0 0 0.62 0.78" fovy="46"/>
    <camera name="pickup_cam" pos="-1.15 0.0 1.42" xyaxes="0 -1 0 1 0 0" fovy="46"/>

    <!-- EXPANDED FLOOR SAFETY LANE MARKINGS (x: -1.80 to +2.10, y: -1.50 to +1.60) -->
    <geom type="box" pos="0.15 -1.50 0.001" size="1.95 0.025 0.001" material="safety_yellow" contype="0" conaffinity="0"/>
    <geom type="box" pos="0.15  1.60 0.001" size="1.95 0.025 0.001" material="safety_yellow" contype="0" conaffinity="0"/>
    <geom type="box" pos="-1.80 0.05 0.001" size="0.025 1.55 0.001" material="safety_yellow" contype="0" conaffinity="0"/>
    <geom type="box" pos="2.10  0.05 0.001" size="0.025 1.55 0.001" material="safety_yellow" contype="0" conaffinity="0"/>

    <!-- AGV Transit Corridor Guide Lines -->
    <geom type="box" pos="0.25 0.0 0.001" size="1.15 0.015 0.001" rgba="0.0 0.82 0.92 0.35" contype="0" conaffinity="0"/>
    <geom type="box" pos="0.35 -1.05 0.001" size="0.18 0.04 0.001" material="safety_yellow" contype="0" conaffinity="0"/>
    <geom type="box" pos="0.35 -0.70 0.001" size="0.18 0.04 0.001" material="safety_yellow" contype="0" conaffinity="0"/>
    <geom type="box" pos="0.35 -0.35 0.001" size="0.18 0.04 0.001" material="safety_yellow" contype="0" conaffinity="0"/>
    <geom type="box" pos="0.35  0.00 0.001" size="0.18 0.04 0.001" material="safety_yellow" contype="0" conaffinity="0"/>
    <geom type="box" pos="0.35  0.35 0.001" size="0.18 0.04 0.001" material="safety_yellow" contype="0" conaffinity="0"/>
    <geom type="box" pos="0.35  0.70 0.001" size="0.18 0.04 0.001" material="safety_yellow" contype="0" conaffinity="0"/>
    <geom type="box" pos="0.35  1.05 0.001" size="0.18 0.04 0.001" material="safety_yellow" contype="0" conaffinity="0"/>

    <!-- INDUSTRIAL PALLET RACKING ALONG NORTH WALL -->
    <body name="warehouse_rack_1" pos="-0.75 1.90 0.0">
      <geom type="box" pos="-0.65 0 0.60" size="0.025 0.025 0.60" material="steel_dark"/>
      <geom type="box" pos=" 0.65 0 0.60" size="0.025 0.025 0.60" material="steel_dark"/>
      <geom type="box" pos="-0.65 0.28 0.60" size="0.025 0.025 0.60" material="steel_dark"/>
      <geom type="box" pos=" 0.65 0.28 0.60" size="0.025 0.025 0.60" material="steel_dark"/>
      <geom type="box" pos="0 0.14 0.45" size="0.68 0.18 0.015" material="safety_yellow"/>
      <geom type="box" pos="0 0.14 0.95" size="0.68 0.18 0.015" material="safety_yellow"/>
      <geom type="box" pos="-0.35 0.14 0.55" size="0.15 0.12 0.08" rgba="0.25 0.35 0.48 1"/>
      <geom type="box" pos=" 0.15 0.14 0.55" size="0.15 0.12 0.08" rgba="0.30 0.40 0.52 1"/>
      <geom type="box" pos="-0.25 0.14 1.05" size="0.16 0.12 0.08" rgba="0.25 0.35 0.48 1"/>
      <geom type="box" pos=" 0.30 0.14 1.05" size="0.16 0.12 0.08" rgba="0.30 0.40 0.52 1"/>
    </body>
    <body name="warehouse_rack_2" pos="0.85 1.90 0.0">
      <geom type="box" pos="-0.65 0 0.60" size="0.025 0.025 0.60" material="steel_dark"/>
      <geom type="box" pos=" 0.65 0 0.60" size="0.025 0.025 0.60" material="steel_dark"/>
      <geom type="box" pos="-0.65 0.28 0.60" size="0.025 0.025 0.60" material="steel_dark"/>
      <geom type="box" pos=" 0.65 0.28 0.60" size="0.025 0.025 0.60" material="steel_dark"/>
      <geom type="box" pos="0 0.14 0.45" size="0.68 0.18 0.015" material="safety_yellow"/>
      <geom type="box" pos="0 0.14 0.95" size="0.68 0.18 0.015" material="safety_yellow"/>
      <geom type="box" pos="-0.30 0.14 0.55" size="0.15 0.12 0.08" rgba="0.25 0.35 0.48 1"/>
      <geom type="box" pos=" 0.25 0.14 1.05" size="0.16 0.12 0.08" rgba="0.30 0.40 0.52 1"/>
    </body>

    <!-- PICKUP & VISION INSPECTION TABLE (x = -1.15) -->
    <body name="pickup_table" pos="-1.15 0.0 0.34">
      <geom type="box" size="0.28 0.52 0.02" material="table_top"/>
      <geom type="box" pos="0 -0.53 0.015" size="0.29 0.012 0.025" material="safety_yellow"/>
      <geom type="box" pos="0  0.53 0.015" size="0.29 0.012 0.025" material="safety_yellow"/>
      <geom type="box" pos="-0.22 -0.46 -0.17" size="0.025 0.025 0.17" material="steel_dark"/>
      <geom type="box" pos=" 0.22 -0.46 -0.17" size="0.025 0.025 0.17" material="steel_dark"/>
      <geom type="box" pos="-0.22  0.46 -0.17" size="0.025 0.025 0.17" material="steel_dark"/>
      <geom type="box" pos=" 0.22  0.46 -0.17" size="0.025 0.025 0.17" material="steel_dark"/>
      <!-- Overhead Camera Gantry -->
      <geom type="box" pos="-0.24 -0.46 0.45" size="0.018 0.018 0.45" material="steel_light"/>
      <geom type="box" pos="-0.24  0.46 0.45" size="0.018 0.018 0.45" material="steel_light"/>
      <geom type="box" pos="-0.24  0.00 0.90" size="0.02 0.48 0.02" material="steel_dark"/>
    </body>

    <!-- SPACED-OUT DELIVERY STATIONS -->
    <body name="station_a" pos="1.65 -0.95 0.34">
      <geom type="box" size="0.26 0.26 0.02" material="table_top"/>
      <geom type="box" pos="0 0 0.022" size="0.20 0.20 0.002" rgba="0.15 0.55 0.95 0.78" contype="0" conaffinity="0"/>
      <geom type="box" pos="0 0 -0.17" size="0.06 0.06 0.17" material="steel_dark"/>
    </body>
    <body name="station_b" pos="1.65 0.95 0.34">
      <geom type="box" size="0.26 0.26 0.02" material="table_top"/>
      <geom type="box" pos="0 0 0.022" size="0.20 0.20 0.002" rgba="0.62 0.32 0.82 0.78" contype="0" conaffinity="0"/>
      <geom type="box" pos="0 0 -0.17" size="0.06 0.06 0.17" material="steel_dark"/>
    </body>
    <body name="station_reject" pos="-0.45 1.30 0.34">
      <geom type="box" size="0.26 0.26 0.02" material="table_top"/>
      <geom type="box" pos="0 0 0.022" size="0.20 0.20 0.002" rgba="0.95 0.48 0.10 0.78" contype="0" conaffinity="0"/>
      <geom type="box" pos="0 0 -0.17" size="0.06 0.06 0.17" material="steel_dark"/>
    </body>

    <!-- ARTICULATED HUMAN WORKER (Random 2D Patrol) -->
    <body name="human_worker" pos="0 0 0">
      <joint name="h_x" type="slide" axis="1 0 0" range="-4.5 4.5"/>
      <joint name="h_y" type="slide" axis="0 1 0" range="-4.5 4.5"/>
      <geom type="capsule" fromto="0 -0.06 0.04 0 -0.06 0.45" size="0.045" material="denim_pants" contype="0" conaffinity="0"/>
      <geom type="capsule" fromto="0  0.06 0.04 0  0.06 0.45" size="0.045" material="denim_pants" contype="0" conaffinity="0"/>
      <geom type="capsule" fromto="0 0 0.46 0 0 0.82" size="0.11" material="hi_vis_vest" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0 0 0.62" size="0.113 0.025" material="safety_yellow" contype="0" conaffinity="0"/>
      <geom type="capsule" fromto="0 -0.14 0.78 0.04 -0.16 0.50" size="0.035" material="hi_vis_vest" contype="0" conaffinity="0"/>
      <geom type="capsule" fromto="0  0.14 0.78 0.04  0.16 0.50" size="0.035" material="hi_vis_vest" contype="0" conaffinity="0"/>
      <geom type="sphere" pos="0 0 0.96" size="0.085" material="skin_tone" contype="0" conaffinity="0"/>
      <geom type="sphere" pos="0 0 1.00" size="0.095" material="safety_yellow" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0.03 0 0.98" size="0.105 0.01" material="safety_yellow" contype="0" conaffinity="0"/>
    </body>

    <!-- KALMAN PREDICTED WORKER FLOOR MARKERS -->
    <body name="pred_marker_1" pos="0 0 0.003">
      <joint name="pm1_x" type="slide" axis="1 0 0" range="-4.5 4.5"/>
      <joint name="pm1_y" type="slide" axis="0 1 0" range="-4.5 4.5"/>
      <geom type="cylinder" size="0.12 0.002" rgba="1 0.15 0.15 0.55" contype="0" conaffinity="0"/>
    </body>
    <body name="pred_marker_2" pos="0 0 0.003">
      <joint name="pm2_x" type="slide" axis="1 0 0" range="-4.5 4.5"/>
      <joint name="pm2_y" type="slide" axis="0 1 0" range="-4.5 4.5"/>
      <geom type="cylinder" size="0.09 0.002" rgba="1 0.15 0.15 0.35" contype="0" conaffinity="0"/>
    </body>

    <!-- MOBILE MANIPULATOR: INDUSTRIAL AMR + 4-DOF ARM -->
    <body name="amr_base" pos="0 0 0.14">
      <joint name="base_x" type="slide" axis="1 0 0" range="-4.5 4.5"/>
      <joint name="base_y" type="slide" axis="0 1 0" range="-4.5 4.5"/>
      <joint name="base_yaw" type="hinge" axis="0 0 1"/>

      <geom type="box" size="0.24 0.18 0.08" material="amr_chassis"/>
      <geom type="box" pos="0 0 0.082" size="0.22 0.16 0.008" material="amr_cyan"/>
      <geom type="box" pos="0.235 0 -0.05" size="0.01 0.17 0.015" material="safety_yellow"/>
      <geom type="cylinder" pos="0.21 0 0.09" size="0.035 0.025" material="steel_dark"/>
      <geom type="cylinder" pos="0.21 0 0.10" size="0.036 0.006" rgba="1 0.15 0.15 1"/>

      <geom type="cylinder" pos="0.14 0.19 -0.05" zaxis="0 1 0" size="0.085 0.025" material="wheel_rubber"/>
      <geom type="cylinder" pos="0.14 -0.19 -0.05" zaxis="0 1 0" size="0.085 0.025" material="wheel_rubber"/>
      <geom type="cylinder" pos="-0.14 0.19 -0.05" zaxis="0 1 0" size="0.085 0.025" material="wheel_rubber"/>
      <geom type="cylinder" pos="-0.14 -0.19 -0.05" zaxis="0 1 0" size="0.085 0.025" material="wheel_rubber"/>

      <body name="arm_pedestal" pos="0.06 0 0.09">
        <geom type="cylinder" size="0.065 0.05" material="amr_chassis"/>
        <body name="link1" pos="0 0 0.05">
          <joint name="j_yaw" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
          <geom type="cylinder" size="0.045 0.06" material="amr_cyan"/>
          <body name="link2" pos="0 0 0.06">
            <joint name="j_shoulder" type="hinge" axis="0 1 0" range="-1.5 1.8"/>
            <geom type="capsule" fromto="0 0 0 0.30 0 0.16" size="0.03" material="amr_chassis"/>
            <body name="link3" pos="0.30 0 0.16">
              <joint name="j_elbow" type="hinge" axis="0 1 0" range="-2.2 2.2"/>
              <geom type="capsule" fromto="0 0 0 0.28 0 -0.10" size="0.025" material="amr_cyan"/>
              <body name="ee_body" pos="0.28 0 -0.10">
                <joint name="j_wrist" type="hinge" axis="0 1 0" range="-2.0 2.0"/>
                <geom type="box" size="0.022 0.042 0.014" material="amr_chassis"/>
                <geom type="box" pos="0 -0.03 -0.026" size="0.01 0.005 0.024" material="amr_cyan"/>
                <geom type="box" pos="0 0.03 -0.026" size="0.01 0.005 0.024" material="amr_cyan"/>
                <site name="ee_site" pos="0 0 -0.038" size="0.01" rgba="1 1 0 1"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- THREE PAYLOADS ON PICKUP TABLE (x = -1.15) -->
    <body name="cube_red" pos="-1.12 -0.26 0.385">
      <freejoint name="fj_red"/>
      <geom name="g_red" type="box" size="0.025 0.025 0.025" material="red_mat" mass="0.12"/>
    </body>
    <body name="cube_blue" pos="-1.15 0.0 0.385">
      <freejoint name="fj_blue"/>
      <geom name="g_blue" type="box" size="0.025 0.025 0.025" material="blue_mat" mass="0.12"/>
    </body>
    <body name="cube_green" pos="-1.12 0.26 0.385">
      <freejoint name="fj_green"/>
      <geom name="g_green" type="box" size="0.025 0.025 0.025" material="green_mat" mass="0.12"/>
    </body>
  </worldbody>
</mujoco>
"""

# ----------------------------------------------------------------------------------------------
# 2. CONSTANTS (Matched to Expanded 6m x 6m Warehouse Coordinates)
# ----------------------------------------------------------------------------------------------
COLORS = ("red", "blue", "green")
CUBE_START = {"red": (-1.12, -0.26), "blue": (-1.15, 0.0), "green": (-1.12, 0.26)}
CUBE_Z, GRASP_DZ = 0.385, 0.032
PICK_X = -0.59                      # AMR standoff x in front of pickup table (-1.15 + 0.56m reach)
BASE_START = np.array([-0.40, 0.0, math.pi])
STOW = np.array([0.0, -1.0, 1.07, -0.07])
ARM_JOINTS = ("j_yaw", "j_shoulder", "j_elbow", "j_wrist")
IK_LO = np.array([-3.1, -1.5, -2.2, -2.0])
IK_HI = np.array([3.1, 1.8, 2.2, 2.0])

DEST = {
    "zone a": {"label": "ZONE A (WARD)", "pose": np.array([1.25, -0.95, 0.0]),         "drop": np.array([1.65, -0.95]), "col": (255, 170, 60)},
    "zone b": {"label": "ZONE B (LAB)",  "pose": np.array([1.25,  0.95, 0.0]),         "drop": np.array([1.65,  0.95]), "col": (255, 120, 220)},
    "reject": {"label": "REJECT BIN",    "pose": np.array([-0.45, 0.87, math.pi / 2]), "drop": np.array([-0.45, 1.30]), "col": (60, 150, 255)},
}
SLOTS = [(0.0, 0.0), (0.0, 0.09), (0.0, -0.09)]

# Random 2D Human Patrol Bounds & SSM Safety Parameters
WORKER_BOUNDS_X = (-0.25, 0.95)
WORKER_BOUNDS_Y = (-1.15, 1.15)
HUMAN_SPEED, HUMAN_R = 0.34, 0.12
STOP_CL, SLOW_CL, RETREAT_CL, ARM_HOLD_CL = 0.18, 0.75, 0.09, 0.06
V_MAX, W_MAX, V_RETREAT, ARM_W = 0.55, 1.8, 0.60, 1.9
BASE_HALF = (0.24, 0.18)

# Fixed HSV Ranges (Lowered blue saturation threshold to handle overhead specular lighting!)
HSV_RANGES = {
    "red":   [((0, 110, 70), (12, 255, 255)), ((168, 110, 70), (180, 255, 255))],
    "blue":  [((95, 65, 55), (138, 255, 255))],
    "green": [((38, 85, 65), (88, 255, 255))],
}
CAM_POS = np.array([-1.15, 0.0, 1.42])
CAM_FOVY = 46.0
PICK_TABLE_X = (-1.43, -0.87)
PICK_TABLE_Y = (-0.52, 0.52)

W, H = 1260, 720
MAIN_W, MAIN_H = 720, 460
CAM_W, CAM_H = 480, 195


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ----------------------------------------------------------------------------------------------
# 3. LANGUAGE UNDERSTANDING
# ----------------------------------------------------------------------------------------------
def parse_segment(text):
    t = text.lower().strip()
    color = next((c for c in COLORS if re.search(r"\b" + c + r"\b", t)), None)
    dest = None
    if re.search(r"\bzone\s*a\b|\bward\b|\bstation\s*a\b|\bto\s+a\b", t):
        dest = "zone a"
    elif re.search(r"\bzone\s*b\b|\blab\b|\bstation\s*b\b|\bto\s+b\b", t):
        dest = "zone b"
    elif re.search(r"\b(reject|bin|discard|trash|scrap|defect\w*)\b", t):
        dest = "reject"
    action = "PICK_AND_PLACE"
    if re.search(r"\b(inspect|scan|check|verify)\b", t):
        action = "INSPECT_AND_DELIVER"
    elif re.search(r"\b(sort|reject|discard|trash|scrap)\b", t):
        action = "SORT_TO_REJECT"
        dest = dest or "reject"
    conf = 0.5 * (color is not None) + 0.4 * (dest is not None) + (
        0.1 if action != "PICK_AND_PLACE" or re.search(r"\b(pick|move|bring|deliver|put|take|carry)\b", t) else 0.0
    )
    issue = ""
    if color is None:
        issue = 'AMBIGUOUS: specify red, blue or green (e.g. "move blue to zone b")'
    elif dest is None:
        issue = "AMBIGUOUS: specify zone a (ward), zone b (lab) or reject bin"
    return {
        "command": text.strip(),
        "action": action,
        "target": color or "?",
        "destination": dest or "?",
        "confidence": round(conf, 2),
        "valid": color is not None and dest is not None,
        "issue": issue,
    }


# ----------------------------------------------------------------------------------------------
# 4. HUMAN MOTION PREDICTION (Constant-Velocity 2D Kalman Filter)
# ----------------------------------------------------------------------------------------------
class CVKalman:
    def __init__(self, accel_sigma=0.9, meas_sigma=0.01):
        self.q = accel_sigma
        self.R = np.eye(2) * meas_sigma ** 2
        self.x = None
        self.P = np.eye(4)

    def reset(self):
        self.x = None
        self.P = np.eye(4)

    def step(self, z, dt):
        if self.x is None:
            self.x = np.array([z[0], z[1], 0.0, 0.0])
            self.P = np.diag([0.01, 0.01, 1.0, 1.0])
            return
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        Q = (G @ G.T) * self.q ** 2
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        Hm = np.zeros((2, 4))
        Hm[0, 0] = Hm[1, 1] = 1.0
        S = Hm @ self.P @ Hm.T + self.R
        K = self.P @ Hm.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.asarray(z) - Hm @ self.x)
        self.P = (np.eye(4) - K @ Hm) @ self.P

    def predict(self, horizon):
        return self.x[:2] + self.x[2:] * horizon


# ----------------------------------------------------------------------------------------------
# 5. SIMULATION
# ----------------------------------------------------------------------------------------------
class Sim:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(MJCF_XML)
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.main_renderer = mujoco.Renderer(self.model, height=MAIN_H, width=MAIN_W)
        self.cam_renderer = mujoco.Renderer(self.model, height=CAM_H, width=CAM_W)

        m = self.model
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        qa = lambda n: int(m.jnt_qposadr[jid(n)])
        da = lambda n: int(m.jnt_dofadr[jid(n)])
        self.q_base = [qa("base_x"), qa("base_y"), qa("base_yaw")]
        self.q_arm = [qa(n) for n in ARM_JOINTS]
        self.dof_arm = [da(n) for n in ARM_JOINTS]
        self.q_h = [qa("h_x"), qa("h_y")]
        self.q_p1 = [qa("pm1_x"), qa("pm1_y")]
        self.q_p2 = [qa("pm2_x"), qa("pm2_y")]
        self.q_cube = {c: qa("fj_" + c) for c in COLORS}
        self.site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "main_3d_cam")

        # Metrics
        self.attempted = self.done = self.failed = 0
        self.estops = self.recoveries = self.collisions = 0
        self.task_times = []
        self.min_clear = 9.9
        self.parse_ms = 0.0
        self.vision_ms = 0.0
        self.stopped = False
        self.last_estop = -99.0
        self.in_collision = False

        # Random 2D patrolling human state
        self.human = {
            "on": True,
            "pos": np.array([0.35, -0.80]),
            "goal": np.array([0.55, 0.85]),
            "wait_t": 0.0,
        }
        self.kf = CVKalman()
        self.pred1 = self.pred2 = self.human["pos"].copy()

        # Camera view & live webcam options
        self.chase_cam = False
        self.use_webcam = False
        self.cap = None

        self.detections = {}
        self.det_meta = {}
        self.cam_view = np.zeros((CAM_H, CAM_W, 3), np.uint8)
        self.queue, self.gen = [], None
        self.goal = None
        self.plan, self.step = [], -1
        self.detail = "IDLE"
        self.nav_target = None
        self.replan = False
        self.status, self.status_kind = 'READY: type a command (e.g. "move blue to zone b") and press ENTER', "info"
        self.dt = 0.016
        self.frame = 0
        self.reset(announce=False)

    def reset(self, announce=True):
        self.gen, self.queue = None, []
        self.base = BASE_START.copy()
        self.arm = STOW.copy()
        self.held = None
        self.cube_pos = {c: np.array([CUBE_START[c][0], CUBE_START[c][1], CUBE_Z]) for c in COLORS}
        self.slots = {"zone a": 0, "zone b": 0, "reject": 0}
        self.plan, self.step, self.goal, self.nav_target = [], -1, None, None
        self.detail = "IDLE"
        if announce:
            self.say("WAREHOUSE RESET: payloads restored to the pickup table", "info")

    def say(self, text, kind="ok"):
        self.status, self.status_kind = text, kind

    def submit(self, text):
        t = text.strip()
        if not t:
            return
        if re.search(r"\b(reset|restart|home)\b", t.lower()):
            self.reset()
            return
        t0 = time.perf_counter()
        segs = [s for s in re.split(r"\s*(?:\band then\b|\bthen\b|;)\s*", t, flags=re.I) if s.strip()]
        tasks = [parse_segment(s) for s in segs]
        self.parse_ms = (time.perf_counter() - t0) * 1000.0
        self.goal = tasks[0]
        bad = next((k for k in tasks if not k["valid"]), None)
        if bad:
            self.say(bad["issue"], "warn")
            return
        self.queue.extend(tasks)
        self.say(f"COMMAND ACCEPTED: {len(tasks)} task(s) queued (parse {self.parse_ms:.2f} ms, conf {tasks[0]['confidence']})", "info")

    def toggle_human(self):
        self.human["on"] = not self.human["on"]
        self.kf.reset()

    def toggle_cam_view(self):
        self.chase_cam = not self.chase_cam
        self.say(f"CAMERA VIEW: {'FOLLOW AMR CHASE CAM' if self.chase_cam else 'WAREHOUSE OVERVIEW'}", "info")

    def toggle_webcam(self):
        if not self.use_webcam:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                self.cap = cap
                self.use_webcam = True
                self.say("PERCEPTION INPUT: LIVE LAPTOP WEBCAM ACTIVE (show red/blue/green object)", "ok")
            else:
                cap.release()
                self.say("WEBCAM NOT FOUND: staying on MuJoCo Overhead RGB Camera", "warn")
        else:
            self.use_webcam = False
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.say("PERCEPTION INPUT: MUJOCO OVERHEAD RGB CAMERA ACTIVE", "info")

    def perturb_cube(self):
        target = None
        if self.goal and self.goal.get("valid") and self.gen is not None and self.held is None:
            target = self.goal["target"]
        if target is None:
            on_table = [c for c in COLORS if self.cube_pos[c][2] < 0.40 and PICK_TABLE_X[0] < self.cube_pos[c][0] < PICK_TABLE_X[1]]
            if not on_table:
                return
            target = random.choice(on_table)
        for _ in range(40):
            p = np.array([random.uniform(-1.24, -1.04), random.uniform(-0.30, 0.30), CUBE_Z])
            if all(np.linalg.norm(p[:2] - self.cube_pos[c][:2]) > 0.10 for c in COLORS if c != target):
                break
        self.cube_pos[target] = p
        if self.gen is not None and self.goal and self.goal.get("target") == target:
            self.replan = True
        self.say(f"PERTURBATION: {target.upper()} moved on the table -> re-localise and re-plan", "warn")

    # ------------------------------------------------------------------ Random 2D Human Patrol + Kalman Prediction
    def _new_worker_waypoint(self):
        for _ in range(30):
            gx = random.uniform(*WORKER_BOUNDS_X)
            gy = random.uniform(*WORKER_BOUNDS_Y)
            # Keep worker out of pickup table & station footprints
            if gx < -0.75 and abs(gy) < 0.65:
                continue
            if gx > 1.25 and abs(gy) > 0.55:
                continue
            if gx < -0.10 and gy > 0.90:
                continue
            return np.array([gx, gy])
        return np.array([0.35, 0.0])

    def update_human(self, dt):
        h = self.human
        if h["on"]:
            if h["wait_t"] > 0:
                h["wait_t"] -= dt
            else:
                diff = h["goal"] - h["pos"]
                dist = float(np.linalg.norm(diff))
                if dist < 0.08:
                    h["goal"] = self._new_worker_waypoint()
                    h["wait_t"] = random.uniform(0.2, 0.9)
                else:
                    h["pos"] = h["pos"] + (diff / dist) * HUMAN_SPEED * dt
            self.kf.step(h["pos"] + np.random.normal(0, 0.01, 2), dt)
            self.pred1 = self.kf.predict(0.7)
            self.pred2 = self.kf.predict(1.4)
        else:
            h["pos"] = np.array([4.0, 4.0])
            self.pred1 = self.pred2 = h["pos"].copy()

    def clearance(self, p):
        x, y, yaw = self.base
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = p[0] - x, p[1] - y
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        qx, qy = max(abs(lx) - BASE_HALF[0], 0.0), max(abs(ly) - BASE_HALF[1], 0.0)
        return math.hypot(qx, qy) - HUMAN_R

    def clearances(self):
        if not self.human["on"]:
            return 9.0, 9.0
        cur = self.clearance(self.human["pos"])
        pred = min(cur, self.clearance(self.pred1), self.clearance(self.pred2))
        return cur, pred

    def flag(self, state):
        stopped_now = state in ("STOP", "YIELD")
        now = time.perf_counter()
        if stopped_now and not self.stopped and now - self.last_estop > 3.0:
            self.estops += 1
            self.last_estop = now
        self.stopped = stopped_now
        return state

    def drive(self, target, dt):
        cur_cl, pred_cl = self.clearances()
        pos, yaw = self.base[:2], self.base[2]
        diff = target[:2] - pos
        dist = float(np.linalg.norm(diff))
        yaw_err = wrap(target[2] - yaw)
        if dist < 0.012 and abs(yaw_err) < 0.025:
            self.base = target.astype(float).copy()
            self.detail = "AMR: ARRIVED"
            return "ARRIVED"
        if self.human["on"] and cur_cl < RETREAT_CL:
            away = pos - self.human["pos"]
            away = away / (np.linalg.norm(away) + 1e-9)
            self.base[:2] += away * V_RETREAT * dt
            self.detail = f"AMR: YIELD - backing away (clearance {cur_cl:.2f} m)"
            return self.flag("YIELD")
        scale = float(np.clip((pred_cl - STOP_CL) / (SLOW_CL - STOP_CL), 0.0, 1.0))
        if scale <= 0.02:
            self.detail = f"AMR: SAFETY STOP - predicted clearance {pred_cl:.2f} m"
            return self.flag("STOP")
        v = min(V_MAX * scale, dist / max(dt, 1e-3))
        if dist > 1e-6:
            self.base[:2] += diff / dist * v * dt
        w = min(W_MAX * max(scale, 0.3), abs(yaw_err) / max(dt, 1e-3))
        self.base[2] += math.copysign(w * dt, yaw_err)
        st = "SLOW" if scale < 0.99 else "MOVING"
        self.detail = f"AMR: {st} {int(scale * 100)}% speed - pred clearance {min(pred_cl, 9.0):.2f} m"
        return self.flag(st)

    def arm_step(self, q_goal, dt):
        d = q_goal - self.arm
        v = np.clip(d * 4.2, -ARM_W, ARM_W)
        self.arm = self.arm + v * dt
        return float(np.max(np.abs(q_goal - self.arm))) < 0.01

    def ik(self, target, seed=None):
        d = self.ik_data
        d.qpos[:] = self.data.qpos
        d.qpos[self.q_base] = self.base
        q = (self.arm if seed is None else seed).astype(float).copy()
        err_n = 9.9
        for _ in range(120):
            d.qpos[self.q_arm] = q
            mujoco.mj_forward(self.model, d)
            err = np.asarray(target) - d.site_xpos[self.site_id]
            err_n = float(np.linalg.norm(err))
            if err_n < 0.003:
                break
            jac = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, d, jac, None, self.site_id)
            J4 = jac[:, self.dof_arm]
            J = np.stack([J4[:, 0], J4[:, 1] - J4[:, 3], J4[:, 2] - J4[:, 3]], axis=1)
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(3), err)
            q[:3] = np.clip(q[:3] + 0.8 * dq, IK_LO[:3], IK_HI[:3])
            q[3] = -(q[1] + q[2])
        return q, err_n < 0.015

    # ------------------------------------------------------------------ Task Generators
    def g_wait(self, secs):
        t = 0.0
        while t < secs:
            t += self.dt
            yield

    def g_drive(self, target, abort=False):
        while True:
            if self.drive(np.asarray(target, float), self.dt) == "ARRIVED":
                return True
            if abort and self.replan:
                return False
            yield

    def g_arm(self, q_goal, abort=False):
        while True:
            cur_cl, _ = self.clearances()
            if self.human["on"] and cur_cl < ARM_HOLD_CL:
                self.detail = f"ARM: PAUSED - worker inside cell (clearance {cur_cl:.2f} m)"
            else:
                if self.arm_step(q_goal, self.dt):
                    return True
                self.detail = "ARM: IK trajectory (damped least squares)"
            if abort and self.replan:
                return False
            yield

    def g_recover(self):
        self.recoveries += 1
        self.say("RECOVERY: target moved -> retract arm, re-detect, re-plan", "warn")
        yield from self.g_arm(STOW)
        yield from self.g_wait(0.5)

    def fail(self, msg):
        self.failed += 1
        self.say(msg, "warn")

    def g_task(self, task):
        color, dest, action = task["target"], task["destination"], task["action"]
        info = DEST[dest]
        t_start = time.perf_counter()
        self.attempted += 1
        self.goal = task
        self.plan = [
            f"1 PERCEIVE  localise {color.upper()} (RGB + HSV + depth plane)",
            "2 NAVIGATE  AMR to pickup standoff (SSM monitored)",
            "3 GRASP     IK pre-grasp, descend, close, lift" + (" + INSPECT" if action == "INSPECT_AND_DELIVER" else ""),
            f"4 TRANSPORT carry pose, drive to {dest.upper()}",
            "5 PLACE     IK place, release, VERIFY position",
        ]
        self.step = 0
        yield from self.g_arm(STOW)
        self.say(f"PERCEIVE: localising {color.upper()} via {'Live Webcam' if self.use_webcam else 'Overhead Camera'}", "info")
        yield from self.g_wait(0.4)

        tries = 0
        while True:
            tries += 1
            if tries > 6:
                self.fail("ABORT: could not secure the object after 6 attempts")
                return
            det = self.detections.get(color)
            if det is None:
                self.fail(f"NOT FOUND: {color.upper()} not visible (press Reset or show object)")
                return
            self.replan = False
            self.step = 1
            standoff = np.array([PICK_X, float(np.clip(det[1], -0.28, 0.28)), math.pi])
            self.nav_target = standoff
            self.say(f"NAVIGATE: AMR to pickup standoff ({color.upper()} at x={det[0]:.2f}, y={det[1]:.2f})", "info")
            ok = yield from self.g_drive(standoff, abort=True)
            if not ok:
                yield from self.g_recover()
                continue
            self.nav_target = None
            self.step = 2
            q_pre, ok1 = self.ik(det + np.array([0, 0, GRASP_DZ + 0.10]))
            q_gr, ok2 = self.ik(det + np.array([0, 0, GRASP_DZ]), seed=q_pre)
            if not (ok1 and ok2):
                self.fail("IK: object out of reach from this standoff")
                return
            self.say("GRASP: IK pre-grasp pose above the object", "info")
            ok = yield from self.g_arm(q_pre, abort=True)
            if not ok:
                yield from self.g_recover()
                continue
            ok = yield from self.g_arm(q_gr, abort=True)
            if not ok:
                yield from self.g_recover()
                continue
            break

        self.held = color
        self.say(f"GRASP: gripper closed on {color.upper()}", "ok")
        yield from self.g_wait(0.3)
        yield from self.g_arm(q_pre)

        if action == "INSPECT_AND_DELIVER":
            meta = self.det_meta.get(color, {"fill": 1.0, "aspect": 1.0})
            passed = meta["fill"] > 0.72 and 0.65 < meta["aspect"] < 1.45
            self.say(f"INSPECT: {color.upper()} fill={meta['fill']:.2f} aspect={meta['aspect']:.2f} -> {'PASS' if passed else 'DEFECT'}", "ok" if passed else "warn")
            yield from self.g_wait(0.8)
            if not passed:
                dest, info = "reject", DEST["reject"]
        yield from self.g_arm(STOW)

        self.step = 3
        self.nav_target = info["pose"]
        self.say(f"TRANSPORT: carrying {color.upper()} to {dest.upper()}", "info")
        yield from self.g_drive(info["pose"])
        self.nav_target = None

        self.step = 4
        dx, dy = SLOTS[self.slots[dest] % 3]
        self.slots[dest] += 1
        if dest == "reject":
            dx, dy = dy, dx
        drop = np.array([info["drop"][0] + dx, info["drop"][1] + dy, CUBE_Z])
        q_pp, ok1 = self.ik(drop + np.array([0, 0, GRASP_DZ + 0.10]))
        q_pl, ok2 = self.ik(drop + np.array([0, 0, GRASP_DZ]), seed=q_pp)
        if not (ok1 and ok2):
            self.fail("IK: drop point out of reach")
            return
        self.say("PLACE: IK pre-place, descend, release", "info")
        yield from self.g_arm(q_pp)
        yield from self.g_arm(q_pl)
        self.held = None
        self.cube_pos[color] = drop.copy()
        yield from self.g_wait(0.25)
        yield from self.g_arm(q_pp)
        yield from self.g_arm(STOW)

        err = float(np.linalg.norm(self.cube_pos[color][:2] - drop[:2]))
        self.done += 1
        self.task_times.append(time.perf_counter() - t_start)
        self.say(f"VERIFIED: {color.upper()} delivered to {dest.upper()} (err {err * 1000:.0f} mm, {self.task_times[-1]:.1f} s)", "ok")
        self.step = 5

    # ------------------------------------------------------------------ Vision (Sim Overhead OR Live Webcam)
    def pix_to_world(self, cx, cy, w, h, f, zd):
        return CAM_POS[0] - ((cy - h / 2.0) / f) * zd, CAM_POS[1] - ((cx - w / 2.0) / f) * zd

    def perceive(self):
        t0 = time.perf_counter()
        if self.use_webcam and self.cap is not None:
            ok, frame = self.cap.read()
            if ok:
                bgr = cv2.resize(frame, (CAM_W, CAM_H))
            else:
                self.toggle_webcam()
                return
        else:
            r = self.cam_renderer
            r.update_scene(self.data, camera="pickup_cam")
            bgr = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, w = bgr.shape[:2]
        f = (h / 2.0) / math.tan(math.radians(CAM_FOVY / 2.0))
        zd = CAM_POS[2] - (CUBE_Z + 0.025)

        if self.use_webcam:
            roi = np.full((h, w), 255, np.uint8)
        else:
            r_lo = int(h / 2.0 - (PICK_TABLE_X[1] - CAM_POS[0]) * f / zd)
            r_hi = int(h / 2.0 - (PICK_TABLE_X[0] - CAM_POS[0]) * f / zd)
            c_lo = int(w / 2.0 - (PICK_TABLE_Y[1] - CAM_POS[1]) * f / zd)
            c_hi = int(w / 2.0 - (PICK_TABLE_Y[0] - CAM_POS[1]) * f / zd)
            roi = np.zeros((h, w), np.uint8)
            roi[max(r_lo, 0):min(r_hi, h), max(c_lo, 0):min(c_hi, w)] = 255

        ann = bgr.copy()
        if not self.use_webcam:
            cv2.rectangle(ann, (max(c_lo, 0), max(r_lo, 0)), (min(c_hi, w) - 1, min(r_hi, h) - 1), (90, 90, 90), 1)

        dets, meta = {}, {}
        min_area = 250 if self.use_webcam else 35
        max_area = 25000 if self.use_webcam else 650

        for name, ranges in HSV_RANGES.items():
            mask = np.zeros((h, w), np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
            mask = cv2.bitwise_and(mask, roi)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best = None
            for c in cnts:
                area = cv2.contourArea(c)
                bx, by, bw, bh = cv2.boundingRect(c)
                if not (min_area <= area <= max_area) or bw == 0 or bh == 0:
                    continue
                aspect, fill = bw / float(bh), area / float(bw * bh)
                if 0.45 <= aspect <= 2.2 and fill > 0.55 and (best is None or area > best[0]):
                    best = (area, bx, by, bw, bh, aspect, fill)
            if best:
                area, bx, by, bw, bh, aspect, fill = best
                if self.use_webcam:
                    # Map real webcam detection onto pickup table workspace
                    wx = float(np.interp(by + bh / 2.0, [0, h], [-1.02, -1.26]))
                    wy = float(np.interp(bx + bw / 2.0, [0, w], [0.30, -0.30]))
                else:
                    wx, wy = self.pix_to_world(bx + bw / 2.0, by + bh / 2.0, w, h, f, zd)
                dets[name] = np.array([wx, wy, CUBE_Z])
                meta[name] = {"area": area, "aspect": aspect, "fill": fill}
                cv2.rectangle(ann, (bx - 3, by - 3), (bx + bw + 3, by + bh + 3), (0, 255, 0), 2)
                cv2.putText(ann, f"{name.upper()} ({wx:.2f},{wy:.2f})", (max(2, bx - 20), max(14, by - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        self.detections, self.det_meta = dets, meta
        cam_lbl = "LIVE WEBCAM [OPENCV HSV]" if self.use_webcam else "OVERHEAD RGB CAMERA | OPENCV HSV + CONTOUR"
        cv2.putText(ann, cam_lbl, (6, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 230, 255), 1, cv2.LINE_AA)
        self.cam_view = ann
        ms = (time.perf_counter() - t0) * 1000.0
        self.vision_ms = ms if self.vision_ms == 0 else 0.8 * self.vision_ms + 0.2 * ms

    # ------------------------------------------------------------------ Per-Frame Update
    def apply_state(self):
        d = self.data
        d.qpos[self.q_base] = self.base
        d.qpos[self.q_arm] = self.arm
        d.qpos[self.q_h] = self.human["pos"]
        d.qpos[self.q_p1] = self.pred1
        d.qpos[self.q_p2] = self.pred2
        for c in COLORS:
            a = self.q_cube[c]
            if c != self.held:
                d.qpos[a:a + 3] = self.cube_pos[c]
                d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.model, d)
        if self.held:
            tip = d.site_xpos[self.site_id].copy()
            self.cube_pos[self.held] = tip - np.array([0, 0, GRASP_DZ])
            a = self.q_cube[self.held]
            d.qpos[a:a + 3] = self.cube_pos[self.held]
            d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
            mujoco.mj_forward(self.model, d)

    def tick(self, dt):
        self.dt = dt
        self.update_human(dt)
        if self.gen is None and self.queue:
            self.gen = self.g_task(self.queue.pop(0))
        if self.gen is not None:
            try:
                next(self.gen)
            except StopIteration:
                self.gen = None
                self.nav_target = None
                if not self.queue:
                    self.detail = "IDLE - awaiting command"
            except Exception as e:
                traceback.print_exc()
                self.gen = None
                self.say(f"INTERNAL ERROR: {e}", "warn")
        cur, _ = self.clearances()
        if self.human["on"]:
            self.min_clear = min(self.min_clear, cur)
        if cur < 0 and not self.in_collision:
            self.collisions += 1
        self.in_collision = cur < 0
        self.apply_state()
        self.frame += 1
        if self.frame % 3 == 0:
            self.perceive()

    # ------------------------------------------------------------------ Drawing & Camera Projection
    def _update_main_cam(self):
        if self.chase_cam:
            self.data.cam_xpos[self.cam_id] = np.array([self.base[0] - 0.35, self.base[1] - 1.65, 1.45])
        else:
            self.data.cam_xpos[self.cam_id] = np.array([0.15, -3.05, 2.45])

    def project(self, p):
        R = self.data.cam_xmat[self.cam_id].reshape(3, 3)
        pc = R.T @ (np.asarray(p, float) - self.data.cam_xpos[self.cam_id])
        if pc[2] >= -1e-3:
            return None
        f = (MAIN_H / 2.0) / math.tan(math.radians(46.0 / 2.0))
        return int(MAIN_W / 2.0 + f * pc[0] / (-pc[2])), int(MAIN_H / 2.0 - f * pc[1] / (-pc[2]))

    def label(self, img, text, world, color, scale=0.45):
        p = self.project(world)
        if p is None:
            return
        x = p[0] - int(len(text) * 4.1)
        cv2.putText(img, text, (x, p[1]), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (x, p[1]), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    def draw_overlays(self, img):
        h = self.human
        if h["on"]:
            for rad, col in ((HUMAN_R + STOP_CL, (60, 60, 255)), (HUMAN_R + SLOW_CL, (0, 190, 255))):
                pts = [self.project((h["pos"][0] + rad * math.cos(a), h["pos"][1] + rad * math.sin(a), 0.01))
                       for a in np.linspace(0, 2 * math.pi, 48)]
                pts = [p for p in pts if p]
                if len(pts) > 3:
                    cv2.polylines(img, [np.array(pts, np.int32)], True, col, 1, cv2.LINE_AA)
            trail = [self.project((p[0], p[1], 0.02)) for p in (h["pos"], self.pred1, self.pred2)]
            if all(trail):
                cv2.polylines(img, [np.array(trail, np.int32)], False, (80, 80, 255), 2, cv2.LINE_AA)
            self.label(img, "WORKER (Random 2D + Kalman)", (h["pos"][0], h["pos"][1], 1.25), (120, 160, 255))
        if self.nav_target is not None:
            a = self.project((self.base[0], self.base[1], 0.02))
            b = self.project((self.nav_target[0], self.nav_target[1], 0.02))
            if a and b:
                cv2.line(img, a, b, (255, 220, 0), 2, cv2.LINE_AA)
                cv2.circle(img, b, 6, (255, 220, 0), 2, cv2.LINE_AA)
        for info in DEST.values():
            self.label(img, info["label"], (info["drop"][0], info["drop"][1], 0.62), info["col"])
        self.label(img, "PICKUP + VISION GANTRY", (-1.15, 0.0, 1.05), (255, 255, 255))
        self.label(img, "AMR + 4-DOF ARM", (self.base[0], self.base[1], 0.95), (0, 230, 255))
        cv2.putText(img, "EXPANDED INDUSTRIAL WAREHOUSE DIGITAL TWIN (MuJoCo)", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 230, 255), 2, cv2.LINE_AA)
        cur, pred = self.clearances()
        cv2.putText(img, f"SSM clearance {min(cur, 9.0):.2f} m | predicted {min(pred, 9.0):.2f} m | [V] View: {'Chase' if self.chase_cam else 'Overview'}",
                    (12, MAIN_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 235, 245), 1, cv2.LINE_AA)

    def compose(self, cmd, buttons):
        self.main_renderer.update_scene(self.data, camera="main_3d_cam")
        self._update_main_cam()
        img = cv2.cvtColor(self.main_renderer.render(), cv2.COLOR_RGB2BGR)
        self.draw_overlays(img)

        c = np.full((H, W, 3), (18, 22, 30), np.uint8)
        put = lambda t, p, s=0.5, col=(235, 240, 245), th=1: cv2.putText(c, t, p, cv2.FONT_HERSHEY_SIMPLEX, s, col, th, cv2.LINE_AA)
        put("VLA-BOT: Vision-Language-Action Warehouse Mobile Manipulator", (20, 36), 0.66, (245, 248, 252), 2)
        avg = (sum(self.task_times) / len(self.task_times)) if self.task_times else 0.0
        put(f"TASKS {self.done}/{self.attempted} | AVG {avg:.1f}s | E-STOPS {self.estops} | RECOVERIES {self.recoveries} | "
            f"COLLISIONS {self.collisions} | MIN CLEAR {self.min_clear if self.min_clear < 9 else 0:.2f}m",
            (668, 36), 0.40, (80, 235, 140), 1)

        c[55:55 + MAIN_H, 20:20 + MAIN_W] = img
        c[55:55 + CAM_H, 760:760 + CAM_W] = self.cam_view

        cv2.rectangle(c, (760, 260), (1240, 372), (30, 36, 48), -1)
        put("1. LANGUAGE -> STRUCTURED JSON GOAL", (772, 280), 0.48, (245, 195, 50))
        goal = self.goal if self.goal else {"status": "awaiting natural-language command"}
        g = {k: v for k, v in goal.items() if k != "issue"}
        lines = textwrap.wrap(json.dumps(g), 56)[:4]
        for i, ln in enumerate(lines):
            put(ln, (772, 300 + i * 15), 0.38)
        det_str = ",".join(k.upper() for k in self.detections.keys()) or "NONE"
        put(f"visible: [{det_str}] | parse {self.parse_ms:.2f}ms | vis {self.vision_ms:.1f}ms | q={len(self.queue)}",
            (772, 364), 0.37, (150, 162, 180))

        cv2.rectangle(c, (760, 382), (1240, 515), (30, 36, 48), -1)
        put("2. TASK PLANNER, SAFETY & TELEMETRY", (772, 402), 0.48, (80, 235, 140))
        for i, s in enumerate(self.plan):
            col = (0, 220, 240) if i == self.step else (160, 170, 185)
            put(s[:60], (772, 420 + i * 15), 0.38, col)
        put(self.detail[:68], (772, 507), 0.38, (200, 210, 225))

        cv2.rectangle(c, (20, 530), (1240, 576), (34, 42, 56), -1)
        cv2.rectangle(c, (20, 530), (1240, 576), (0, 220, 240), 2)
        if cmd:
            put(f"NL COMMAND > {cmd}_", (34, 560), 0.62, (255, 255, 255), 2)
        else:
            put('NL COMMAND > type here, e.g. "move red to zone a then blue to zone b", then press ENTER', (34, 560), 0.54, (130, 145, 165))

        buttons.clear()
        specs = [
            ("1 Red>A", "CMD", "inspect and deliver red to zone a", 125),
            ("2 Blue>B", "CMD", "pick blue and move to zone b", 125),
            ("3 Green>Rej", "CMD", "sort green to reject bin", 140),
            ("4 Queue All", "CMD", "move red to zone a then blue to zone b then green to reject", 140),
            ("TAB Shift", "PERTURB", None, 118),
            (f"` Worker:{'ON' if self.human['on'] else 'OFF'}", "HUMAN", None, 145),
            (f"V View:{'Chase' if self.chase_cam else 'Wide'}", "VIEW", None, 145),
            (f"W Cam:{'Web' if self.use_webcam else 'Sim'}", "WEBCAM", None, 130),
            ("Reset", "RESET", None, 96),
        ]
        bx = 20
        for label, kind, payload, wd in specs:
            x1, x2 = bx, bx + wd
            buttons.append((x1, 590, x2, 632, kind, payload))
            col = {
                "CMD": (24, 95, 130),
                "PERTURB": (145, 78, 20),
                "HUMAN": (155, 35, 35) if self.human["on"] else (55, 65, 80),
                "VIEW": (45, 95, 85),
                "WEBCAM": (95, 55, 135) if self.use_webcam else (55, 65, 80),
                "RESET": (52, 64, 82),
            }[kind]
            cv2.rectangle(c, (x1, 590), (x2, 632), col, -1)
            put(label, (x1 + 8, 616), 0.43, (255, 255, 255))
            bx = x2 + 7

        col = {"ok": (80, 235, 140), "info": (240, 220, 120), "warn": (90, 130, 255)}[self.status_kind]
        put("STATUS: " + self.status[:105], (20, 675), 0.55, col, 2)
        put("Keys: ENTER run | 1-4 presets (empty box) | TAB shift cube | ` worker | V cam view | W live webcam | ESC quit",
            (20, 704), 0.41, (150, 162, 180))
        return c


# ----------------------------------------------------------------------------------------------
# 6. MAIN LOOP
# ----------------------------------------------------------------------------------------------
PRESETS = [
    "inspect and deliver red to zone a",
    "pick blue and move to zone b",
    "sort green to reject bin",
    "move red to zone a then blue to zone b then green to reject",
]


def main():
    sim = Sim()
    win = "VLA-BOT Industrial Warehouse Digital Twin"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    state = {"cmd": ""}
    buttons = []

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if 20 <= x <= 1240 and 530 <= y <= 576:
            state["cmd"] = ""
            return
        for x1, y1, x2, y2, kind, payload in buttons:
            if x1 <= x <= x2 and y1 <= y <= y2:
                if kind == "CMD":
                    sim.submit(payload)
                elif kind == "PERTURB":
                    sim.perturb_cube()
                elif kind == "HUMAN":
                    sim.toggle_human()
                elif kind == "VIEW":
                    sim.toggle_cam_view()
                elif kind == "WEBCAM":
                    sim.toggle_webcam()
                elif kind == "RESET":
                    sim.reset()

    cv2.setMouseCallback(win, on_mouse)
    last = time.perf_counter()
    try:
        while True:
            now = time.perf_counter()
            dt = min(max(now - last, 0.001), 0.05)
            last = now
            sim.tick(dt)
            cv2.imshow(win, sim.compose(state["cmd"], buttons))
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            elif k in (10, 13):
                sim.submit(state["cmd"])
                state["cmd"] = ""
            elif k in (8, 127):
                state["cmd"] = state["cmd"][:-1]
            elif k == 9:
                sim.perturb_cube()
            elif k == 96:
                sim.toggle_human()
            elif state["cmd"] == "" and k in (ord("1"), ord("2"), ord("3"), ord("4")):
                sim.submit(PRESETS[k - ord("1")])
            elif state["cmd"] == "" and k in (ord("v"), ord("V")):
                sim.toggle_cam_view()
            elif state["cmd"] == "" and k in (ord("w"), ord("W")):
                sim.toggle_webcam()
            elif 32 <= k <= 126 and len(state["cmd"]) < 90:
                state["cmd"] += chr(k)
    finally:
        if sim.cap is not None:
            sim.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()