# Multi-Gait Locomotion for Unitree H1 in Isaac Lab


> *Framework: NVIDIA Isaac Lab v2.1.0* 
> *Robot: Unitree H1 Humanoid*

## 1. Overview

This repository implements a versatile, command-conditioned Reinforcement Learning (RL) controller for the **Unitree H1** humanoid robot.

Building upon the **"Multiplicity of Behavior" (MoB)** paradigm introduced in and the **Whole-Body Control** architecture of *HUGWBC*, this project extends the standard locomotion interface. Instead of a simple 3D velocity command, we utilize a **10D Command Interface** that decouples **Task** (velocity goals) from **Behavior** (gait style and posture)

### Key Capabilities
* **Dynamic Gait Switching:** Seamlessly transition between **Walking** (anti-phase) and **Jumping** (in-phase) in real-time by modulating timing offsets.
* **10D Command Space:** Direct control over cadence, stance duration, swing height, and upper-body posture (pitch/waist/height).
* **Isaac Lab Native:** Built on NVIDIA's latest modular RL framework (v2.1.0) for improved scalability, sensor integration, and sim-to-real transfer.

---

## 2. The 10D Command Interface

The policy input is a vector of shape `(num_envs, 10)`. The commands are categorized into Task goals (Movement) and Behavior modifiers (Gait & Posture). This extended command space allows for versatile locomotion behaviors.

| Index | Parameter | Description | Range / Unit | Functionality |
| :--- | :--- | :--- | :--- | :--- |
| **Task** | | | | *Target Goals* |
| `0` | `lin_vel_x` | Forward Velocity | $\pm 2.0$ m/s | Base movement speed ($v_x$). |
| `1` | `lin_vel_y` | Lateral Velocity | $\pm 0.6$ m/s | Side-stepping speed ($v_y$). |
| `2` | `ang_vel_z` | Yaw Rate | $\pm 1.0$ rad/s | Turning rate ($\omega_z$). |
| **Gait** | | | | *Cycle Modulation* |
| `3` | `frequency` | Cadence | $1.5 - 3.5$ Hz | Controls the speed of the gait cycle ($f^{cmd}$). |
| `4` | `phase_offset` | **Timing Offset** | $0.0$ or $0.5$ | Determines the gait type (Walk vs Jump). |
| `5` | `duration` | Stance Duration | $0.5$ | Ratio of stance phase vs swing phase ($\phi_{stance}$). |
| `6` | `foot_swing` | Swing Height | $0.05 - 0.2$ m | Target peak height of the foot in air ($h_z^{f,cmd}$). |
| **Posture** | | | | *Whole-Body Control* |
| `7` | `body_height` | Base Height | $\pm 0.1$ m | Vertical CoM offset (Crouch vs Stand) ($h$). |
| `8` | `body_pitch` | Torso Pitch | $\pm 0.2$ rad | Leaning forward/backward ($p$). |
| `9` | `waist_yaw` | Waist Joint | $\pm 0.3$ rad | Waist orientation offset ($w$). |

---

## 3. Gait Mechanism: Timing & Phase Scheduler

The core logic for multi-gait locomotion relies on a **Phase-Based Scheduler**. We maintain a global phase variable $\phi(t)$ that loops continuously from $0 \to 1$ driven by the command frequency.

### 3.1. Phase Calculation
The phase for each foot is calculated using the global phase and the commanded offset:

$$
\phi_{left} = \phi_{global}
$$

$$
\phi_{right} = (\phi_{global} + \text{phase\_offset}) \pmod{1}
$$

### 3.2. Understanding Timing Offset (Command `[4]`)
The `phase_offset` determines the synchronization between the left and right legs.

#### **Mode A: Walking (Anti-Phase)**
* **Command:** `phase_offset = 0.5`
* **Behavior:** The legs are $180^\circ$ out of phase. When the Left foot starts a cycle ($0.0$), the Right foot is exactly halfway ($0.5$).
* **Visual Schedule:**
    ```text
    Time:   0%    25%   50%   75%   100%
    Left:   [STANCE----][SWING-----]
    Right:  [SWING-----][STANCE----]
    ```

#### **Mode B: Jumping (In-Phase)**
* **Command:** `phase_offset = 0.0`
* **Behavior:** The legs are synchronized. Both feet enter stance and swing phases simultaneously.
* **Visual Schedule:**
    ```text
    Time:   0%    25%   50%   75%   100%
    Left:   [STANCE----][SWING-----]
    Right:  [STANCE----][SWING-----]
    ```

### 3.3. Stance Duration & Smoothing
The `duration` (Command `[5]`) controls the split between Stance and Swing. In our implementation, this value is currently **fixed to 0.5**, resulting in an equal stance/swing ratio. We apply a **Von Mises / Normal-CDF smoothing** function at the phase transitions to generate *soft contact targets* in the range $[0, 1]$. This encourages the RL policy to learn smooth contact transitions instead of abrupt impacts.


---

## 4. Visual Demonstrations

*The following demonstrations showcase the policy running in Isaac Lab.*

Link demo: [here](https://www.canva.com/design/DAHAy2zprkQ/RsNm2OBLlsDOltMOcWSWXA/edit?utm_content=DAHAy2zprkQ&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

## 5. Reward Function Architecture

The reward function is designed to enforce the commanded behavior while maintaining stability. It is a weighted sum of **Task**, **Gait**, and **Regularization** terms.

### Primary Rewards
* **Velocity Tracking:** Penalties for deviations from target $v_x, v_y, \omega_z$.
* **Gait Enforcement (Phase-Dependent):**
    * `walking_jumping`: Mode-dependent contact-pattern penalty to prevent degenerate gaits.
      * Walking mode (`cmd[4] = 0.5`): Penalizes 0-contact and 2-contact states, encourages single support (1 foot in contact)
      * Jumping mode (`cmd[4] = 0.0`): Penalizes single-contact states, prevents one-leg hopping and encourages synchronized contacts (both stance or both swing)
      * Disabled when commanded velocity is near zero.
    * `tracking_contacts_shaped_force`: Penalizes ground reaction forces when the scheduler dictates **Swing Phase** (must lift foot).
    * `tracking_contacts_shaped_vel`: Penalizes foot velocity when the scheduler dictates **Stance Phase** (no sliding).
    * `foot_clearance_cmd_linear`: Penalizes insufficient swing-foot clearance relative to the commanded swing height (cmd[6]).
* **Posture Control:**
    * `orientation_control`: Tracks the desired body pitch (`cmd[8]`).
    * `base_height`: Maintains the commanded body height (`cmd[7]`).
    * `waist_control`: Tracks the commanded waist joint position (`cmd[9]`) using an L2 penalty on waist joint angle error.

### Regularization
* **Jumping Symmetry:** A specific penalty used when `phase_offset ≈ 0` to enforce left/right symmetry and prevent "galloping" artifacts during jumps.
* **Joint Constraints:** Penalties for joint acceleration, velocity limits, and awkward arm poses (`joint_deviation_arms`).

## 6. INSTALLATION (ISAAC LAB v2.1.0)
Install Isaac Lab (official guide)
https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/pip_installation.html

Clone/copy this repository outside the IsaacLab core directory.

Install as editable extension
python -m pip install -e source/UnitreeG1

## USAGE
List available tasks
```
python scripts/list_envs.py
```

Train
```
python scripts/rsl_rl/train.py --task=Template-Unitreeg1-v0 --num_envs=4096 --headless
```
Play

Modify the command parameters in lines 165–174 of play.py to test different behaviors, based on the ranges listed in Section 2.
```
python scripts/rsl_rl/play.py --task=Template-Unitreeg1-v0 --num_envs=1 --checkpoint=model_25400.pt
```

