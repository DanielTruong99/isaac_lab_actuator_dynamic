import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg, IdealPDActuatorCfg, ImplicitActuatorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaac_lab_actuator_dynamic.assets import LOCAL_ASSETS_DATA_DIR
import math

LEG_CHANGED_IMU_USD_PATH = f"/home/humanoid2/DanielTruong/isaac_lab_actuator_dynamic/source/isaac_lab_actuator_dynamic/isaac_lab_actuator_dynamic/assets/usd/Leg_Changed_IMU/leg_changed_imu.usd"
LEG_CHANGED_IMU_URDF_PATH = f"/home/humanoid2/DanielTruong/isaac_lab_actuator_dynamic/source/isaac_lab_actuator_dynamic/isaac_lab_actuator_dynamic/assets/urdf/Leg_Changed_IMU/Wholebody_URDF_IMU_Changed.urdf"

EFFECTIVE_HIP_ARMATURE = 0.075340
EFFECTIVE_HIP2_ARMATURE = 0.125722
EFFECTIVE_THIGH_ARMATURE = 0.17 
EFFECTIVE_CALF_ARMATURE = 0.142563
EFFECTIVE_TOE_ARMATURE = 0.045970

LEG_CHANGED_IMU_STA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=LEG_CHANGED_IMU_USD_PATH,
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            fix_root_link=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # pos=(0.0, 0.0, 0.73),
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            'L_hip_joint': 0.0, # limit -35, 35 (degrees)
            'L_hip2_joint': 0.0, # limit -35, 35 (degrees)
            'L_thigh_joint': 0.18185188, # limit -70, 70 (degrees)
            'L_calf_joint': -0.3369326, # limit -110, 0 (degrees)
            'L_toe_joint': 0.1548768, # limit -45, 45 (degrees)
            'R_hip_joint': 0.0, # limit -35, 35 (degrees)
            'R_hip2_joint': 0.0, # limit -35, 35 (degrees)
            'R_thigh_joint': 0.18185188, # limit -70, 70 (degrees)
            'R_calf_joint': -0.3369326, # limit -110, 0 (degrees)
            'R_toe_joint': 0.1548768, # limit -45, 45 (degrees)
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.99,
    actuators={
        "left_leg": DelayedPDActuatorCfg(
            joint_names_expr=["L_hip_joint", "L_hip2_joint", "L_thigh_joint"],
            effort_limit_sim=98.0,
            velocity_limit_sim=23.0,
            min_delay=3,
            max_delay=4,
            stiffness={
                "L_hip_joint": 0.0,
                "L_hip2_joint": 0.0,
                "L_thigh_joint": 0.0,
            },
            damping={
                "L_hip_joint": 0.0,
                "L_hip2_joint": 0.0,
                "L_thigh_joint": 0.0,
            },
            armature={
                "L_hip_joint": EFFECTIVE_HIP_ARMATURE,
                "L_hip2_joint": EFFECTIVE_HIP2_ARMATURE,
                "L_thigh_joint": EFFECTIVE_THIGH_ARMATURE,
            },
        ),
        "L_calf": DelayedPDActuatorCfg(
            joint_names_expr=["L_calf_joint"],
            min_delay=3,
            max_delay=4,
            effort_limit_sim=150.0,
            velocity_limit_sim=16.7,
            stiffness={
                "L_calf_joint": 0.0,
            },
            damping={
                "L_calf_joint": 0.0,
            },
            armature={
                "L_calf_joint": EFFECTIVE_CALF_ARMATURE,
            },
        ),
        "L_feet": DelayedPDActuatorCfg(
            joint_names_expr=["L_toe_joint"],
            min_delay=2,
            max_delay=3,
            effort_limit_sim=28.0,
            velocity_limit_sim=8.69,
            stiffness={"L_toe_joint": 0.0},
            damping={"L_toe_joint": 0.0},
            armature={"L_toe_joint": EFFECTIVE_TOE_ARMATURE},
        ),

        "right_leg": DelayedPDActuatorCfg(
            joint_names_expr=["R_hip_joint", "R_hip2_joint", "R_thigh_joint"],
            min_delay=3,
            max_delay=4,
            effort_limit_sim=98.0,
            velocity_limit_sim=23.0,
            stiffness={
                "R_hip_joint": 0.0,
                "R_hip2_joint": 0.0,
                "R_thigh_joint": 0.0,
            },
            damping={
                "R_hip_joint": 0.0,
                "R_hip2_joint": 0.0,
                "R_thigh_joint": 0.0,
            },
            armature={
                "R_hip_joint": EFFECTIVE_HIP_ARMATURE,
                "R_hip2_joint": EFFECTIVE_HIP2_ARMATURE,
                "R_thigh_joint": EFFECTIVE_THIGH_ARMATURE,
            },
        ),

        "R_calf": DelayedPDActuatorCfg(
            joint_names_expr=["R_calf_joint"],
            effort_limit_sim=150.0,
            velocity_limit_sim=16.7,
            min_delay=3,
            max_delay=4,
            stiffness={
                "R_calf_joint": 0.0,
            },
            damping={
                "R_calf_joint": 0.0,
            },
            armature={
                "R_calf_joint": EFFECTIVE_CALF_ARMATURE,
            },
        ),

        "R_feet": DelayedPDActuatorCfg(
            joint_names_expr=["R_toe_joint"],
            min_delay=2,
            max_delay=3,
            effort_limit_sim=28.0,
            velocity_limit_sim=8.69,
            stiffness={"R_toe_joint": 0.0},
            damping={"R_toe_joint": 0.0},
            armature={"R_toe_joint": EFFECTIVE_TOE_ARMATURE},
        ),
    },
)


