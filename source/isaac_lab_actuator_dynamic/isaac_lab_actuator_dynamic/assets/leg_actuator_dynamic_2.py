import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaac_lab_actuator_dynamic.assets import LOCAL_ASSETS_DATA_DIR

LEGACTUATORDYNAMIC_USD_PATH = f"/home/humanoid2/DanielTruong/isaac_lab_actuator_dynamic/source/isaac_lab_actuator_dynamic/isaac_lab_actuator_dynamic/assets/usd/leg05.usd"

LEGACTUATORDYNAMIC_2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=LEGACTUATORDYNAMIC_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4, fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.93),
        joint_pos={
            'L_hip2_joint': 0.0, # limit -35, 35 (degrees)
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.97,
    actuators={
        "left_leg": ImplicitActuatorCfg(
            joint_names_expr=["L_hip2_joint", "L_hip_joint", "L_thigh_joint", "L_calf_joint", "L_toe_joint"],
            effort_limit=300.0,
            velocity_limit=50.0,
            stiffness={
                "L_hip_joint": 200.0,
                "L_hip2_joint": 350.0,
                "L_thigh_joint": 350.0,
                "L_calf_joint": 200.0,
                "L_toe_joint": 50.0,
            },
            damping={
                "L_hip_joint": 5.0,
                "L_hip2_joint": 5.0,
                "L_thigh_joint": 5.0,
                "L_calf_joint": 5.0,
                "L_toe_joint": 5.0,
            },
        ),
        "right_leg": ImplicitActuatorCfg(
            joint_names_expr=["R_hip2_joint", "R_hip_joint", "R_thigh_joint", "R_calf_joint", "R_toe_joint"],
            effort_limit=300.0,
            velocity_limit=50.0,
            stiffness={
                "R_hip_joint": 1000.0,
                "R_hip2_joint": 1000.0,
                "R_thigh_joint": 1000.0,
                "R_calf_joint": 1000.0,
                "R_toe_joint": 1000.0,
            },
            damping={
                "R_hip_joint": 3.0,
                "R_hip2_joint": 4.0,
                "R_thigh_joint": 4.0,
                "R_calf_joint": 1.0,
                "R_toe_joint": 1.0,
            },
        ),        
    },
)

