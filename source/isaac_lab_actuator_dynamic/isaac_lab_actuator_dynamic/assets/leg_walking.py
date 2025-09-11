import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaac_lab_actuator_dynamic.assets import LOCAL_ASSETS_DATA_DIR

LEGACTUATORDYNAMIC_USD_PATH = f"/home/humanoid2/DanielTruong/isaac_lab_actuator_dynamic/source/isaac_lab_actuator_dynamic/isaac_lab_actuator_dynamic/assets/usd/leg05.usd"

LEGWALKING_CFG = ArticulationCfg(
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
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            'L_hip2_joint': 0.0, # limit -35, 35 (degrees)
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.97,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_hip2_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=300.0,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_joint": 50.0,
                ".*_hip2_joint": 70.0,
                ".*_thigh_joint": 350.0,
                ".*_calf_joint": 120.0,
            },
            damping={
                ".*_hip_joint": 3.0,
                ".*_hip2_joint": 4.0,
                ".*_thigh_joint": 4.0,
                ".*_calf_joint": 1.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_toe_joint"],
            effort_limit=30.0,
            velocity_limit=50.0,
            stiffness={".*_toe_joint": 50.0},
            damping={".*_toe_joint": 1.0},
        ),  
    },
)

