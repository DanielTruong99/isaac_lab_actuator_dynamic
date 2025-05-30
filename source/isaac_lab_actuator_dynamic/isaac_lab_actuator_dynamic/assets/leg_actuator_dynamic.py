import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaac_lab_actuator_dynamic.assets import LOCAL_ASSETS_DATA_DIR

LEGACTUATORDYNAMIC_USD_PATH = f"{LOCAL_ASSETS_DATA_DIR}/Robots/Aidin/aidin_quadruped/aidin_quadruped.usd"

LEGACTUATORDYNAMIC_CFG = ArticulationCfg(
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
            'LFJ_scap': 0.0, # limit -35, 35 (degrees)
            'LFJ_hip': 0.0, # limit -180, 180 (degrees)
            'LFJ_knee': 0.0, # limit 0, 180 (degrees)
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.97,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["LFJ_scap", "LFJ_hip", "LFJ_knee"],
            effort_limit=300.0,
            velocity_limit=100.0,
            stiffness={
                "LFJ_scap": 0.0,
                "LFJ_hip": 0.0,
                "LFJ_knee": 0.0,
            },
            damping={
                "LFJ_scap": 0.0,
                "LFJ_hip": 0.0,
                "LFJ_knee": 0.0,
            },
        ),
    },
)

