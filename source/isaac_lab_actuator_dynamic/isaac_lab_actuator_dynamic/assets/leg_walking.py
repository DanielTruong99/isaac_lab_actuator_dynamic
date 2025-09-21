import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaac_lab_actuator_dynamic.assets import LOCAL_ASSETS_DATA_DIR

LEGACTUATORDYNAMIC_USD_PATH = f"/home/humanoid2/DanielTruong/isaac_lab_actuator_dynamic/source/isaac_lab_actuator_dynamic/isaac_lab_actuator_dynamic/assets/urdf/leg05/robot.usd"

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
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4,
            # fix_root_link=True
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.78),
        joint_pos={
            'L_hip_joint': 0.0, # limit -35, 35 (degrees)
            'L_hip2_joint': 0.0, # limit -35, 35 (degrees)
            'L_thigh_joint': 0.51, # limit -70, 70 (degrees)
            'L_calf_joint': -0.85, # limit -110, 0 (degrees)
            'L_toe_joint': 0.6, # limit -45, 45 (degrees)
            'R_hip_joint': 0.0, # limit -35, 35 (degrees)
            'R_hip2_joint': 0.0, # limit -35, 35 (degrees)
            'R_thigh_joint': 0.51, # limit -70, 70 (degrees)
            'R_calf_joint': -0.85, # limit -110, 0 (degrees)
            'R_toe_joint': 0.6, # limit -45, 45
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.99,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_hip2_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=300.0,
            velocity_limit=100.0,
            stiffness={
                # ".*_hip_joint": 50.0,
                # ".*_hip2_joint": 70.0,
                # ".*_thigh_joint": 350.0,
                # ".*_calf_joint": 120.0,
                # ".*_hip_joint": 30.0,
                # ".*_hip2_joint": 30.0,
                # ".*_thigh_joint": 30.0,
                # ".*_calf_joint": 30.0,
                ".*_hip_joint": 0.0,
                ".*_hip2_joint": 0.0,
                ".*_thigh_joint": 0.0,
                ".*_calf_joint": 0.0,
            },
            damping={
                # ".*_hip_joint": 3.0,
                # ".*_hip2_joint": 4.0,
                # ".*_thigh_joint": 4.0,
                # ".*_calf_joint": 1.0,
                # ".*_hip_joint": 5.0,
                # ".*_hip2_joint": 5.0,
                # ".*_thigh_joint": 5.0,
                # ".*_calf_joint": 5.0,
                ".*_hip_joint": 0.0,
                ".*_hip2_joint": 0.0,
                ".*_thigh_joint": 0.0,
                ".*_calf_joint": 0.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_toe_joint"],
            effort_limit=30.0,
            velocity_limit=50.0,
            stiffness={".*_toe_joint": 0.0},
            damping={".*_toe_joint": 0.0},
        ),  
    },
)

