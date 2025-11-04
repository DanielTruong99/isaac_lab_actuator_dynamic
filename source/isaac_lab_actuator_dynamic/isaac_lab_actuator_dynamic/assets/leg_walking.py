import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg, IdealPDActuatorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaac_lab_actuator_dynamic.assets import LOCAL_ASSETS_DATA_DIR
import math

LEGACTUATORDYNAMIC_USD_PATH = f"/home/humanoid2/DanielTruong/isaac_lab_actuator_dynamic/source/isaac_lab_actuator_dynamic/isaac_lab_actuator_dynamic/assets/urdf/leg05/robot.usd"
LEGACTUATORDYNAMIC_2_USD_PATH = f"/home/humanoid2/DanielTruong/isaac_lab_actuator_dynamic/source/isaac_lab_actuator_dynamic/isaac_lab_actuator_dynamic/assets/urdf/leg05/robot_5.usd"
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
        # rot=(math.cos(math.radians(15.0)/2), 0.0, -math.sin(math.radians(15.0)/2), 0.0),
        pos=(0.0, 0.0, 0.65),
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
            # 'L_hip_joint': 0.0, # limit -35, 35 (degrees)
            # 'L_hip2_joint': 0.0, # limit -35, 35 (degrees)
            # 'L_thigh_joint': 0.2, # limit -70, 70 (degrees)
            # 'L_calf_joint': -0.34, # limit -110, 0 (degrees)
            # 'L_toe_joint': 0.2, # limit -45, 45 (degrees)
            # 'R_hip_joint': 0.0, # limit -35, 35 (degrees)
            # 'R_hip2_joint': 0.0, # limit -35, 35 (degrees)
            # 'R_thigh_joint': 0.2, # limit -70, 70 (degrees)
            # 'R_calf_joint': -0.34, # limit -110, 0 (degrees)
            # 'R_toe_joint': 0.2, # limit -45, 45
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.99,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_hip2_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=300.0,
            velocity_limit=100.0,
            friction={
                ".*_hip_joint": 0.1,
                ".*_hip2_joint": 0.1,
                ".*_thigh_joint": 0.2,
                ".*_calf_joint": 0.1,
            },
            # dynamic_friction={
            #     ".*_hip_joint": 0.1,
            #     ".*_hip2_joint": 0.1,
            #     ".*_thigh_joint": 0.2,
            #     ".*_calf_joint": 0.1,
            # },
            stiffness={
                # ".*_hip_joint": 50.0,
                # ".*_hip2_joint": 70.0,
                # ".*_thigh_joint": 700.0,
                # ".*_calf_joint": 700.0,
                ".*_hip_joint": 30.0,
                ".*_hip2_joint": 30.0,
                ".*_thigh_joint": 30.0,
                ".*_calf_joint": 30.0,
                # ".*_hip_joint": 0.0,
                # ".*_hip2_joint": 0.0,
                # ".*_thigh_joint": 0.0,
                # ".*_calf_joint": 0.0,
            },
            damping={
                # ".*_hip_joint": 3.0,
                # ".*_hip2_joint": 4.0,
                # ".*_thigh_joint": 4.0,
                # ".*_calf_joint": 1.0,
                ".*_hip_joint": 5.0,
                ".*_hip2_joint": 5.0,
                ".*_thigh_joint": 5.0,
                ".*_calf_joint": 5.0,
                # ".*_hip_joint": 0.0,
                # ".*_hip2_joint": 0.0,
                # ".*_thigh_joint": 0.0,
                # ".*_calf_joint": 0.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_toe_joint"],
            effort_limit=30.0,
            velocity_limit=50.0,
            stiffness={".*_toe_joint": 30.0},
            damping={".*_toe_joint": 5.0},
        ),  
    },
)

LEGWALKING_HIGH_GAIN_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=LEGACTUATORDYNAMIC_2_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=4,
            # fix_root_link=True
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # rot=(math.cos(math.radians(15.0)/2), 0.0, -math.sin(math.radians(15.0)/2), 0.0),
        pos=(0.0, 0.0, 0.86),
        joint_pos={
            'L_hip_joint': 0.0, # limit -35, 35 (degrees)
            'L_hip2_joint': 0.0, # limit -35, 35 (degrees)
            'L_thigh_joint': 0.55, # limit -70, 70 (degrees)
            'L_calf_joint': -1.0, # limit -110, 0 (degrees)
            'L_toe_joint': 0.45, # limit -45, 45 (degrees)
            'R_hip_joint': 0.0, # limit -35, 35 (degrees)
            'R_hip2_joint': 0.0, # limit -35, 35 (degrees)
            'R_thigh_joint': 0.55, # limit -70, 70 (degrees)
            'R_calf_joint': -1.0, # limit -110, 0 (degrees)
            'R_toe_joint': 0.45, # limit -45, 45 (degrees)
        },
        # pos=(0.0, 0.0, 0.65),
        # joint_pos={
        #     'L_hip_joint': 0.0, # limit -35, 35 (degrees)
        #     'L_hip2_joint': 0.0, # limit -35, 35 (degrees)
        #     'L_thigh_joint': 0.0, # limit -70, 70 (degrees)
        #     'L_calf_joint': 0.0, # limit -110, 0 (degrees)
        #     'L_toe_joint': 0.45, # limit -45, 45 (degrees)
        #     'R_hip_joint': 0.0, # limit -35, 35 (degrees)
        #     'R_hip2_joint': 0.0, # limit -35, 35 (degrees)
        #     'R_thigh_joint': 0.0, # limit -70, 70 (degrees)
        #     'R_calf_joint': 0.0, # limit -110, 0 (degrees)
        #     'R_toe_joint': 0.0, # limit -45, 45 (degrees)
        # },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.99,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_hip2_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=125.0,
            velocity_limit=50.0,
            # Low gains first
            stiffness={
                ".*_hip_joint": 45.0,
                ".*_hip2_joint": 45.0,
                ".*_thigh_joint": 45.0,
                ".*_calf_joint": 45.0,
            },
            damping={
                ".*_hip_joint": 1.5,
                ".*_hip2_joint": 1.5,
                ".*_thigh_joint": 1.5,
                ".*_calf_joint": 1.5,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_toe_joint"],
            effort_limit=30.0,
            velocity_limit=50.0,
            stiffness={".*_toe_joint": 45.0},
            damping={".*_toe_joint": 0.8},
        ),  
    },
)

LEGWALKING_HIGH_GAIN_AMARTURE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=LEGACTUATORDYNAMIC_2_USD_PATH,
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            fix_root_link=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            approximation="boundingCube",
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.71),
        # pos=(0.0, 0.0, 0.99),
        joint_pos={
            'L_hip_joint': 0.0, # limit -35, 35 (degrees)
            'L_hip2_joint': 0.0, # limit -35, 35 (degrees)
            'L_thigh_joint': 0.2, # limit -70, 70 (degrees)
            'L_calf_joint': -0.25, # limit -110, 0 (degrees)
            'L_toe_joint': 0.1, # limit -45, 45 (degrees)
            'R_hip_joint': 0.0, # limit -35, 35 (degrees)
            'R_hip2_joint': 0.0, # limit -35, 35 (degrees)
            'R_thigh_joint': 0.2, # limit -70, 70 (degrees)
            'R_calf_joint': -0.25, # limit -110, 0 (degrees)
            'R_toe_joint': 0.1, # limit -45, 45 (degrees)
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.97,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_hip2_joint", ".*_thigh_joint"],
            effort_limit=105.0,
            velocity_limit=23.0,
            # Low gains first
            stiffness={
                ".*_hip_joint": 100.0,
                ".*_hip2_joint": 100.0,
                ".*_thigh_joint": 100.0,
                # ".*_calf_joint": 300.0,
            },
            damping={
                ".*_hip_joint": 3.5,
                ".*_hip2_joint": 3.5,
                ".*_thigh_joint": 3.5,
                # ".*_calf_joint": 7.5,
            },
            armature={
                ".*_hip_joint": 0.167592,
                ".*_hip2_joint": 0.167592,
                ".*_thigh_joint": 0.12109824,
                # ".*_calf_joint": 0.015,
            },
            friction={
                ".*_hip_joint": 0.02,
                ".*_hip2_joint": 0.5,
                ".*_thigh_joint": 0.5,
                # ".*_calf_joint": 0.02,
            }
        ),
        "calf": ImplicitActuatorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=159.0,
            velocity_limit=16.7,
            # Low gains first
            stiffness={
                ".*_calf_joint": 300.0,
            },
            damping={
                ".*_calf_joint": 7.5,
            },
            armature={
                ".*_calf_joint": 0.12109824,
            },
            friction=0.02
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_toe_joint"],
            effort_limit=28.0,
            velocity_limit=8.69,
            stiffness={".*_toe_joint": 20.0},
            damping={".*_toe_joint": 1.5},
            armature={".*_toe_joint": 0.0312822},
            friction=0.02,
            # viscous_friction={".*_toe_joint": 2.0},
        ),  
    },
)


LEGWALKING_HIGH_GAIN_AMARTURE_2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=LEGACTUATORDYNAMIC_2_USD_PATH,
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            fix_root_link=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            approximation="boundingCube",
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        # pos=(0.0, 0.0, 0.99),
        joint_pos={
            'L_hip_joint': 0.0, # limit -35, 35 (degrees)
            'L_hip2_joint': 0.0, # limit -35, 35 (degrees)
            'L_thigh_joint': 0.2, # limit -70, 70 (degrees)
            'L_calf_joint': -0.25, # limit -110, 0 (degrees)
            'L_toe_joint': 0.1, # limit -45, 45 (degrees)
            'R_hip_joint': 0.0, # limit -35, 35 (degrees)
            'R_hip2_joint': 0.0, # limit -35, 35 (degrees)
            'R_thigh_joint': 0.2, # limit -70, 70 (degrees)
            'R_calf_joint': -0.25, # limit -110, 0 (degrees)
            'R_toe_joint': 0.1, # limit -45, 45 (degrees)
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.97,
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_hip2_joint", ".*_thigh_joint"],
            effort_limit=105.0,
            velocity_limit=23.0,
            # Low gains first
            # stiffness={
            #     ".*_hip_joint": 90.0,
            #     ".*_hip2_joint": 120.0,
            #     ".*_thigh_joint": 300.0,
            #     # ".*_calf_joint": 300.0,
            # },
            # damping={
            #     ".*_hip_joint": 1.5,
            #     ".*_hip2_joint": 1.5,
            #     ".*_thigh_joint": 1.5,
            #     # ".*_calf_joint": 7.5,
            # },
            stiffness={
                ".*_hip_joint": 45.0,
                ".*_hip2_joint": 45.0,
                ".*_thigh_joint": 45.0,
                # ".*_calf_joint": 300.0,
            },
            damping={
                ".*_hip_joint": 1.5,
                ".*_hip2_joint": 1.5,
                ".*_thigh_joint": 1.5,
                # ".*_calf_joint": 7.5,
            },
            # armature={
            #     ".*_hip_joint": 0.167592,
            #     ".*_hip2_joint": 0.167592,
            #     ".*_thigh_joint": 0.12109824,
            #     # ".*_calf_joint": 0.015,
            # },
            # friction={
            #     ".*_hip_joint": 0.02,
            #     ".*_hip2_joint": 0.5,
            #     ".*_thigh_joint": 0.5,
            #     # ".*_calf_joint": 0.02,
            # }
        ),
        "calf": IdealPDActuatorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=159.0,
            velocity_limit=16.7,
            # Low gains first
            # stiffness={
            #     ".*_calf_joint": 300.0,
            # },
            # damping={
            #     ".*_calf_joint": 7.5,
            # },
            stiffness={
                ".*_calf_joint": 45.0,
            },
            damping={
                ".*_calf_joint": 1.5,
            },
            # armature={
            #     ".*_calf_joint": 0.12109824,
            # },
            # friction=0.02
        ),
        "feet": IdealPDActuatorCfg(
            joint_names_expr=[".*_toe_joint"],
            effort_limit=28.0,
            velocity_limit=8.69,
            stiffness={".*_toe_joint": 45.0},
            damping={".*_toe_joint": 0.8},
            # armature={".*_toe_joint": 0.0312822},
            # friction=0.02,
            # viscous_friction={".*_toe_joint": 2.0},
        ),  
    },
)



LEGWALKING_HIGH_GAIN_AMARTURE_3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=LEGACTUATORDYNAMIC_2_USD_PATH,
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            fix_root_link=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            approximation="boundingCube",
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            'L_hip_joint': 0.0, # limit -35, 35 (degrees)
            'L_hip2_joint': 0.0, # limit -35, 35 (degrees)
            'L_thigh_joint': 0.0, # limit -70, 70 (degrees)
            'L_calf_joint': 0.0, # limit -110, 0 (degrees)
            'L_toe_joint': 0.0, # limit -45, 45 (degrees)
            'R_hip_joint': 0.0, # limit -35, 35 (degrees)
            'R_hip2_joint': 0.0, # limit -35, 35 (degrees)
            'R_thigh_joint': 0.0, # limit -70, 70 (degrees)
            'R_calf_joint': 0.0, # limit -110, 0 (degrees)
            'R_toe_joint': 0.0, # limit -45, 45 (degrees)
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.97,
    actuators={
        "left_leg": ImplicitActuatorCfg(
            joint_names_expr=["L_hip_joint", "L_hip2_joint", "L_thigh_joint"],
            effort_limit=105.0,
            velocity_limit=23.0,
            stiffness={
                "L_hip_joint": 45.0,
                "L_hip2_joint": 45.0,
                "L_thigh_joint": 45.0,
            },
            damping={
                "L_hip_joint": 1.5,
                "L_hip2_joint": 1.5,
                "L_thigh_joint": 1.5,
            },
            armature={
                "L_hip_joint": 0.102588,
                "L_hip2_joint": 0.457870,
                "L_thigh_joint": 0.596446,
            },
        ),
        "L_calf": ImplicitActuatorCfg(
            joint_names_expr=["L_calf_joint"],
            effort_limit=159.0,
            velocity_limit=16.7,
            stiffness={
                "L_calf_joint": 45.0,
            },
            damping={
                "L_calf_joint": 1.5,
            },
            armature={
                "L_calf_joint": 0.076797,
            },
        ),
        "L_feet": ImplicitActuatorCfg(
            joint_names_expr=["L_toe_joint"],
            effort_limit=28.0,
            velocity_limit=8.69,
            stiffness={"L_toe_joint": 45.0},
            damping={"L_toe_joint": 0.8},
            armature={"L_toe_joint": 0.040897},
        ),  

        "right_leg": ImplicitActuatorCfg(
            joint_names_expr=["R_hip_joint", "R_hip2_joint", "R_thigh_joint"],
            effort_limit=105.0,
            velocity_limit=23.0,
            stiffness={
                "R_hip_joint": 45.0,
                "R_hip2_joint": 45.0,
                "R_thigh_joint": 45.0,
            },
            damping={
                "R_hip_joint": 1.5,
                "R_hip2_joint": 1.5,
                "R_thigh_joint": 1.5,
            },
            armature={
                "R_hip_joint": 0.130943,
                "R_hip2_joint": 0.438781,
                "R_thigh_joint": 0.336888,
            },
        ),

        "R_calf": ImplicitActuatorCfg(
            joint_names_expr=["R_calf_joint"],
            effort_limit=159.0,
            velocity_limit=16.7,
            stiffness={
                "R_calf_joint": 45.0,
            },
            damping={
                "R_calf_joint": 1.5,
            },
            armature={
                "R_calf_joint": 0.231993,
            },
        ),

        "R_feet": ImplicitActuatorCfg(
            joint_names_expr=["R_toe_joint"],
            effort_limit=28.0,
            velocity_limit=8.69,
            stiffness={"R_toe_joint": 45.0},
            damping={"R_toe_joint": 0.8},
            armature={"R_toe_joint": 0.062950},
        ),
    },
)

LEGWALKING_HIGH_GAIN_AMARTURE_5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=LEGACTUATORDYNAMIC_2_USD_PATH,
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            fix_root_link=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            approximation="boundingCube",
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            'L_hip_joint': 0.0, # limit -35, 35 (degrees)
            'L_hip2_joint': 0.0, # limit -35, 35 (degrees)
            'L_thigh_joint': 0.0, # limit -70, 70 (degrees)
            'L_calf_joint': 0.0, # limit -110, 0 (degrees)
            'L_toe_joint': 0.0, # limit -45, 45 (degrees)
            'R_hip_joint': 0.0, # limit -35, 35 (degrees)
            'R_hip2_joint': 0.0, # limit -35, 35 (degrees)
            'R_thigh_joint': 0.0, # limit -70, 70 (degrees)
            'R_calf_joint': 0.0, # limit -110, 0 (degrees)
            'R_toe_joint': 0.0, # limit -45, 45 (degrees)
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.97,
    actuators={
        "left_leg": ImplicitActuatorCfg(
            joint_names_expr=["L_hip_joint", "L_hip2_joint", "L_thigh_joint"],
            effort_limit=105.0,
            velocity_limit=23.0,
            stiffness={
                "L_hip_joint": 100.0,
                "L_hip2_joint": 100.0,
                "L_thigh_joint": 100.0,
            },
            damping={
                "L_hip_joint": 3.5,
                "L_hip2_joint": 3.5,
                "L_thigh_joint": 3.5,
            },
            armature={
                "L_hip_joint": 0.102588,
                "L_hip2_joint": 0.457870,
                "L_thigh_joint": 0.596446,
            },
        ),
        "L_calf": ImplicitActuatorCfg(
            joint_names_expr=["L_calf_joint"],
            effort_limit=159.0,
            velocity_limit=16.7,
            stiffness={
                "L_calf_joint": 300.0,
            },
            damping={
                "L_calf_joint": 7.5,
            },
            armature={
                "L_calf_joint": 0.076797,
            },
        ),
        "L_feet": ImplicitActuatorCfg(
            joint_names_expr=["L_toe_joint"],
            effort_limit=28.0,
            velocity_limit=8.69,
            stiffness={"L_toe_joint": 45.0},
            damping={"L_toe_joint": 0.8},
            armature={"L_toe_joint": 0.040897},
        ),  

        "right_leg": ImplicitActuatorCfg(
            joint_names_expr=["R_hip_joint", "R_hip2_joint", "R_thigh_joint"],
            effort_limit=105.0,
            velocity_limit=23.0,
            stiffness={
                "R_hip_joint": 100.0,
                "R_hip2_joint": 100.0,
                "R_thigh_joint": 100.0,
            },
            damping={
                "R_hip_joint": 3.5,
                "R_hip2_joint": 3.5,
                "R_thigh_joint": 3.5,
            },
            armature={
                "R_hip_joint": 0.130943,
                "R_hip2_joint": 0.438781,
                "R_thigh_joint": 0.336888,
            },
        ),

        "R_calf": ImplicitActuatorCfg(
            joint_names_expr=["R_calf_joint"],
            effort_limit=159.0,
            velocity_limit=16.7,
            stiffness={
                "R_calf_joint": 300.0,
            },
            damping={
                "R_calf_joint": 7.5,
            },
            armature={
                "R_calf_joint": 0.231993,
            },
        ),

        "R_feet": ImplicitActuatorCfg(
            joint_names_expr=["R_toe_joint"],
            effort_limit=28.0,
            velocity_limit=8.69,
            stiffness={"R_toe_joint": 45.0},
            damping={"R_toe_joint": 0.8},
            armature={"R_toe_joint": 0.062950},
        ),
    },
)
