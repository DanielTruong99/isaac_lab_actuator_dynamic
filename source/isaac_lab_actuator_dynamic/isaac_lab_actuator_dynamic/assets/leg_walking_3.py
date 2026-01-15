import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg, IdealPDActuatorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaac_lab_actuator_dynamic.assets import LOCAL_ASSETS_DATA_DIR
import math

LEGACTUATORDYNAMIC_USD_PATH = f"/home/humanoid2/DanielTruong/isaac_lab_actuator_dynamic/source/isaac_lab_actuator_dynamic/isaac_lab_actuator_dynamic/assets/urdf/leg05/robot.usd"
LEGACTUATORDYNAMIC_2_USD_PATH = f"/home/humanoid2/DanielTruong/isaac_lab_actuator_dynamic/source/isaac_lab_actuator_dynamic/isaac_lab_actuator_dynamic/assets/urdf/leg05/robot_5.usd"
ROBOT_FLATFOOT_WITHPIN_USD_PATH = f"/home/humanoid2/DanielTruong/isaac_lab_actuator_dynamic/source/isaac_lab_actuator_dynamic/isaac_lab_actuator_dynamic/assets/urdf/leg05/robot_flatfoot_withpin.usd"

EFFECTIVE_HIP_ARMATURE = 0.092030
EFFECTIVE_HIP2_ARMATURE = 0.1020675
EFFECTIVE_THIGH_ARMATURE = 0.16071
EFFECTIVE_CALF_ARMATURE = 0.14273
EFFECTIVE_TOE_ARMATURE = 0.037552

HIP_ARMATURE = 0.07
HIP2_ARMATURE = 0.075
THIGH_ARMATURE = 0.08
CALF_ARMATURE = 0.077
TOE_ARMATURE = 0.0375

ETA = 1.25
NATURAL_FREQUENCY = 2.0 * math.pi * 6.65 # 6.65 Hz
STIFFNESS = NATURAL_FREQUENCY * NATURAL_FREQUENCY
DAMPING = 2.0 * ETA * NATURAL_FREQUENCY
HIP_STIFFNESS = STIFFNESS * HIP_ARMATURE
HIP_DAMPING = DAMPING * HIP_ARMATURE

HIP2_STIFFNESS = STIFFNESS * HIP2_ARMATURE
HIP2_DAMPING = DAMPING * HIP2_ARMATURE

THIGH_STIFFNESS = STIFFNESS * THIGH_ARMATURE
THIGH_DAMPING = DAMPING * THIGH_ARMATURE

CALF_STIFFNESS = STIFFNESS * CALF_ARMATURE
CALF_DAMPING = DAMPING * CALF_ARMATURE

TOE_STIFFNESS = STIFFNESS * TOE_ARMATURE
TOE_DAMPING = DAMPING * TOE_ARMATURE

FLATFOOT_WITHPIN_HIGHGAIN_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ROBOT_FLATFOOT_WITHPIN_USD_PATH,
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
        pos=(0.0, 0.0, 0.8305),
        joint_pos={
            'L_hip_joint': 0.0, # limit -35, 35 (degrees)
            'L_hip2_joint': 0.0, # limit -35, 35 (degrees)
            'L_thigh_joint': 0.65, # limit -70, 70 (degrees)
            'L_calf_joint': -1.05, # limit -110, 0 (degrees)
            'L_toe_joint': 0.4, # limit -45, 45 (degrees)
            'R_hip_joint': 0.0, # limit -35, 35 (degrees)
            'R_hip2_joint': 0.0, # limit -35, 35 (degrees)
            'R_thigh_joint': 0.65, # limit -70, 70 (degrees)
            'R_calf_joint': -1.05, # limit -110, 0 (degrees)
            'R_toe_joint': 0.4, # limit -45, 45 (degrees)
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.97,
    actuators={
        "left_leg": ImplicitActuatorCfg(
            joint_names_expr=["L_hip_joint", "L_hip2_joint", "L_thigh_joint"],
            effort_limit_sim=105.0,
            velocity_limit_sim=23.0,
            stiffness={
                "L_hip_joint": HIP_STIFFNESS,
                "L_hip2_joint": HIP2_STIFFNESS,
                "L_thigh_joint": THIGH_STIFFNESS,
            },
            damping={
                "L_hip_joint": HIP_DAMPING,
                "L_hip2_joint": HIP2_DAMPING,
                "L_thigh_joint": THIGH_DAMPING,
            },
            armature={
                "L_hip_joint": EFFECTIVE_HIP_ARMATURE,
                "L_hip2_joint": EFFECTIVE_HIP2_ARMATURE,
                "L_thigh_joint": EFFECTIVE_THIGH_ARMATURE,
            },
        ),
        "L_calf": ImplicitActuatorCfg(
            joint_names_expr=["L_calf_joint"],
            effort_limit_sim=159.0,
            velocity_limit_sim=16.7,
            stiffness={
                "L_calf_joint": CALF_STIFFNESS,
            },
            damping={
                "L_calf_joint": CALF_DAMPING,
            },
            armature={
                "L_calf_joint": EFFECTIVE_CALF_ARMATURE,
            },
        ),
        "L_feet": ImplicitActuatorCfg(
            joint_names_expr=["L_toe_joint"],
            effort_limit_sim=28.0,
            velocity_limit_sim=8.69,
            stiffness={"L_toe_joint": TOE_STIFFNESS},
            damping={"L_toe_joint": TOE_DAMPING},
            armature={"L_toe_joint": EFFECTIVE_TOE_ARMATURE},
        ),

        "right_leg": ImplicitActuatorCfg(
            joint_names_expr=["R_hip_joint", "R_hip2_joint", "R_thigh_joint"],
            effort_limit_sim=105.0,
            velocity_limit_sim=23.0,
            stiffness={
                "R_hip_joint": HIP_STIFFNESS,
                "R_hip2_joint": HIP2_STIFFNESS,
                "R_thigh_joint": THIGH_STIFFNESS,
            },
            damping={
                "R_hip_joint": HIP_DAMPING,
                "R_hip2_joint": HIP2_DAMPING,
                "R_thigh_joint": THIGH_DAMPING,
            },
            armature={
                "R_hip_joint": EFFECTIVE_HIP_ARMATURE,
                "R_hip2_joint": EFFECTIVE_HIP2_ARMATURE,
                "R_thigh_joint": EFFECTIVE_THIGH_ARMATURE,
            },
        ),

        "R_calf": ImplicitActuatorCfg(
            joint_names_expr=["R_calf_joint"],
            effort_limit_sim=159.0,
            velocity_limit_sim=16.7,
            stiffness={
                "R_calf_joint": CALF_STIFFNESS,
            },
            damping={
                "R_calf_joint": CALF_DAMPING,
            },
            armature={
                "R_calf_joint": EFFECTIVE_CALF_ARMATURE,
            },
        ),

        "R_feet": ImplicitActuatorCfg(
            joint_names_expr=["R_toe_joint"],
            effort_limit_sim=28.0,
            velocity_limit_sim=8.69,
            stiffness={"R_toe_joint": TOE_STIFFNESS},
            damping={"R_toe_joint": TOE_DAMPING},
            armature={"R_toe_joint": EFFECTIVE_TOE_ARMATURE},
        ),
    },
)


HIGHGAIN_ACTION_SCALE = {}
for a in FLATFOOT_WITHPIN_HIGHGAIN_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            HIGHGAIN_ACTION_SCALE[n] = 0.25 * e[n] / s[n]



FLATFOOT_WITHPIN_LOWGAIN_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ROBOT_FLATFOOT_WITHPIN_USD_PATH,
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
        pos=(0.0, 0.0, 0.8305),
        joint_pos={
            'L_hip_joint': 0.0, # limit -35, 35 (degrees)
            'L_hip2_joint': 0.0, # limit -35, 35 (degrees)
            'L_thigh_joint': 0.65, # limit -70, 70 (degrees)
            'L_calf_joint': -1.05, # limit -110, 0 (degrees)
            'L_toe_joint': 0.4, # limit -45, 45 (degrees)
            'R_hip_joint': 0.0, # limit -35, 35 (degrees)
            'R_hip2_joint': 0.0, # limit -35, 35 (degrees)
            'R_thigh_joint': 0.65, # limit -70, 70 (degrees)
            'R_calf_joint': -1.05, # limit -110, 0 (degrees)
            'R_toe_joint': 0.4, # limit -45, 45 (degrees)
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.97,
    actuators={
        "left_leg": ImplicitActuatorCfg(
            joint_names_expr=["L_hip_joint", "L_hip2_joint", "L_thigh_joint"],
            effort_limit_sim=105.0,
            velocity_limit_sim=23.0,
            stiffness={
                "L_hip_joint": 30.0,
                "L_hip2_joint": 30.0,
                "L_thigh_joint": 30.0,
            },
            damping={
                "L_hip_joint": 5.0,
                "L_hip2_joint": 5.0,
                "L_thigh_joint": 5.0,
            },
            armature={
                "L_hip_joint": HIP_ARMATURE,
                "L_hip2_joint": HIP2_ARMATURE,
                "L_thigh_joint": THIGH_ARMATURE,
            },
        ),
        "L_calf": ImplicitActuatorCfg(
            joint_names_expr=["L_calf_joint"],
            effort_limit_sim=159.0,
            velocity_limit_sim=16.7,
            stiffness={
                "L_calf_joint": 30.0,
            },
            damping={
                "L_calf_joint": 5.0,
            },
            armature={
                "L_calf_joint": CALF_ARMATURE,
            },
        ),
        "L_feet": ImplicitActuatorCfg(
            joint_names_expr=["L_toe_joint"],
            effort_limit_sim=28.0,
            velocity_limit_sim=8.69,
            stiffness={"L_toe_joint": 30.0},
            damping={"L_toe_joint": 5.0},
            armature={"L_toe_joint": TOE_ARMATURE},
        ),

        "right_leg": ImplicitActuatorCfg(
            joint_names_expr=["R_hip_joint", "R_hip2_joint", "R_thigh_joint"],
            effort_limit_sim=105.0,
            velocity_limit_sim=23.0,
            stiffness={
                "R_hip_joint": 30.0,
                "R_hip2_joint": 30.0,
                "R_thigh_joint": 30.0,
            },
            damping={
                "R_hip_joint": 5.0,
                "R_hip2_joint": 5.0,
                "R_thigh_joint": 5.0,
            },
            armature={
                "R_hip_joint": HIP_ARMATURE,
                "R_hip2_joint": HIP2_ARMATURE,
                "R_thigh_joint": THIGH_ARMATURE,
            },
        ),

        "R_calf": ImplicitActuatorCfg(
            joint_names_expr=["R_calf_joint"],
            effort_limit_sim=159.0,
            velocity_limit_sim=16.7,
            stiffness={
                "R_calf_joint": 30.0,
            },
            damping={
                "R_calf_joint": 5.0,
            },
            armature={
                "R_calf_joint": CALF_ARMATURE,
            },
        ),

        "R_feet": ImplicitActuatorCfg(
            joint_names_expr=["R_toe_joint"],
            effort_limit_sim=28.0,
            velocity_limit_sim=8.69,
            stiffness={"R_toe_joint": 30.0},
            damping={"R_toe_joint": 5.0},
            armature={"R_toe_joint": TOE_ARMATURE},
        ),
    },
)