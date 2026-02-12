import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg, IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaac_lab_actuator_dynamic.assets import LOCAL_ASSETS_DATA_DIR
import math

LEG_CHANGED_IMU_USD_PATH = f"/home/humanoid2/DanielTruong/isaac_lab_actuator_dynamic/source/isaac_lab_actuator_dynamic/isaac_lab_actuator_dynamic/assets/usd/Leg_Changed_IMU/leg_changed_imu.usd"
LEG_CHANGED_IMU_URDF_PATH = f"/home/humanoid2/DanielTruong/isaac_lab_actuator_dynamic/source/isaac_lab_actuator_dynamic/isaac_lab_actuator_dynamic/assets/urdf/Leg_Changed_IMU/Wholebody_URDF_IMU_Changed.urdf"

# EFFECTIVE_HIP_ARMATURE = 0.092030
# EFFECTIVE_HIP2_ARMATURE = 0.1020675

# # EFFECTIVE_THIGH_ARMATURE = 0.16071
# EFFECTIVE_THIGH_ARMATURE = 0.061379 #! Torque Scale 0.751666

# # EFFECTIVE_CALF_ARMATURE = 0.14273
# EFFECTIVE_CALF_ARMATURE = 0.061914 #! Torque Scale 0.573606
 
# EFFECTIVE_TOE_ARMATURE = 0.037790 #! Torque Scale 1.0

EFFECTIVE_HIP_ARMATURE = 0.102588
EFFECTIVE_HIP2_ARMATURE = 0.457870
EFFECTIVE_THIGH_ARMATURE = 0.596446 
EFFECTIVE_CALF_ARMATURE = 0.14273 
EFFECTIVE_TOE_ARMATURE = 0.037790 

# EFFECTIVE_VISCOUSE_FRICTION_LHIP = 0.158
# EFFECTIVE_VISCOUSE_FRICTION_LHIP2 = 0.6522
# EFFECTIVE_VISCOUSE_FRICTION_LTHIGH = 0.686212
# EFFECTIVE_VISCOUSE_FRICTION_LCALF = 3.834728
# EFFECTIVE_VISCOUSE_FRICTION_LTOE = 0.083082
# EFFECTIVE_VISCOUSE_FRICTION_RHIP = 0.158
# EFFECTIVE_VISCOUSE_FRICTION_RHIP2 = 0.6522
# EFFECTIVE_VISCOUSE_FRICTION_RTHIGH = 0.686212
# EFFECTIVE_VISCOUSE_FRICTION_RCALF = 3.834728
# EFFECTIVE_VISCOUSE_FRICTION_RTOE = 0.083082

# EFFECTIVE_DYNAMIC_FRICTION_LHIP = 0.623915
# EFFECTIVE_DYNAMIC_FRICTION_LHIP2 = 1.844928
# EFFECTIVE_DYNAMIC_FRICTION_LTHIGH = 4.754382
# EFFECTIVE_DYNAMIC_FRICTION_LCALF = 8.390081
# EFFECTIVE_DYNAMIC_FRICTION_LTOE = 0.43967
# EFFECTIVE_DYNAMIC_FRICTION_RHIP = 1.344627
# EFFECTIVE_DYNAMIC_FRICTION_RHIP2 = 3.484132
# EFFECTIVE_DYNAMIC_FRICTION_RTHIGH = 4.407304
# EFFECTIVE_DYNAMIC_FRICTION_RCALF = 5.085046
# EFFECTIVE_DYNAMIC_FRICTION_RTOE = 0.43967

HIP_ARMATURE = 0.102588
HIP2_ARMATURE = 0.15
THIGH_ARMATURE = 0.16
CALF_ARMATURE = 0.16
TOE_ARMATURE = 0.040897

ETA = 1.1
NATURAL_FREQUENCY = 2.0 * math.pi * 5.45 # 6.65 Hz
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

LEG_CHANGED_IMU_HIGHGAIN_CFG = ArticulationCfg(
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
        pos=(0.0, 0.0, 0.73),
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
        "left_leg": ImplicitActuatorCfg(
            joint_names_expr=["L_hip_joint", "L_hip2_joint", "L_thigh_joint"],
            effort_limit_sim=98.0,
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
            # viscous_friction={
            #     "L_hip_joint": EFFECTIVE_VISCOUSE_FRICTION_LHIP,
            #     "L_hip2_joint": EFFECTIVE_VISCOUSE_FRICTION_LHIP2,
            #     "L_thigh_joint": EFFECTIVE_VISCOUSE_FRICTION_LTHIGH,
            # },
            # dynamic_friction={
            #     "L_hip_joint": EFFECTIVE_DYNAMIC_FRICTION_LHIP,
            #     "L_hip2_joint": EFFECTIVE_DYNAMIC_FRICTION_LHIP2,
            #     "L_thigh_joint": EFFECTIVE_DYNAMIC_FRICTION_LTHIGH,
            # },
            # friction={
            #     "L_hip_joint": EFFECTIVE_DYNAMIC_FRICTION_LHIP,
            #     "L_hip2_joint": EFFECTIVE_DYNAMIC_FRICTION_LHIP2,
            #     "L_thigh_joint": EFFECTIVE_DYNAMIC_FRICTION_LTHIGH,
            # },
        ),
        "L_calf": ImplicitActuatorCfg(
            joint_names_expr=["L_calf_joint"],
            effort_limit_sim=150.0,
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
            # viscous_friction={
            #     "L_calf_joint": EFFECTIVE_VISCOUSE_FRICTION_LCALF,
            # },
            # dynamic_friction={
            #     "L_calf_joint": EFFECTIVE_DYNAMIC_FRICTION_LCALF,
            # },
            # friction={
            #     "L_calf_joint": EFFECTIVE_DYNAMIC_FRICTION_LCALF,
            # },
        ),
        "L_feet": ImplicitActuatorCfg(
            joint_names_expr=["L_toe_joint"],
            effort_limit_sim=28.0,
            velocity_limit_sim=8.69,
            stiffness={"L_toe_joint": TOE_STIFFNESS},
            damping={"L_toe_joint": TOE_DAMPING},
            armature={"L_toe_joint": EFFECTIVE_TOE_ARMATURE},
            # viscous_friction={
            #     "L_toe_joint": EFFECTIVE_VISCOUSE_FRICTION_LTOE,
            # },
            # dynamic_friction={
            #     "L_toe_joint": EFFECTIVE_DYNAMIC_FRICTION_LTOE,
            # },
            # friction={
            #     "L_toe_joint": EFFECTIVE_DYNAMIC_FRICTION_LTOE,
            # },
        ),

        "right_leg": ImplicitActuatorCfg(
            joint_names_expr=["R_hip_joint", "R_hip2_joint", "R_thigh_joint"],
            effort_limit_sim=98.0,
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
            # viscous_friction={
            #     "R_hip_joint": EFFECTIVE_VISCOUSE_FRICTION_RHIP,
            #     "R_hip2_joint": EFFECTIVE_VISCOUSE_FRICTION_RHIP2,
            #     "R_thigh_joint": EFFECTIVE_VISCOUSE_FRICTION_RTHIGH,
            # },
            # dynamic_friction={
            #     "R_hip_joint": EFFECTIVE_DYNAMIC_FRICTION_RHIP,
            #     "R_hip2_joint": EFFECTIVE_DYNAMIC_FRICTION_RHIP2,
            #     "R_thigh_joint": EFFECTIVE_DYNAMIC_FRICTION_RTHIGH,
            # },
            # friction={
            #     "R_hip_joint": EFFECTIVE_DYNAMIC_FRICTION_RHIP,
            #     "R_hip2_joint": EFFECTIVE_DYNAMIC_FRICTION_RHIP2,
            #     "R_thigh_joint": EFFECTIVE_DYNAMIC_FRICTION_RTHIGH,
            # },
        ),

        "R_calf": ImplicitActuatorCfg(
            joint_names_expr=["R_calf_joint"],
            effort_limit_sim=150.0,
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
            # viscous_friction={
            #     "R_calf_joint": EFFECTIVE_VISCOUSE_FRICTION_RCALF,
            # },
            # dynamic_friction={
            #     "R_calf_joint": EFFECTIVE_DYNAMIC_FRICTION_RCALF,
            # },
            # friction={
            #     "R_calf_joint": EFFECTIVE_DYNAMIC_FRICTION_RCALF,
            # },
        ),

        "R_feet": ImplicitActuatorCfg(
            joint_names_expr=["R_toe_joint"],
            effort_limit_sim=28.0,
            velocity_limit_sim=8.69,
            stiffness={"R_toe_joint": TOE_STIFFNESS},
            damping={"R_toe_joint": TOE_DAMPING},
            armature={"R_toe_joint": EFFECTIVE_TOE_ARMATURE},
            # viscous_friction={
            #     "R_toe_joint": EFFECTIVE_VISCOUSE_FRICTION_RTOE,
            # },
            # dynamic_friction={
            #     "R_toe_joint": EFFECTIVE_DYNAMIC_FRICTION_RTOE,
            # },
            # friction={
            #     "R_toe_joint": EFFECTIVE_DYNAMIC_FRICTION_RTOE,
            # },
        ),
    },
)


LEG_CHANGED_IMU_HIGHGAIN_ACTION_SCALE = {}
for a in LEG_CHANGED_IMU_HIGHGAIN_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            LEG_CHANGED_IMU_HIGHGAIN_ACTION_SCALE[n] = 0.25 * e[n] / s[n]



LEG_CHANGED_IMU_LOWGAIN_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=LEG_CHANGED_IMU_USD_PATH,
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            fix_root_link=False,
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
            effort_limit_sim=98.0,
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
                "L_hip_joint": EFFECTIVE_HIP_ARMATURE,
                "L_hip2_joint": EFFECTIVE_HIP2_ARMATURE,
                "L_thigh_joint": EFFECTIVE_THIGH_ARMATURE,
            },
        ),
        "L_calf": ImplicitActuatorCfg(
            joint_names_expr=["L_calf_joint"],
            effort_limit_sim=150.0,
            velocity_limit_sim=16.7,
            stiffness={
                "L_calf_joint": 30.0,
            },
            damping={
                "L_calf_joint": 5.0,
            },
            armature={
                "L_calf_joint": EFFECTIVE_CALF_ARMATURE,
            },
        ),
        "L_feet": ImplicitActuatorCfg(
            joint_names_expr=["L_toe_joint"],
            effort_limit_sim=28.0,
            velocity_limit_sim=8.69,
            stiffness={"L_toe_joint": 30.0},
            damping={"L_toe_joint": 5.0},
            armature={"L_toe_joint": EFFECTIVE_TOE_ARMATURE},
        ),

        "right_leg": ImplicitActuatorCfg(
            joint_names_expr=["R_hip_joint", "R_hip2_joint", "R_thigh_joint"],
            effort_limit_sim=98.0,
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
                "R_hip_joint": EFFECTIVE_HIP_ARMATURE,
                "R_hip2_joint": EFFECTIVE_HIP2_ARMATURE,
                "R_thigh_joint": EFFECTIVE_THIGH_ARMATURE,
            },
        ),

        "R_calf": ImplicitActuatorCfg(
            joint_names_expr=["R_calf_joint"],
            effort_limit_sim=150.0,
            velocity_limit_sim=16.7,
            stiffness={
                "R_calf_joint": 30.0,
            },
            damping={
                "R_calf_joint": 5.0,
            },
            armature={
                "R_calf_joint": EFFECTIVE_CALF_ARMATURE,
            },
        ),

        "R_feet": ImplicitActuatorCfg(
            joint_names_expr=["R_toe_joint"],
            effort_limit_sim=28.0,
            velocity_limit_sim=8.69,
            stiffness={"R_toe_joint": 30.0},
            damping={"R_toe_joint": 5.0},
            armature={"R_toe_joint": EFFECTIVE_TOE_ARMATURE},
        ),
    },
)