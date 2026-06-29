import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.env.mujoco_env import MujocoEnv
from src.controllers.robot_model import RobotModel, StateEstimator
from src.controllers.dcm_gait import DCMWalkingGait
from src.controllers.wbqp import WholeBodyQP

OUT = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(OUT, exist_ok=True)
scene = os.path.abspath(os.path.join(os.path.dirname(__file__), '../assets/unitree_g1/scene.xml'))
env = MujocoEnv(scene, rate_hz=500.0)
robot = RobotModel(env.model); est = StateEstimator(robot)
gait = DCMWalkingGait(robot, {"step_length":0.03,"step_width":0.22,"t_ss":0.30,
    "t_ds":0.12,"k_dcm":2.5,"k_cap":0.8})
wbqp = WholeBodyQP(robot, {"kp_torso":250.0,"kd_torso":30.0,"w_torso":12.0,
    "kp_com":[90.0,80.0,90.0],"kd_com":[28.0,20.0,19.0],"w_com":30.0,
    "kp_posture":40.0,"w_posture_joint":{"hip_yaw":90.0},
    "w_swing_rot":3.0,"kp_swing_rot":80.0,"kd_swing_rot":18.0,
    "kp_swing":400.0,"kd_swing":40.0,"w_swing":50.0})
robot.set_home(env.data); gait.reset(est.update(env.data))
dt = env.model.opt.timestep
T=[];cx=[];cy=[];rx=[];ry=[];dx=[];dy=[];drx=[];dry=[];zx=[];zy=[]
roll=[];pitch=[];sat=[];flz=[];frz=[]
start=env.data.time
while env.data.time-start < 12.0:
    st=est.update(env.data)
    refs,info=gait.update(st,dt)
    tau=wbqp.compute(env.data,refs["com_des"],refs["com_vel_des"],refs["torso_R"],
                     refs["contacts"],refs["swing"],com_acc_ff=refs["com_acc_ff"])
    env.data.ctrl[:]=tau; env.step()
    t=env.data.time-start
    T.append(t); cx.append(st["com"][0]); cy.append(st["com"][1])
    rx.append(refs["com_des"][0]); ry.append(refs["com_des"][1])
    dx.append(info["dcm"][0]); dy.append(info["dcm"][1])
    drx.append(info["dcm_ref"][0]); dry.append(info["dcm_ref"][1])
    zx.append(info["zmp_cmd"][0]); zy.append(info["zmp_cmd"][1])
    roll.append(st["base_rpy"][0]); pitch.append(st["base_rpy"][1])
    sat.append(np.max(np.abs(tau)/np.maximum(robot.tau_limit,1e-9)))
    flz.append(st["lfoot_force"]); frz.append(st["rfoot_force"])
T=np.array(T)

# Figure 1: lateral DCM/CoM/ZMP vs time (the limit cycle sway)
fig,ax=plt.subplots(2,1,figsize=(7.4,2.9),sharex=True)
ax[0].plot(T,dy,label=r"$\xi_y$ (measured DCM)",lw=1.0)
ax[0].plot(T,dry,'--',label=r"$\xi_y^{ref}$",lw=1.0)
ax[0].plot(T,cy,label=r"$c_y$ (CoM)",lw=1.0,color='k')
ax[0].plot(T,zy,':',label=r"$p_y^{cmd}$ (ZMP)",lw=0.9,color='tab:red')
ax[0].set_ylabel("lateral [m]"); ax[0].legend(ncol=2,fontsize=7,loc='upper right')
ax[0].grid(alpha=0.3)
ax[1].plot(T,np.array(flz),label=r"$F_L$",lw=0.8)
ax[1].plot(T,np.array(frz),label=r"$F_R$",lw=0.8)
ax[1].set_ylabel("normal force [N]"); ax[1].set_xlabel("time [s]")
ax[1].legend(fontsize=7,loc='upper right'); ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_lateral.pdf"))

# Figure 2: sagittal CoM tracking + tilt + saturation
fig,ax=plt.subplots(2,1,figsize=(7.4,2.9),sharex=True)
ax[0].plot(T,cx,label=r"$c_x$ (CoM)",lw=1.0,color='k')
ax[0].plot(T,rx,'--',label=r"$c_x^{ref}$",lw=1.0)
ax[0].plot(T,dx,label=r"$\xi_x$",lw=0.8,color='tab:green',alpha=0.8)
ax[0].set_ylabel("forward [m]"); ax[0].legend(fontsize=7,loc='upper left'); ax[0].grid(alpha=0.3)
ax2=ax[1].twinx()
ax[1].plot(T,np.array(roll),label="roll",lw=0.8,color='tab:blue')
ax[1].plot(T,np.array(pitch),label="pitch",lw=0.8,color='tab:orange')
ax2.plot(T,np.array(sat),label="torque sat.",lw=0.6,color='tab:gray',alpha=0.6)
ax[1].set_ylabel("tilt [rad]"); ax2.set_ylabel("sat. ratio"); ax[1].set_xlabel("time [s]")
ax[1].legend(fontsize=7,loc='upper left'); ax2.legend(fontsize=7,loc='upper right')
ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_sagittal.pdf"))
print("steps=",gait.step_count,"figs written")
print("RMS lateral DCM err=",np.sqrt(np.mean((np.array(dy)-np.array(dry))**2)))
print("RMS sagittal CoM err=",np.sqrt(np.mean((np.array(cx)-np.array(rx))**2)))
