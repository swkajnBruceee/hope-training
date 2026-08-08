#!/usr/bin/env python3

# Topic  : /motion/control/neck_joint_control
# Message: sensor_msgs::msg::JointState


import json
import requests
import time
from datetime import datetime


def create_header():
    now = datetime.utcnow()
    header = {
        "stamp": {
            "sec": int(now.timestamp()),
            "nanosec": now.microsecond * 1000,
        },
        "frame_id": "neck_joint"
    }
    return header

def create_mcarm_control_channel(header, name, pos, vel, eff):
    return {
        "header": header,
        "name": name,
        "position": pos,
        "velocity": vel,
        "effort": eff
    }


def send_pose_data(url, pose_data):
    headers = {'Content-Type': 'application/json'}
    response = requests.Session().post(url, headers=headers, json=pose_data)
    return response


def main():
    url = "http://127.0.0.1:56322/channel/%2Fmotion%2Fcontrol%2Fneck_joint_command/ros2%3Asensor_msgs%2Fmsg%2FJointState"

    # 发送的位置数据
    name = [
        "idx27_head_joint1", 
        "idx28_head_joint2"
    ]
    vel = [0.0] * 2
    eff = [0.0] * 2
    pos = [-2.0, -0.5]   
    increasing = True
    step = 0.05
    # # 发送的姿态数据

    try:
        while True:
            #for pos in positions:
                
                header = create_header()
                mcmove_arm_data = create_mcarm_control_channel(
                header, name, pos, vel, eff)
                
                if increasing:
                    pos[0] = pos[0] + step
                    pos[1] = pos[1] + step
                    if pos[0] >= 2.0 or pos[1] >= 0.5:
                        pos = [2.0, 0.5]
                        increasing = False
                else:
                    pos[0] = pos[0] - step
                    pos[1] = pos[1] - step
                    if pos[0] <= -2.0 or pos[1] <= -0.5:
                        pos = [-2.0, -0.5]
                        increasing = True

                time.sleep(0.01)
                response = send_pose_data(url, mcmove_arm_data)
                print(
                    f"Sent data: {json.dumps(mcmove_arm_data, indent=2)}")
                print(f"Response: {response.status_code} - {response.text}\n")




    except KeyboardInterrupt:
        requests.Session().close()
        print("Stopped by user")


if __name__ == "__main__":
    main()