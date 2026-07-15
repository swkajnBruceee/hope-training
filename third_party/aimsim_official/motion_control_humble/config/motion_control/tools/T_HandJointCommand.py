#!/usr/bin/env python3

# Topic  : /motion/control/hand_joint_control
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
        "frame_id": "hand_joint"
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
    url = "http://127.0.0.1:56322/channel/%2Fmotion%2Fcontrol%2Fhand_joint_command/ros2%3Asensor_msgs%2Fmsg%2FJointState"

    # 发送的位置数据
    name =  [
        "L_thumb_swing_joint", 
        "L_thumb_1_joint", 
        "L_index_1_joint", 
        "L_middle_1_joint", 
        "L_ring_1_joint", 
        "L_little_1_joint", 
        "R_thumb_swing_joint", 
        "R_thumb_1_joint", 
        "R_index_1_joint", 
        "R_middle_1_joint", 
        "R_ring_1_joint", 
        "R_little_1_joint"
    ]
    vel = [0.0] * 12
    eff = [0.0] * 12
    pos = 0.0
    increasing = True
    step = 100
    # # 发送的姿态数据

    try:
        while True:
            #for pos in positions:
                
                header = create_header()
                position = [pos] * 12
                mcmove_arm_data = create_mcarm_control_channel(
                header, [], position, [], eff)
                
                if increasing:
                    pos += step
                    if pos >= 2000:
                        pos = 2000
                        increasing = False
                else:
                    pos -= step
                    if pos <= 0:
                        pos = 0
                        increasing = True

                time.sleep(0.1)
                response = send_pose_data(url, mcmove_arm_data)
                print(
                    f"Sent data: {json.dumps(mcmove_arm_data, indent=2)}")
                print(f"Response: {response.status_code} - {response.text}\n")




    except KeyboardInterrupt:
        requests.Session().close()
        print("Stopped by user")


if __name__ == "__main__":
    main()