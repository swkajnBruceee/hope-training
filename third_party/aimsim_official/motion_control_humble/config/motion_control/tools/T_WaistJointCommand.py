#!/usr/bin/env python3

# Topic  : /motion/control/waist_joint_control
# Message: sensor_msgs::msg::JointState


import json
import requests
import time
from datetime import datetime


def create_header():
    now = datetime.utcnow()
    header = {
        "seq": 0,
        "timestamp": {
            "seconds": int(now.timestamp()),
            "nanos": now.microsecond * 1000,
            "ms_since_epoch": int(now.timestamp() * 1000),
        },
        "frame_id": "user_McScript",
        "control_source": "ControlSource_SAFE",
        "uuid": "",
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
    url = "http://127.0.0.1:56322/channel/%2Fmotion%2Fcontrol%2Fwaist_joint_command/ros2%3Asensor_msgs%2Fmsg%2FJointState"

    # 发送的位置数据
    name = [
      'waist_yaw_joint',
      'waist_roll_joint',
      'waist_pitch_joint',
    ]
    vel = [0.0] * 3
    eff = [0.0] * 3
    pos = 0.0   
    increasing = True
    step = 0.01
    # # 发送的姿态数据

    try:
        while True:
            #for pos in positions:
                
                header = create_header()
                position = [pos] * 3
                mcmove_arm_data = create_mcarm_control_channel(
                header, name, position, vel, eff)
                
                if increasing:
                    pos += step
                    if pos >= 0.5:
                        pos = 0.5
                        increasing = False
                else:
                    pos -= step
                    if pos <= -0.5:
                        pos = -0.5
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