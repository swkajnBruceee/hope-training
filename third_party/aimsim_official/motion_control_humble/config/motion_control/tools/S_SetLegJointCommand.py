#!/usr/bin/env python3

import json
import random
import requests
from datetime import datetime

def create_header():
    now = datetime.utcnow()
    header = {
        "timestamp": {
            "seconds": int(now.timestamp()),
            "nanos": now.microsecond * 1000,
            "ms_since_epoch": int(now.timestamp() * 1000),
        },
        "control_source": "ControlSource_SAFE",
        "uuid": "",
        "trace_id": "user_McScript",
        "domin": "",
    }
    return header

joint_names = [
    'left_hip_pitch_joint',
    'left_hip_roll_joint',
    'left_hip_yaw_joint',
    'left_knee_joint',
    'left_ankle_pitch_joint',
    'left_ankle_roll_joint',
    'right_hip_pitch_joint',
    'right_hip_roll_joint',
    'right_hip_yaw_joint',
    'right_knee_joint',
    'right_ankle_pitch_joint',
    'right_ankle_roll_joint',
]

# 构造目标列表的函数
def build_joint_list(names):
    # 生成与关节数量一致的随机且不重复的position值（范围：0~10，保留2位小数）
    # 先生成随机数列表，确保不重复
    position_values = random.sample([round(i/100, 2) for i in range(0, 1000)], len(names))
    joint_list = []
    
    # 遍历名称列表，逐个构造字典并添加到列表
    for idx, name in enumerate(names):
        joint_dict = {
            "name": name,
            "sequence": idx,  # sequence从0开始按顺序编号
            "position": position_values[idx],  # 随机且不同的position值
            "velocity": 0,
            "effort": 0,
            "stiffness": 0,
            "damping": 0,
            # 注意：你原示例中重复定义了position，这里只保留一个
        }
        joint_list.append(joint_dict)
    return joint_list

def service_call():
    url = 'http://127.0.0.1:56322/rpc/aimdk.protocol.MotionControlJointService/SetLegJointCommand'
    headers = {'Content-Type': 'application/json'}
    joint_data_list = build_joint_list(joint_names)
    payload = {
        "header": create_header(),
        "commands": joint_data_list,
    }
    
    response = requests.Session().post(url, headers=headers, json=payload)
    response.raise_for_status()
    
    return response.json()

def pretty_print_json(json_data):
    print("Response:")
    print(json.dumps(json_data, indent=2, ensure_ascii=False))


def main():
    response = service_call()
    pretty_print_json(response)
    requests.Session().close()

if __name__ == "__main__":
    main()
