#!/usr/bin/env python3

## 功能：设置头部关节值


import json
import requests
from datetime import datetime

# 生成请求头部


def create_header():
    now = datetime.utcnow()
    header = {
        "timestamp": {
            "seconds": int(now.timestamp()),
            "nanos": now.microsecond * 1000,
            "ms_since_epoch": int(now.timestamp() * 1000),
        },
        "control_source": "ControlSource_SAFE"
    }
    return header

# 创建手部控制命令数据（支持单手或双手）


def create_neck_data():
    neck_data = {
        "shake": {
            "name": "idx27_head_joint1",
            "position": 0.5,
            "velocity": 0.0,
            "effort": 0.0
        },
        "nod": {
            "name": "idx28_head_joint2",
            "position": 0.5,
            "velocity": 0.0,
            "effort": 0.0,
        }
    }

    return neck_data


def service_call():
    # 发送服务请求

    url = 'http://127.0.0.1:56322/rpc/aimdk.protocol.McMotionService/SetNeckCommand'
    headers = {
        'Content-Type': 'application/json',
        'timeout': '60000'
    }

    payload = {
        "header": create_header(),
        "data": create_neck_data()
    }

    response = requests.Session().post(url, headers=headers, json=payload)

    return response

# 主程序入口


def main():

    response = service_call()
    print(response.text)
    requests.Session().close()

if __name__ == "__main__":
    main()
