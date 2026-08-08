#!/usr/bin/env python3

## 功能：发送坐下站起通知


import enum
import json
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
        "control_source": "ControlSource_SAFE"
    }
    return header


def service_call(notify_msg):
    url = 'http://127.0.0.1:56322/rpc/aimdk.protocol.McMotionService/SitStandStateNotify'
    headers = {'Content-Type': 'application/json'}
    payload = {
        "header": create_header(),
        "state_notify": notify_msg,
    }
    
    response = requests.session().post(url, headers=headers, json=payload)
    
    return response

def main():
    notify_msg = input("Enter sit-stand state notify message: ")
    print(f"Selected sit-stand state notify message: {notify_msg}")
    response = service_call(notify_msg)
    print(response.text)
    requests.session().close()

if __name__ == "__main__":
    main()
