#!/usr/bin/env python3

## 功能：设置舞蹈类型


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
        "control_source": "ControlSource_SAFE",
        "uuid": "",
        "trace_id": "user_McScript",
        "domin": "",
    }
    return header

#    taiji
#    classical
#    jiangnan
def service_call(dance_name):
    url = 'http://127.0.0.1:56322/rpc/aimdk.protocol.MotionControlMotionService/SelectDanceType'
    headers = {'Content-Type': 'application/json'}
    payload = {
        "header": create_header(),
        "dance_name": dance_name,
    }
    
    response = requests.session().post(url, headers=headers, json=payload)
    
    return response

def main():
    dancetype = input("Enter dance type (taiji/classical/jiangnan): ")
    print(f"Selected dance type: {dancetype}")
    response = service_call(dancetype)
    print(response.text)
    requests.session().close()

if __name__ == "__main__":
    main()
