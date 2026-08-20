#!/usr/bin/env python3

import itertools
import json
import time
from pathlib import Path


# ============================
# Action grid
# 保持 B21 已验证空间
# ============================

ACTION_VALUES = [-1.0, 0.0, 1.0]


def generate_actions():
    return list(itertools.product(
        ACTION_VALUES,
        ACTION_VALUES,
        ACTION_VALUES
    ))


# ============================
# Reward
# 不修改物理
# 只评价结果
# ============================

def evaluate(result):

    reward = 0.0

    if result["RACKET_CONTACT"]:
        reward += 20

    if result["CROSS_NET"]:
        reward += 30

    if result["OPPONENT_TABLE_LANDING"]:
        reward += 100

    if result["LEGAL_RETURN"]:
        reward += 200

    if result["NET_CLEARANCE"] is not None:
        reward += max(
            0,
            result["NET_CLEARANCE"]
        )

    return reward



def main():

    print("V2C1_ACTION_OPTIMIZER_START")

    actions = generate_actions()

    print(
        "ACTION_COUNT=",
        len(actions)
    )


    output = []

    for idx, action in enumerate(actions):

        print(
            f"TEST_ACTION {idx}/{len(actions)}",
            action
        )


        # TODO:
        #
        # 下一步这里调用：
        #
        # Model21800IsaacExecutor
        #
        # 与B21完全一致
        #
        # result = run_case(action)


        result = {
            "ACTION": action,
            "REWARD": 0,
            "LEGAL_RETURN": False
        }


        result["REWARD"] = evaluate(result)

        output.append(result)



    out = Path(
        "results/model21800_v2B/action_optimizer"
    )

    out.mkdir(
        parents=True,
        exist_ok=True
    )


    (out/"action_search.json").write_text(
        json.dumps(
            output,
            indent=2
        )
    )


    print(
        "V2C1_ACTION_OPTIMIZER_COMPLETE"
    )


if __name__ == "__main__":
    main()
