# A3 上下身策略部署包

这个目录是给真机部署使用的模型交付包，包含当前训练工程中确认过来源的两个冻结策略：

- `model_900_upper_fixed_base.pt`：上身固定基座 backhand 策略，负责腰部和右臂击球。
- `model_3396_lower_stage_a.pt`：历史 Stage-A 下身支撑策略，负责浮动基座下的腿部支撑。
- `model_3396_lower_stage_a_policy.onnx`：`model_3396` 已有的 ONNX 导出。

## 文件结构

```text
model_deployment/
├── README.md
├── MODEL_MANIFEST.yaml       # 角色、来源、结构、契约和校验信息
├── SHA256SUMS                # 全部交付文件的 SHA256
├── weights/
│   ├── model_900_upper_fixed_base.pt
│   ├── model_3396_lower_stage_a.pt
│   └── model_3396_lower_stage_a_policy.onnx
├── metadata/
│   ├── model_900/            # model_900 的训练参数和环境快照
│   └── model_3396/           # model_3396 的训练参数和 lineage
└── evidence/                 # 对应的仿真评估和确定性轨迹
```

## 关键使用方式

这两个权重不是一个可以直接替换到官方 A3 `action[29]` 接口的单体模型。它们是在联合协调器内部以 inference-only 方式加载的两个不同 observation/action contract：

| 权重 | 输入 observation | 输出 action | 角色 |
|---|---:|---:|---|
| `model_900` | 56 | 10 | 固定基座上身击球先验 |
| `model_3396` | 126 | 14 | 历史浮动基座 Stage-A 下身先验 |

`.pt` 中包含 actor、critic、optimizer 和 observation normalizer；部署时必须保留与训练契约一致的 observation 顺序、归一化参数、动作缩放、关节映射和时序 prelude。不能只提取 actor 的线性层而忽略 normalizer。

当前仓库中的联合协调器配置已经记录了这两个权重的加载关系，见 `hope_training/whole_body_tracking/cfg/task/HOPEA3JointCoordinator.yaml`。当前 F1 文档同时明确：历史 `model_3396` 不能作为新合同训练的 warm start；本包中的 `model_3396` 仅作为历史/联合协调器先验交付。

## 真机部署注意事项

1. 先用 `MODEL_MANIFEST.yaml` 和 `SHA256SUMS` 校验文件没有损坏或被替换。
2. 真机 C++ runtime 当前使用 ONNX/RKNN 等部署格式，并不是直接读取 Isaac Lab 的 `.pt` checkpoint。
3. `model_3396` 已提供 ONNX，但仍需由部署侧确认 ONNX 输入预处理与 126-D 历史 observation contract 一致。
4. 当前 `model_900` 目录只有 PyTorch checkpoint，尚未提供经过部署侧确认的 ONNX；不要把它和官方 29-DOF 单体 A3 ONNX 模型混用。
5. 上下身策略只能在训练代码定义的联合 wrapper、ready-pose、lookahead、动作缩放和安全限制下使用。交付包不代表已经完成真机安全验收。

详细来源、结构和限制见 `MODEL_MANIFEST.yaml`；评估原始证据在 `evidence/`。
