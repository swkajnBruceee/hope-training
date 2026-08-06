# A3 发球控制程序

这是一个专用的真机发球程序，只保留一条运行路径：

1. 启动自定义 HAL；
2. 启动 100 Hz 控制器和 50 Hz 下肢平衡模型；
3. 手臂自动播放一次 `arm/serve_upper_trajectory.bin`；
4. 轨迹结束后回到轨迹第一帧并保持；
5. 按空格重新播放一次；
6. 按 Ctrl+C 后，整机先平滑回到程序启动时记录的姿态；
7. 复位稳定后停止控制器和 HAL，随后允许掉使能并退出。

## 运行

在 MDU 上执行：

```bash
cd /agibot/data/user_deploy/mc
bash scripts/run_model3396_real.sh
```

不需要额外参数，也不再支持 fixed-point、运行时长覆盖、dry-run 等分支。

## 固定运行配置

- 控制频率：100 Hz
- 下肢策略频率：50 Hz
- 下肢策略增益：4.0
- 轨迹速度倍率：1.0
- 发球轨迹：`arm/serve_upper_trajectory.bin`
- 复位时间：2.0 s 平滑插值
- 复位确认：位置和速度满足阈值后连续稳定 0.5 s

## Ctrl+C 顺序

Ctrl+C 只向控制器提出退出请求。复位期间 HAL、AimRT backend 和控制线程保持运行。控制器确认复位完成并自然退出后，外层脚本才停止 HAL。

HAL 使用独立进程组运行，因此终端 Ctrl+C 不会提前把 HAL 一起关闭。

## 日志

- 控制器：`user_model3396.log`
- HAL：`hal_user_run.log`
