# 智元远征 A3 — 机器人完整访问参考文档

> **生成时间**: 2026-07-27  
> **本机环境**: Ubuntu 22.04 / Kernel 6.8.0-101 / x86_64  
> **本机用户**: `gzy` (uid=1000, sudo 组, 无免密 sudo)  
> **本机有线 MAC**: `c4:c6:e6:28:48:0a`  
> **本机 Wi-Fi MAC**: `dc:97:ba:7a:de:01`

---

## 当前有效链路与快速入口（先看这里）

当前已经固化并经过完全重启验证的链路只有这一套：

```text
PC enp12s0: 192.168.50.230/24
  → Ethernet
小黑盒 AP: SSID=pingpong, 4 MHz
  → HaLow
机器人 MDU taixin_mdu: 192.168.50.30/24
```

### 直接进入机器人 MDU（无需 ADB）

```bash
ssh agi@192.168.50.30
```

登录信息：

```text
用户名：agi
密码：1
```

进入 MDU 后进入 HDU：

```bash
ssh agi@10.42.10.10
```

### 连接检查

```bash
ip addr show enp12s0
ping -c 3 192.168.50.30
arping -I enp12s0 -c 5 192.168.50.30
nc -zvw3 192.168.50.30 22
nc -zvw3 192.168.50.30 1883
```

PC 连接小黑盒时，`enp12s0` 使用专网；普通互联网继续走 PC 的 Wi-Fi，不需要拔掉 Wi-Fi。当前专网没有默认网关，不要添加 `via 192.168.50.1`。

### 已废弃的进入方式

正常使用不再需要：

```text
ADB → SetTaixinToGCS
ADB 恢复脚本
旧的 10.42.2.50 地址
旧 SSID RAGA30P17C6100185
```

后文旧章节保留作历史审计记录；涉及旧地址或旧 SSID 时，以本文档第 16 章及本节为准。

---

## 一、机器人两台 RK3588 设备

| 节点 | 角色 | 系统 | 内核 | 架构 | 主机名 |
|------|------|------|------|------|--------|
| **HDU** | 头部单元/网关/对外出口 | Debian 12 (bookworm) | 6.1.118 PREEMPT_RT | aarch64 | `hdu` |
| **MDU** | 运动控制单元/从机 | Debian 12 (bookworm) | 6.1.118 PREEMPT_RT | aarch64 | `mdu` |

> 两台机器内核完全相同 (BuildID 41928, 2026-06-18)，且都有 `agi` 用户(uid=1001)。

---

## 二、HDU 访问方式

### 2.1 USB ADB（主要方式，推荐）

```bash
# 设备序列号
HDU_SERIAL=aba5a5b7b4c92aca

# 查看设备
adb devices
# 看到: aba5a5b7b4c92aca   device

# 直接进 shell (HDU adb 跑的就是 root, 不需授权)
adb -s aba5a5b7b4c92aca shell
# 在 shell 里: whoami → root
```

**HDU 没有 SSH 守护进程对外监听**（`22` 端口外部不通，仅 adb 通道可用）。

### 2.2 ADB 端口转发（已建好，下面是所有当前生效的 forward）

```bash
# 查看
adb -s aba5a5b7b4c92aca forward --list
# 输出:
#   aba5a5b7b4c92aca tcp:11883 tcp:18883   # MQTT: PC → MDU:1883
#   aba5a5b7b4c92aca tcp:10022 tcp:11022   # SSH:  PC → MDU:22
#   aba5a5b7b4c92aca tcp:15555 tcp:11555   # adb:  PC → MDU:5555 (备用, 当前 offline)

# 这些 forward 走的是:
#   PC:port → adb tunnel → HDU 内部 → HDU 上的 python relay → MDU
# 依赖 HDU 上 3 个 python relay 进程:
#   /tmp/r1.py:  HDU 18883 → 10.42.10.12:1883  (mosquitto)
#   /tmp/r2.py:  HDU 11022 → 10.42.10.12:22    (sshd)
#   /tmp/r3.py:  HDU 11555 → 10.42.10.12:5555  (adbd)
```

### 2.3 HDU 主要网络接口

| 接口 | 类型 | IP/掩码 | MAC | 用途 |
|------|------|--------|-----|------|
| `eth_hdu` | 千兆有线 (rk_gmac-dwmac) | `10.42.10.10/24` | `92:27:c0:08:c3:bd` | **机器人内部网** (与 MDU 直连) |
| `wifi_hdu` | RTL8852BE PCIe (5GHz STA) | `172.23.21.172/23` | `78:be:81:ea:f5:f8` | 主互联网出口 (连 SSID `Agibot-Robot`) |
| `rmnet_mhi0.1` | Quectel RM520N-GL 5G | `10.33.69.135/28` | `02:50:f4:00:00:01` | 蜂窝备份出口 (PCIe MHI) |
| `can0` | MCP251xFD (SPI) | — | — | CAN 总线 (DOWN) |
| `p2p_hdu` | 同 WiFi 芯片 P2P | — | `7a:be:81:ea:f5:f8` | P2P 模式 (DOWN) |

**HDU 路由**:
- 默认 `via 172.23.20.1 dev wifi_hdu` (metric 600, 主)
- 备份 `via 10.33.69.136 dev rmnet_mhi0.1` (metric 1000)
- 直连 `10.42.10.0/24 dev eth_hdu`

**HDU ip_forward=1, NAT 规则**: 见附录 A

### 2.4 HDU 业务进程（9 个 ROS2 agent）

```
agent (6571), data_exporter (4249), hal_audio (4326), hal_led (4640),
health_monitor (4516), hdu_camera (4581), ota_service (4532),
process_manager (4135), recordbag (4269), resource_manager (4416),
setting (4312)
```

全部连 MDU 1883 (10.42.10.12:1883) 跑 MQTT 业务。**无任何进程直接使用 taixin_mdu**。

### 2.5 HDU 关键服务端口

- `tcp/7000-7010` (本机): 各业务模块 HTTP RPC
- `udp/65400`: agent 多播
- `udp/4500` + `udp/500`: strongSwan IPsec
- `udp/1701`: xl2tpd L2TP

---

## 三、MDU 访问方式

### 3.1 首选：SSH 跳板 (经 HDU ADB forward)

```bash
# PC 上直接 SSH 到 MDU
sshpass -p '1' ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o ConnectTimeout=5 agi@127.0.0.1 -p 10022
# 登入后: hostname=mdu, id=agi(uid=1001)
```

**用户名/密码**: `agi` / `1`（同一凭据也用于 ADB、agibot software）
**agi 用户权限**: 在 `sudo` 组 / `root` 组 / `dialout` / `i2c` / `tcpdump` 等，权限很高  
**密码 sudo 提示**: `sudo` 需要 tty，直接通过 sshpass 会被卡住；通过 pty (`ssh -tt`) 或 `echo 1 | sudo -S` 喂密码

### 3.2 备用：直接 ADB

```bash
# 走 HDU 转发桥 (r3.py) → MDU adb
adb connect 127.0.0.1:15555
# 注: 当前 15555 forward 状态 offline, 可能需要重启 r3.py
```

### 3.3 MDU 主要网络接口

| 接口 | 类型 | IP/掩码 | MAC | 用途 |
|------|------|--------|-----|------|
| `eth_mdu` | 千兆有线 (rk_gmac-dwmac) | `10.42.10.12/24` | `6a:df:f0:e7:c4:c2` | **机器人内部网** (与 HDU 直连) |
| `taixin_mdu` | **HGIC 虚拟网口** (SDIO `hgicf` 驱动) | `10.42.2.50/24` | `ea:5c:43:82:7b:c0` | **Wi-Fi HaLow 802.11ah 客户端, 对端=小黑盒** |
| `ecat` | Intel I210 (igb) | — | `00:1b:21:ff:ff:ff` | EtherCAT 主站 (100Mb/s, 100% Full) |
| `ecat2` | Intel I210 (igb) | — | `00:1b:21:ff:ff:ff` | EtherCAT 主站 (同上) |

**MDU 路由**:
- 默认 `via 10.42.10.10 dev eth_mdu` (metric 102, 经 HDU 出网)
- 直连 `10.42.10.0/24 dev eth_mdu`
- 直连 `10.42.2.0/24 dev taixin_mdu`

**MDU 关键服务端口**:
- `tcp/22` sshd
- `tcp/1883` mosquitto (无认证, 任何 client 可连)
- `tcp/22524` (cos) / `tcp/5345` (lttng) / `tcp/2049` (NFS)
- `tcp/1701` xl2tpd / `tcp/500` + `tcp/4500` strongSwan
- `tcp/5555` adbd (备用)
- `udp/7777` `python3 -m elink.core` (HAL Elink, 内部)
- `udp/1701`/`udp/500`/`udp/4500`

### 3.4 MDU 业务进程

- `elink.core` (PID 1833): Python, `cwd=/opt/elink-tool/tool`, UDP:7777 — HAL 业务核心
- `mosquitto` (PID 1117): MQTT broker on tcp/1883
- `colink` (PID 879), `cos` (PID 882), `lttng-sessiond` (PID 894)
- RPC: `SetRgbLightCommand`, `StartFcServer`, `StartElinkHandServer` (skillpilot 域)

---

## 四、小黑盒 (智元泰芯 HaLow CPE)

### 4.1 真实身份

**泰芯 Wi-Fi HaLow (802.11ah) 终端**, 做以太网↔802.11ah 转换

### 4.2 已知 MAC

| 位置 | MAC | IP |
|------|-----|-----|
| **以太网侧** (PC 网线接的口) | `ea:5c:43:5f:7f:b8` | DHCP client (从 PC 拿 IP, 曾经拿 `192.168.77.65`) |
| **HaLow 侧** (与机器人关联) | `ea:5c:43:7f:7a:c0` | `10.42.2.10` |
| **机器人侧** (MDU 端 taixin_mdu) | `ea:5c:43:82:7b:c0` | `10.42.2.50` |

> 三个 MAC 同 OUI `ea:5c:43`, 同为泰芯/HGIC 设备家族

### 4.3 关键行为 (审计结论)

| 行为 | 实测结果 |
|------|----------|
| **DHCP Client** | ✅ 反复发 DHCP Request, 配 PC 共享模式后拿 IP |
| **二层桥接到 HaLow** | ❌ taixin_mdu 90 秒抓包零包 (含 ARP/广播/DHCP) |
| **三层路由转发 IP** | ❌ 流量到小黑盒 MAC 后被静默丢弃 |
| **管理面 (22/80/443)** | ❌ 全部 `Connection refused` (nmap -sV) |
| **假 DNS 劫持** | ⚠️ 53/UDP 返回 `198.18.0.0/30` 段地址 (与 PC Meta 隧道配合) |
| **MQTT 业务** | ✅ 短连接 GCS 客户端, 连 MDU:1883, 拉数据/发指令后断开 |

**结论**: **小黑盒不是给 PC 用来当路由/桥接的设备** — 它是配对机器人 HaLow 链路的伴生设备, 业务模型是"短连接 GCS 客户端"。

### 4.4 HaLow 链路参数 (机器人侧)

```
频道: 9060, 9100, 9140, 9180, 9220, 9260 (中国 Sub-1GHz)
带宽: 4 MHz
模式: sta (机器人 = 客户端, 小黑盒 = AP)
认证: WPA-PSK
SSID: <机器人 SN> (来自 /agibot/data/info/sn)
PSK: sha256("AAAA_" + SN + "_BBBB")
固件: 2.4.1.5 (svn 41928)
```

**链路质量** (最后一次读取): RSSI −42dBm, EVM −15, TX_SNR 38dB, RX_SNR 43dB

### 4.5 iptables NAT 规则 (MDU)

```bash
# 放行小黒盒 HaLow 侧 (10.42.2.10) 访问公网 (走 eth_mdu → HDU 出网)
iptables -t nat -A POSTROUTING -s 10.42.2.10/32 -o eth_mdu -j MASQUERADE
iptables -A FORWARD -i taixin_mdu -o eth_mdu -s 10.42.2.10/32 -j ACCEPT
# 64515 端口转发
iptables -t nat -A PREROUTING -i taixin_mdu -p tcp --dport 64515 \
  -j DNAT --to-destination 10.42.10.10:64515
```

---

## 五、PC 端访问机器人的所有可用入口

### 5.1 推荐方式 (已建好, 直接用)

```bash
# === MQTT 业务: 任何 client 都可以连 127.0.0.1:11883 ===
mosquitto_sub -h 127.0.0.1 -p 11883 -t '#' -v
mosquitto_pub -h 127.0.0.1 -p 11883 -t 'your/topic' -m 'hello'

# === SSH 到机器人 MDU ===
sshpass -p '1' ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  agi@127.0.0.1 -p 10022

# 登入后可以:
ssh agi@127.0.0.1 -p 10022 'hostname; ip -br addr show taixin_mdu; \
  mosquitto_sub -h 127.0.0.1 -p 1883 -t # -C 1 -W 5'
```

### 5.2 不要做的事

- ❌ **不要给 PC 有线口配 10.42.2.x/24** — 小黑盒不做三层转发, 配了也连不到 10.42.2.50
- ❌ **不要给小黒盒配静态 IP** — 它是 DHCP client
- ❌ **不要尝试走 小黑盒→HaLow→机器人 路径** — 路径不存在
- ❌ **不要改机器人 SN/HGIC 配置/SSID/PSK** — 会断 HaLow 配对

### 5.3 如果要走完整业务 (需触发小黑盒重连)

小黑盒是**短连接**: 连上 1883 → 拉数据/收指令 → 断开。空闲期 taixin_mdu 链路零包。

要触发新业务, 可:
- 启动 PC 上的 skillpilot / 智元 GCS 客户端 (如果有)
- 或在 PC 端用 mosquitto_pub 发指令到对应 topic, 等小黑盒 subscribe 时接收
- 或在 MDU 上跑 `mosquitto_pub -h 127.0.0.1 -t 'cmd/skillpilot/...' -m '...'`

---

## 六、PC 上已建的临时配置 (需要时回滚)

### 6.1 NetworkManager 连接

```bash
# 查看
nmcli connection show
# 新建的: A3-HaLow-BlackBox (DOWN, 没删除, 可重新 up)
# 原有的: 小黑盒 (10.42.10.50/24, 实际没用, 仍存在)
# 动捕 (192.168.50.230/24, 必须保留)
# Agibot-guest (WiFi, 必须保留)
# Meta (tun, 必须保留)
```

### 6.2 adb forward 列表

```bash
adb -s aba5a5b7b4c92aca forward --list
# 当前:
#   tcp:11883 → tcp:18883   (MQTT)
#   tcp:10022 → tcp:11022   (SSH)
#   tcp:15555 → tcp:11555   (ADB)
```

### 6.3 HDU 上后台进程 (依赖项)

```bash
# 必须存在的 3 个 python relay
# PID 在 /tmp/r{1,2,3}.log 里有
# 它们在 HDU 上, 由 r1.py / r2.py / r3.py 实现
# 崩溃了需要 adb push 重新跑
```

### 6.4 完全回滚命令

```bash
# PC 端
adb -s aba5a5b7b4c92aca forward --remove-all
adb -s aba5a5b7b4c92aca shell "pkill -9 -f 'python3 -u /tmp/r[123].py'"
sudo nmcli connection down "A3-HaLow-BlackBox"
sudo nmcli connection delete "A3-HaLow-BlackBox"   # 可选
sudo nmcli connection up "小黑盒"                     # 恢复原 profile
# 动捕配置 / WiFi / Meta 不动
```

---

## 七、敏感/动态信息位置

| 项目 | 位置 (机器人) | 备注 |
|------|---------------|------|
| 机器人 SN | `/agibot/data/info/sn` | 决定 HaLow SSID 和 PSK |
| 启动脚本 | `/etc/rc.local` | 设置 HGIC SSID/PSK + NAT 规则 |
| HGIC 配置 | `/etc/hgicf.conf` | mode=sta, key_mgmt=WPA-PSK, bss_bw=4, 6 个 channel |
| HGIC 工具 | `/usr/bin/hgicf` (我曾误称 hgpriv) | 实际命令是 `hgicf`, 例如 `hgicf taixin_mdu get ssid` |
| ELink 文档 | `/opt/elink-tool/tool/README.md` | Abox 调试工具说明 |
| hal_elink 日志 | `/agibot/data/log/hal_elink/hal_elink_a3.log` | 业务遥测记录 |
| DDS 日志 | `/agibot/data/log/dds/aimrt_hal_elink_a3_node_*.log` | |
| mosquitto 日志 | `/agibot/log/mosquitto/mosquitto.log` | 启动历史 |
| mosquitto 配置 | `/etc/mosquitto/mosquitto.conf` | listener 1883, allow_anonymous true |
| ROS2 环境变量 | 各 agent 启动脚本 | MQTT_BROKER_IP=10.42.10.10 等 |

---

## 八、完整命令速查

```bash
# === 一键登入 MDU (走 PC:10022) ===
sshpass -p '1' ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  agi@127.0.0.1 -p 10022

# === 一键进 HDU shell ===
adb -s aba5a5b7b4c92aca shell

# === 看 HDU 业务 ===
adb -s aba5a5b7b4c92aca shell "ps -eo pid,cmd | grep -E 'start_(agent|hal|setting|ota)'"

# === 看 MDU mosquitto 客户端 ===
sshpass -p '1' ssh agi@127.0.0.1 -p 10022 'ss -ntp 2>/dev/null | grep 1883'

# === 看 HaLow 链路状态 (需 sudo) ===
sshpass -p '1' ssh -tt agi@127.0.0.1 -p 10022 'echo 1 | sudo -S cat /proc/hgicf/status'

# === MDU 端订阅一个 MQTT topic 看 ===
sshpass -p '1' ssh agi@127.0.0.1 -p 10022 \
  'python3 -c "import socket,struct,time
s=socket.socket();s.settimeout(2);s.connect((\"127.0.0.1\",1883))
s.send(b\"\\x10\\x10\\x00\\x04MQTT\\x04\\xc2\\x00\\x3c\\x00\\x04test\")
print(s.recv(4).hex())
s.send(b\"\\x82\\x08\\x00\\x01#\\x00\\x02\\x00\\x01\"); s.recv(3)
end=time.time()+15
while time.time()<end:
  try:
    h=s.recv(1); rl=0; sh=1
    while True:
      b=s.recv(1)[0]; rl+=(b&0x7f)*sh
      if not(b&0x80): break
      sh*=128
    body=b\"\"; 
    while len(body)<rl: body+=s.recv(rl-len(body))
    if (h[0]>>4)==3:
      tl=struct.unpack(\">H\",body[:2])[0]; tp=body[2:2+tl].decode(errors=\"replace\")
      pl=body[2+tl:]; print(f\"{tp} L={len(pl)} {pl[:200]!r}\")
  except: pass"'
```

---

## 九、附录 A: HDU 完整 NAT 规则 (iptables-save)

```
*nat
:PREROUTING ACCEPT
:INPUT ACCEPT
:OUTPUT ACCEPT
:POSTROUTING ACCEPT
-A PREROUTING -p tcp --dport 59201 -j REDIRECT --to-ports 59301
-A PREROUTING -p tcp --dport 56322 -j DNAT --to-destination 10.42.10.12:56322
-A PREROUTING -p tcp --dport 56422 -j DNAT --to-destination 10.42.10.12:56422
-A PREROUTING -p tcp --dport 51011 -j DNAT --to-destination 10.42.10.12:51011
-A PREROUTING -p tcp --dport 50587 -j DNAT --to-destination 10.42.10.12:50587
-A PREROUTING -p tcp --dport 57900 -j DNAT --to-destination 10.42.10.12:57900
-A PREROUTING -i wifi_hdu -p tcp --dport 1883 -j DNAT --to-destination 10.42.10.12:1883
-A PREROUTING -i wifi_hdu -p tcp --dport 8883 -j DNAT --to-destination 10.42.10.12:8883
-A PREROUTING -p tcp --dport 51056 -j DNAT --to-destination 10.42.10.12:51056
-A POSTROUTING -s 10.42.10.12 -o wifi_hdu -j MASQUERADE
-A POSTROUTING -d 10.42.10.12 -p tcp --dport {56322,56422,51011,50587,57900,1883,8883,51056} -j MASQUERADE
-A POSTROUTING -s 10.42.10.12 -o rmnet_mhi0.1 -j MASQUERADE
COMMIT
```

**含义**: HDU 把外部 (WiFi/5G) 来的 56322/56422/51011/50587/57900/51056/1883/8883 端口都 DNAT 到 MDU, MDU 出网时 MASQUERADE。**完全没有任何 taixin_mdu/10.42.2.x 相关规则**。

## 十、附录 B: 关键判断回顾

| 假设 | 实际 | 证据 |
|------|------|------|
| 小黑盒是透明桥 | ✗ | taixin_mdu 90秒专项抓包零包 |
| 小黑盒是路由器 | ✗ | 静默丢弃所有 IP 包 |
| PC 配 10.42.2.x 能通 | ✗ | 小黑盒不转发 |
| HDU/MDU 在 192.168.x | ✗ | 实际 10.42.10.0/24 |
| HDU 有 SSH | ✗ | 只有 ADB |
| MDU 有 SSH | ✓ | tcp/22, 需经 HDU forward |
| 业务用 ROS topic | 部分 | HAL 用 MQTT protobuf, ROS2 内部用 DDS |
| 10.42.2.10 持续连接 | ✗ | 短连接 FIN-WAIT-1 |
| HGIC 模组类型 | SDIO | dmesg: vendor:a012 id:4002 |
| HaLow 频段 | 906-926 MHz | /etc/hgicf.conf |

## 十一、版本信息

- **完整版本**: `v3.1.19-0-8b01fd2a8-gf84a3dda`
- **BSP 版本**: `v0.7.3`
- **目标**: MDU
- **Git 分支**: `release/a3/v3.1`
- **构建时间**: 2026-06-18 20:52:15
- **Pipeline**: https://code.agibot.com/agibot_bsp/a3_build_system/-/pipelines/407523

---

## 十二、2026-07-27 当前最终实验状态补充

### 12.1 一句话结论

截至 2026-07-27，本轮尚未打通 A3 小黑盒 GCS 专网。机器人端曾被 `SetTaixinToGCS` 切换到 `192.168.50.30/24`，但小黑盒没有提供预期的 `192.168.50.1`，所以没有形成 PC → 小黑盒 → HaLow → 机器人闭环。

当前现场已恢复并保持为：

```text
taixin_app_mode = common
taixin_mdu      = 10.42.2.50/24
PC enp12s0      = 10.42.10.50/24，连接“小黑盒” profile
ADB             = 在线
setting1        = PID 4297，未停止/未重启
agent           = PID 4979，未停止/未重启
```

### 12.2 真实拓扑与路径边界

```text
ADB 路径（独立恢复通道）
PC USB → HDU → SSH 10.42.10.12 → taixin_mdu

目标小黑盒路径（本轮未闭环）
PC enp12s0 → 小黑盒 Ethernet
                  ↓ 当前未发现 GCS/AP 地址或转发行为
             小黑盒 HaLow → taixin_mdu
```

设备标识：

| 设备 | 地址 | 状态 |
|---|---|---|
| 小黑盒 Ethernet 侧 | MAC `ea:5c:43:5f:7f:b8` | 历史 Common 测试可 ARP |
| 小黑盒 HaLow 侧 | MAC `ea:5c:43:7f:7a:c0`，历史 IP `10.42.2.10` | 历史已关联 |
| 机器人 `taixin_mdu` | MAC `ea:5c:43:82:7b:c0`，Common IP `10.42.2.50/24` | 当前确认 |
| PC 有线口 | `enp12s0`，MAC `c4:c6:e6:28:48:0a` | 当前确认 |

ADB、USB 转发或 HDU—MDU 内网成功，均不能作为小黑盒专网成功证据。

### 12.3 机器人端 GCS 切换实验

曾使用已构造的 AimRT/MQTT shadow 执行过一次 `SetTaixinToGCS`，目标字段为：

```text
IP          = 192.168.50.30
subnet_mask = 255.255.255.0
gateway     = 192.168.50.1
SSID        = pingpong
bss_bw      = 4
chan_list  = 9060,9100,9140,9180,9220,9260
key_mgmt   = WPA-PSK
```

PSK 未写入本文档，也未在日志中输出明文。

当时观测到：

```text
SetTaixinToGCS 返回 AimRT status code 0
taixin_mdu 变为 192.168.50.30/24
默认路由变为 via 192.168.50.1
```

MDU 抓包持续出现：

```text
Who has 192.168.50.1?
```

但没有任何 ARP 回复。PC 临时使用 `192.168.50.230/24` 后，对 `192.168.50.1` 和 `192.168.50.30` 的 ARP 也没有回复。随后已调用 Common 回滚，机器人恢复到 `10.42.2.50/24`。

**重要限定：**这次成功返回来自 MDU shadow，不是官方 HDU `setting1` 的完整调用链；它只证明机器人侧代码可以改接口和路由，不证明小黑盒已经同步切换。

### 12.4 小黑盒 Ethernet 侧复核

历史上启用 `A3-HaLow-BlackBox` shared profile 后，小黑盒曾响应：

```text
小黑盒历史地址 = 192.168.77.65
ARP MAC        = ea:5c:43:5f:7f:b8
ping           = 成功
```

同一专项测试证明 Common 模式下小黑盒不是通用三层路由器或透明桥：PC 发往 `10.42.2.50` 的包可以到达小黑盒 Ethernet MAC，但 MDU `taixin_mdu` 抓包没有对应转发流量。

本轮重新连接网线后，再启用 `A3-HaLow-BlackBox` 观察 15 秒：

```text
物理 carrier = 1
速率         = 100 Mb/s Full Duplex
捕获到       = 仅 PC 自己的 ARP
捕获不到     = 小黑盒 MAC、DHCP、ARP reply、GCS 地址通告
```

因此可以确认：网线物理层正常，但当前小黑盒没有在该 Ethernet 侧提供可发现的业务地址或 DHCP/ARP 响应。旧租约文件中的 `192.168.77.65` 只能作为历史证据，不能当作当前地址。

### 12.5 官方 RPC 调用链实际含义

协议中存在：

```text
RobotX86SettingService/GetTaixinInfo
RobotX86SettingService/SetTaixinToGCS
RobotX86SettingService/SetTaixinToCommon
```

但当前官方 `setting1` 日志实际显示的是：

```text
RegisterClientFunc
Client 'pb:/aimdk.protocol.RobotX86SettingService/GetTaixinInfo'
Init client for service ... succeeded
```

不是服务端注册。ROS2 查询结果为：

```text
Type: ros2_plugin_proto/srv/RosRpcWrapper
Clients count: 1
Services count: 0
```

因此从 PC 调用该 ROS2 名称不会进入当前 `setting1` 的处理函数，也不会在 `setting1.log` 产生 `recv GetTaixinInfo req` 或 `recv SetTaixinToGCS req`。

当前 `setting_config.yaml.dump` 还显示：

```yaml
x86_setting_srv: {}
x86_setting_srv_proxy: {}
soc_index_: SOC1
model_index_: A2ULTRA
```

当前官方 `setting1` 实际 Serve 的主要是 `RobotSettingService`、`RobotOrinSettingService` 和 `RobotFunctionSettingService`；`RobotX86SettingService` 在此进程中表现为代理/客户端资源。

### 12.6 官方 ROS2 环境复现

系统 ROS2 CLI 直接调用会遇到 Fast-CDR ABI 不匹配：

```text
undefined symbol: eprosima::fastcdr::Cdr::serialize(unsigned int)
```

机器人官方启动环境为：

```text
ROS_DOMAIN_ID=232
RMW_LIBRARY_PATH=/agibot/software/v0/bin/librmw_fastrtps.so
FASTRTPS_DEFAULT_PROFILES_FILE=/opt/agibot/entry/cfg/privileged_ros_dds_configuration.xml
MQTT_BROKER_IP=10.42.10.12
```

通过机器人自带 ROS2 库编译的 native `RosRpcWrapper` 只读探针可以正常调用已实际 Serve 的接口。例如：

```text
/aimdk_2Eprotocol_2ERobotSettingService/GetSystemWifiHotSpotState
RPC code = 0
响应内容 = Failed to read Wi-Fi hotspot name
```

这说明 native 调用方法本身可用；`RobotX86SettingService` 的问题是没有发现服务端，而不是简单的 CLI 参数错误。

本机保留只读探针源文件：

```text
/home/gzy/tools/taixin_ros2_probe.cpp
```

### 12.7 机器人热点与 GCS 侧证据

官方 `setting` 二进制中包含以下逻辑字符串：

```text
set wifi hotspot ssid
set wifi hotspot password
/opt/ap/ssid
/opt/ap/passwd
Failed to start Wi-Fi hotspot
Failed to read Wi-Fi hotspot name
```

但当前机器人镜像中：

```text
/opt/ap           = 不存在
/opt/ap/ssid      = 不存在
/opt/ap/passwd    = 不存在
```

官方只读热点状态 RPC 也返回读取热点名称失败。因此当前证据支持：机器人镜像包含热点控制代码，但实际热点运行状态、配置文件或对应的 x86/GCS 服务组件缺失或未启用。

这与小黑盒 Ethernet 侧没有 `192.168.50.1`、没有 DHCP/ARP 响应的现象相互吻合。

### 12.8 结论分级

#### 已证实

- Common HaLow 机器人链路当前可恢复并保持正常。
- 机器人侧 `SetTaixinToGCS` 可以把 `taixin_mdu` 改成 `192.168.50.30/24` 并设置 `192.168.50.1` 路由。
- GCS 切换后 MDU 会主动 ARP 查询 `192.168.50.1`。
- 当前小黑盒 Ethernet 物理链路正常，但本轮没有 DHCP、ARP 或业务帧响应。
- 当前官方 `setting1` 对 `RobotX86SettingService` 是客户端，不是可调用的服务端。
- 当前官方 ROS2 图中该服务 `Services count=0`。
- 当前机器人热点状态查询失败，`/opt/ap` 配置目录不存在。

#### 高概率推断

- 当前 OTA/运行配置没有启用 A3 所需的 `x86_setting_srv` 服务端。
- 小黑盒没有进入 GCS/AP 模式，或其 GCS 地址/热点配置未加载。
- `pingpong` 对应 PSK 可能没有被正确配置到小黑盒侧；不能仅凭机器人保存的 Common PSK 推断。

#### 尚未确认

- 小黑盒的实际固件版本和当前工作模式。
- 小黑盒是否能通过厂商工具切换为 GCS/AP。
- `pingpong` 的真实 PSK。
- `192.168.50.1` 是小黑盒固定地址，还是由官方 PC/GCS 客户端启动时提供。
- 当前硬件是否需要另一个 SOC0/x86 专用进程或独立 OTA 组件。

#### 已排除

- 不能再把问题归因于网线 carrier 或 PC 没有设置 `192.168.50.230`。
- 不能把旧 DHCP 租约 `192.168.77.65` 当作当前小黑盒在线地址。
- 不能把 MDU shadow 的一次返回成功当作官方完整专网闭环成功。
- 不能把 ADB/HDU—MDU 访问当作小黑盒 HaLow 专网成功。

### 12.9 当前未改变及已回滚内容

- 未修改 `/etc/hgicf.conf`。
- 未修改机器人默认路由、EtherCAT、电机或运动控制配置。
- 未重启 HDU、MDU、机器人、`setting1`、`agent` 或 `mosquitto`。
- 未删除或覆盖“动捕”连接。
- PC 临时连接 `A3`（`192.168.50.230/24`）和 `A3-HaLow-BlackBox`（`192.168.77.1/24`）均未保持激活。
- PC 当前已恢复到原“小黑盒”连接，`enp12s0=10.42.10.50/24`。
- 本轮临时 MQTT relay 和 `adb forward tcp:11883` 已清理；已有的 `10022/11022/15555/11555` 转发属于现场原有通道，未删除。

### 12.10 下一步真正需要的资料

不能再靠重复切换机器人端解决。要完成闭环，必须先取得或确认以下任一项：

1. 智元 A3 当前 OTA 对应的 `x86_setting_srv` 服务端启动配置和进程；
2. 小黑盒 GCS/AP 模式的厂商配置工具、固件版本和切换命令；
3. `pingpong` 专网真实 PSK 及小黑盒侧 SSID/IP 配置；
4. AimMaster/AimDK 使用的实际 GCS 客户端和调用通道；
5. 若小黑盒由 PC 客户端提供 GCS 功能，则该客户端的启动程序、配置文件和监听端口。

取得组件后，正确验收顺序应为：

```text
确认小黑盒进入 GCS/AP
→ 看到小黑盒 Ethernet/HaLow 侧 MAC 与 192.168.50.1
→ PC 临时启用 A3：192.168.50.230/24
→ 只读调用官方 GetTaixinInfo
→ 单次 SetTaixinToGCS
→ MDU 出现 192.168.50.30/24 且关联 192.168.50.1
→ PC 与 MDU 双端抓包证明流量经过小黑盒
→ 访问一个真实只读服务
```

在小黑盒侧出现 `192.168.50.1` 并能对 ARP 响应之前，不应再次执行 GCS 切换。

---

+## 十三、2026-07-27 最终 NetAT/GCS 受控实验（最新状态）

> 本章是本文档的最新权威状态。第 12 章记录的是早期 Common/未切换阶段；其中“当前已恢复 Common”等描述不覆盖本章最终实验结果。

### 13.1 一句话结论

机器人侧 `SetTaixinToGCS` 已成功，小黑盒无线侧也已写入新的 `pingpong` 参数并能报告 RSSI；但小黑盒没有提供 `192.168.50.1` 的 Ethernet/LAN 地址，也没有把 PC Ethernet 流量转发到 HaLow。因此本次没有形成 PC → 小黑盒 → 机器人真实服务的专网闭环。

### 13.2 小黑盒最终实测配置

~~~text
管理地址：192.168.77.65
MAC：ea:5c:43:5f:7f:b8
管理协议：UDP NetAT/56789
模式查询结果：ap
SSID：pingpong
加密：WPA-PSK
BSS_BW：4 MHz
CHAN_LIST：9060,9100,9140,9180,9220,9260
RSSI：-38 dBm
~~~

曾尝试写入 `wifimode=wnbap`，命令返回成功，但随后查询仍返回 `ap`。因此不能把写入返回成功解释成固件已经运行在 `wnbap`；最终可证实的模式仍是 `ap`。

本次生成的 PSK 只在同一执行进程内分别写入小黑盒和机器人 RPC 请求，没有输出到日志或本文档。后续不得从日志猜测该 PSK。

### 13.3 公开 GitHub 资料固化

公开工具仓库：

~~~text
https://github.com/aliosa27/taixin_tools
~~~

确认的 NetAT 特征：

~~~text
UDP 端口：56789
发现方式：广播扫描
单播方式：按小黑盒 MAC 发送 AT 请求
查询：WIFIMODE、SSID、ENCRYPT、BSS_BW、CHAN_LIST、RSSI、VERSION、STA_INFO
配置：WIFIMODE、SSID、ENCRYPT、KEY、BSS_BW、CHAN_LIST
~~~

参考：

~~~text
https://github.com/aliosa27/taixin_tools/blob/main/docs/netat_protocol.md
https://github.com/aliosa27/taixin_tools/blob/main/docs/at_commands_2x_firmware.md
~~~

关键边界：公开 NetAT 工具能配置 HaLow 无线模块，但没有发现配置小黑盒 Linux Ethernet/LAN 地址、网关或路由/NAT 的命令。工具的 `saveconfig` 也只是保存无线 AT 参数，不等于配置 LAN。

### 13.4 NetAT 管理命令

PC 临时管理网络：

~~~bash
sudo ip addr add 192.168.77.1/24 dev enp12s0
~~~

发现设备：

~~~bash
sudo python3 libnetat.py enp12s0 --command scan
~~~

只读查询：

~~~bash
sudo python3 libnetat.py enp12s0 \\
  --dest_mac ea:5c:43:5f:7f:b8 \\
  --command 'at+wifimode?'
~~~

写入参数的顺序应为：信道/带宽 → 加密/密钥 → SSID → 工作模式。示例：

~~~bash
sudo python3 libnetat.py enp12s0 --dest_mac ea:5c:43:5f:7f:b8 --command 'at+chan_list=9060,9100,9140,9180,9220,9260'
sudo python3 libnetat.py enp12s0 --dest_mac ea:5c:43:5f:7f:b8 --command 'at+bss_bw=4'
sudo python3 libnetat.py enp12s0 --dest_mac ea:5c:43:5f:7f:b8 --command 'at+encrypt=1'
sudo python3 libnetat.py enp12s0 --dest_mac ea:5c:43:5f:7f:b8 --command 'at+ssid=pingpong'
sudo python3 libnetat.py enp12s0 --dest_mac ea:5c:43:5f:7f:b8 --command 'at+wifimode=ap'
~~~

`at+key=...` 的原文禁止进入 shell 历史、普通日志或报告；应通过受控进程环境变量传递。

### 13.5 机器人官方 RPC 实测

通过临时 ADB → HDU → MDU MQTT relay，使用官方 AimRT protobuf probe 调用一次：

~~~text
RobotX86SettingService.SetTaixinToGCS
~~~

请求字段：

~~~text
IP：192.168.50.30
subnet_mask：255.255.255.0
gateway：192.168.50.1
SSID：pingpong
bss_bw：4
chan_list：9060,9100,9140,9180,9220,9260
key_mgmt：WPA-PSK
wpa_psk：本次生成值，未打印
~~~

RPC 返回：

~~~text
status=suc, code 0, msg: OK
~~~

MDU 实际状态：

~~~text
taixin_mdu：192.168.50.30/24
默认路由：via 192.168.50.1
~~~

ADB、`setting1`、`agent` 均保持在线，官方进程未重启；本次没有调用回滚。

### 13.6 双端抓包和访问结果

MDU `taixin_mdu` 抓包持续看到：

~~~text
192.168.50.30 → ARP who-has 192.168.50.1
~~~

没有任何 `192.168.50.1` 的 ARP reply。

PC 最终配置：

~~~text
NetworkManager：A3
接口：enp12s0
地址：192.168.50.230/24
默认网关：无
~~~

实测：

~~~text
192.168.50.1 ARP：失败
192.168.50.1 ping：失败
192.168.50.30 ping：失败
192.168.50.30:22：失败
192.168.50.30:1883：失败
~~~

系统未安装 `arping`，所以不能把“arping 命令不可用”误写成 ARP 成功或失败；但 PC ping、MDU 抓包和邻居表均证明 `.1` 没有响应。

另通过临时路由测试真实管理地址是否能转发 GCS 流量：

~~~bash
sudo ip route add 192.168.50.30/32 via 192.168.77.65 dev enp12s0
~~~

仍无 ping/TCP 响应，MDU 抓包也没有看到来自 `192.168.50.230` 的流量。验证后已删除临时路由和 `192.168.77.1` 地址。

### 13.7 当前拓扑判断

已证实：

~~~text
PC Ethernet
    │
    ▼
小黑盒 Ethernet 管理面：192.168.77.65
    │ NetAT 可管理
    ▼
小黑盒 HaLow AP：pingpong / 4 MHz / RSSI -38 dBm
    )))
机器人 taixin_mdu：192.168.50.30/24
~~~

未成立：

~~~text
小黑盒 Ethernet/LAN：192.168.50.1
PC Ethernet → 小黑盒 → HaLow → 192.168.50.30 的 IP 转发
~~~

准确分类：

~~~text
无线关联层：已基本成立
小黑盒 LAN/GCS 网关层：未成立
PC 到机器人业务层：未成立
~~~

### 13.8 不得混淆的路径和禁止事项

~~~text
ADB 路径：PC → USB ADB → HDU → SSH → MDU
HaLow 路径：PC Ethernet → 小黑盒 Ethernet → 小黑盒 HaLow → taixin_mdu
~~~

ADB、SSH、MQTT relay 或 AimRT RPC 成功都不能证明 HaLow 专网成功。

禁止用以下方式伪造闭环：

~~~text
在 PC 上冒充 192.168.50.1
在 MDU 上手工添加 192.168.50.30 或网关
在 PC 上做 NAT 后称为小黑盒转发
把 ADB forward 当作 HaLow 路径
重复 SetTaixinToGCS 试密码
~~~

### 13.9 真正完成闭环所缺的组件

当前必须取得或确认以下至少一项：

1. 厂商 GCS/AP 模式配置工具；
2. 小黑盒固件中设置 LAN IP 为 `192.168.50.1/24` 的方法；
3. 小黑盒桥接/路由/NAT 配置说明；
4. 智元配套 PC GCS 客户端或启动脚本；
5. 小黑盒专用 GCS 固件或配置文件；
6. 明确 `192.168.50.1` 是小黑盒地址，还是由 PC 客户端虚拟提供。

只有同时满足以下条件，才允许报告“专网闭环成功”：

~~~text
小黑盒对 who-has 192.168.50.1 回复
机器人不再持续 ARP FAILED
PC 192.168.50.230 的包出现在 MDU taixin_mdu 抓包
PC 能访问机器人至少一个真实只读服务
~~~

### 13.10 最终实验产物和状态

实验目录：

~~~text
/home/gzy/a3_taixin_gcs_final_20260727_193309/
~~~

当前最终状态：

~~~text
机器人：GCS，192.168.50.30/24
小黑盒：pingpong / AP / 4 MHz / 目标信道
PC：A3，192.168.50.230/24
ADB：在线，独立于 HaLow
专网闭环：未完成
~~~

最后一次清理已移除临时 MQTT relay、`adb forward tcp:11883`、临时 `192.168.77.1` 地址和测试路由；PC 的 `A3` 配置保留并处于激活状态，原有“动捕”连接未删除或覆盖。

---

## 十四、2026-07-28 小黑盒模式切换复核（最新增量）

### 14.1 本轮目标

在不切换机器人的前提下，尝试通过小黑盒 Ethernet 侧 UDP NetAT/56789 将无线模式从 `ap` 切换为 `wnbap`，并验证重启后的真实运行态。

### 14.2 只读基线

小黑盒管理路径仍为：

~~~text
PC enp12s0 -> 临时 192.168.77.1/24 -> UDP NetAT/56789
小黑盒 MAC：ea:5c:43:5f:7f:b8
小黑盒管理地址：192.168.77.65
~~

2026-07-28 查询结果：

~~~text
WIFIMODE = ap
VERSION  = v2.4.1.3-41928, app:0
SSID     = HALOW_5F7FB8
BSS_BW   = 8
CHAN_LIST = 9080,9160,9240
RSSI     = 0
~~

`RSSI=0` 表示当前没有已关联的 STA。机器人保持 Common，未执行新的 `SetTaixinToGCS`。

### 14.3 wnbap 切换实测

执行：

~~~bash
sudo python3 libnetat.py enp12s0 \\
  --dest_mac ea:5c:43:5f:7f:b8 \\
  --command 'at+wifimode=wnbap'
sudo python3 libnetat.py enp12s0 \\
  --dest_mac ea:5c:43:5f:7f:b8 \\
  --command 'at+wifimode?'
sudo python3 libnetat.py enp12s0 \\
  --dest_mac ea:5c:43:5f:7f:b8 \\
  --command 'at+rst'
~~~

实际结果：

~~~text
写入响应：mode=wnbap
写入后查询：WIFIMODE=ap
重启响应：OK
重启后查询：WIFIMODE=ap
重启后 SSID：HALOW_5F7FB8
重启后 BSS_BW：8
~~

曾额外尝试 `at+auto_save=1`，设备无响应；不能据此认为该参数已开启。重启后无线参数恢复默认，证明此前写入是临时运行态，未持久化。

### 14.4 结论分级

**已证实：**

- NetAT 通道可用，管理地址和 MAC 已验证；
- 当前固件真实运行模式是 `ap`；
- `at+wifimode=wnbap` 可以返回一个接受字符串，但不能改变当前运行态；
- 重启后模式和无线参数均恢复默认；
- 固件版本为 `v2.4.1.3-41928, app:0`；
- 公开 `taixin_tools` 文档明确标注 `wnbap/wnbsta` 为私有协议模式，默认固件不支持；
- `at+mode=wnbap` 不是该固件的有效入口。

**高概率推断：**

- 该盒子是标准协议 AP 固件，不具备可由公开 NetAT 开启的泰芯私有 `wnbap` 功能；
- “GCS 专网模式”不是当前盒子公开 AT 层的 `wnbap` 开关，可能需要厂商专用固件、授权参数或官方 GCS 配置工具；
- `192.168.50.1` 的缺失仍然是 Ethernet/LAN/桥接层问题，不会由 `wifimode` 命令自动创建。

**已排除：**

- 不是 PC 没有接网线：NetAT 单播查询、ARP 和 `192.168.77.65` ping 均正常；
- 不是命令拼写导致的唯一失败：公开命令写入有响应，但重启后运行态仍为 `ap`；
- 不是机器人当前切换造成的现象：本轮未调用机器人 GCS RPC，机器人保持 Common。

### 14.5 当前状态和下一步所需资料

本轮已清理临时 `192.168.77.1/24` 地址，未修改“动捕”连接，ADB 仍在线。小黑盒当前恢复为：

~~~text
WIFIMODE = ap
SSID = HALOW_5F7FB8
BSS_BW = 8
CHAN_LIST = 9080,9160,9240
~~

要真正切换到任务专用 GCS 形态，不能继续猜公开 AT 命令；需要向智元/泰芯索取以下任一项：

1. 支持 `wnbap` 或 GCS 专网的匹配固件/版本；
2. 小黑盒 LAN IP、桥接、路由/NAT 的官方配置方法；
3. 官方 GCS/PC 客户端及启动参数；
4. 与机器人 `SetTaixinToGCS` 对应的盒端配置工具或授权配置文件；
5. 明确 `192.168.50.1` 是小黑盒 LAN 地址，还是官方客户端在 PC 侧提供的网关地址。

当前不能把 `at+wifimode=wnbap` 的一次返回当成模式切换成功，也不能把机器人 ADB/RPC 通道当成小黑盒专网已经打通。

本轮证据目录：

~~~text
/home/gzy/a3_taixin_mode_switch_20260728_083445/
~~~

---

## 十五、2026-07-28 单次切换 + WNB 状态深度探测（本轮最新）

> 本章是本轮（`a3_taixin_bridge_final_20260728_085726`）的全部实验结论。第 12、13 章的旧 GCS 切换结论保留但不覆盖本章。本章严格遵循"密钥冻结、不重生成、不研究 wnbap、不重新泛化扫描"四条边界。

### 15.1 一句话结论

`SetTaixinToGCS` 在机器人侧接口、路由、HGIC 配置上**全部生效**（`taixin_mdu=192.168.50.30/24`、默认路由 `via 192.168.50.1`、HGIC FW_STATE 活跃），PC 端 `192.168.50.230` 也正确发出 ARP/ping；**但小黑盒 `v2.4.1.3-41928, app:0` 这一版固件不响应任何 WNB 调试/查询命令**（`AT+CONN_STATE`/`AT+WNBCFG`/`AT+STA_LIST` 等全部无响应），且 PC 的 Ethernet 帧**完全没进入 WNB 数据面**。专网未通。

### 15.2 实验现场基线（2026-07-28 09:16–09:18）

| 项 | 实测 |
|---|---|
| ADB | `aba5a5b7b4c92aca` 在线 |
| MQTT broker 经 `127.0.0.1:11883` | CONNACK=0 通过（HDU SSH 隧道 PID 60918 → MDU:1883） |
| `taixin_gcs_probe` | `GetTaixinInfo status=suc`，返回 `{"module_type":"258","fw_version":"fw info:2.4.1.5, svn version:41928, app:0","mac":"ea:5c:43:82:7b:c0"}` |
| 机器人初始 | `taixin_mdu=10.42.2.50/24`（Common） |
| 冻结 PSK | `7d78879ee30cdf2cf0beb34298dfd810`（32 字符，`.taixin_psk`） |
| PC 初始 | `enp12s0=192.168.50.230/24`（来自 `动捕` profile）+ `192.168.77.1/24`（管理面，来自临时 `A3-HaLow-BlackBox` shared profile） |
| 小黑盒初始 | `WIFIMODE=ap, SSID=HALOW_5F7FB8, BSS_BW=8, CHAN_LIST=9080,9160,9240`（默认） |
| 小黑盒切后 | `WIFIMODE=ap, SSID=pingpong, BSS_BW=4, CHAN_LIST=9060,9100,9140,9180,9220,9260, ENCRYPT=1` |
| `libaimrt_mqtt_plugin.so` 加载 | 需 `LD_LIBRARY_PATH=/home/gzy/a3_deploy_example/.local/a3_sim/libstdcxx:/home/gzy/robot_pingpang/third_party/aimsim_official/motion_control_humble/bin`（GLIBCXX_3.4.31） |
| aimrt_main 选择 | `/home/gzy/robot_pingpang/third_party/aimsim_official/motion_control_humble/bin/aimrt_main`（v1.6.0，能加载探针模块），不是 `mujoco_sim_standalone` 下的（v1.0.0 段错误） |

### 15.3 同步抓包

PC `enp12s0` 与 MDU `taixin_mdu` 同时开启 600s tcpdump（保存在 `/home/gzy/a3_taixin_bridge_final_20260728_085726/captures/`），分别落盘 `pc_enp12s0.pcap`（11.5 KB）与 `mdu_taixin_mdu.pcap`（20 KB）。

### 15.4 SetTaixinToGCS 单次切换

参数：

~~~text
IP           = 192.168.50.30
subnet_mask  = 255.255.255.0
gateway      = 192.168.50.1
SSID         = pingpong
bss_bw       = 4
chan_list    = 9060,9100,9140,9180,9220,9260
key_mgmt     = WPA-PSK
wpa_psk      = 7d78879ee30cdf2cf0beb34298dfd810 (从 .taixin_psk 读取)
~~~

调用方式：`TAIXIN_ACTION=set_gcs TAIXIN_WPA_PSK=<frozen> ./aimrt_main --cfg_file_path=probe_cfg.yaml`

返回：

~~~text
[SET_TAIXIN_TO_GCS] status=suc, code 0, msg: OK
[SET_TAIXIN_TO_GCS] response={"header":{"code":"0","msg":"","trace_id":"","domin":""},"state":"CommonState_UNKNOWN"}
~~~

机器人侧立即确认：

~~~text
taixin_mdu       UNKNOWN        192.168.50.30/24 fe80::6a3c:c1d2:4e35:f337/64
default via 192.168.50.1 dev taixin_mdu
default via 10.42.10.10 dev eth_mdu proto static metric 103  (HDU 备份)
192.168.50.0/24 dev taixin_mdu proto kernel scope link src 192.168.50.30
5: taixin_mdu: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500
    link/ether ea:5c:43:82:7b:c0
    RX: 0 packets, 0 bytes
    TX: 0 packets, 0 bytes
/proc/hgicf/status:
    RADIO:1, FW_STATE:0, ALIVE_TMR:1547, TX_CTRL:2492, TX_DATA:296
~~~

### 15.5 数据面验收

PC 侧：

~~~bash
sudo ip neigh flush dev enp12s0
sudo arping -I enp12s0 -c 5 192.168.50.30   →  0/5 reply, 100% unanswered
ping -I enp12s0 -c 5 192.168.50.30          →  Destination Host Unreachable
nc -zvw3 192.168.50.30 22                    →  No route to host
nc -zvw3 192.168.50.30 1883                 →  timed out
~~~

抓包判据：

| 抓包位置 | 关键统计 | 结果 |
|---|---|---|
| PC `enp12s0` | `who-has 192.168.50.30 tell 192.168.50.230` | 16 次发出，0 次 reply |
| PC `enp12s0` | 小黑盒 MAC `ea:5c:43:5f:7f:b8`（排除 DHCP） | 0 帧 |
| MDU `taixin_mdu` | 来自 PC `192.168.50.230` 或小黑盒 `192.168.50.1` | **0 帧** |
| MDU `taixin_mdu` | 机器人自发的 `who-has 192.168.50.1` | 318 次，0 次 reply |
| PC `enp12s0` | 小黑盒 DHCP Request（每 60 s） | 持续存在，仅此一帧类型 |

判断按用户给的决策表 → **MDU 完全看不到 PC 帧**，落在"小黑盒关联、收发计数和 forward 数据面"那一支，而不是"MDU 能看到 PC 帧但机器人不回复"那一支。

### 15.6 WNB 状态深度探测（用户追加要求）

`AT+SYSDBG=WNB,1` 接受后，按用户给的命令清单逐项探测：

| 命令 | 实测 |
|---|---|
| `AT+CONN_STATE` / `CONN_STATE?` | 无响应（命令不存在） |
| `AT+WNBCFG` / `WNBCFG?` | 无响应 |
| `AT+SYSCFG?` / `SYS?` / `FW?` | 无响应 |
| `AT+STA_LIST` / `WLIST` / `WSTATUS` / `WSTATE` / `STATUS` / `STATE?` / `LINK?` / `FORWARD?` | 无响应 |
| `AT+STA0` / `FDB_LIST` / `ARP_LIST` | 无响应 |
| `AT+STA_INFO` | `ERROR`（无 STA 信息） |
| `AT+PAIR` / `PAIR?` | 接受 `PAIR=1` 返回 "Start pairing!"，但 `PAIR?` 始终返回 "Stop pairing!"，**配对状态不能持久**，未见 PAIR SUCCESS |
| `AT+SYSDBG=WNB,1` / `WNB,2` | `OK` 接受但不返回任何数据 |
| `AT+SYSDBG?` / `SYSDBG` | `ERROR`（无法读回调试开关/计数） |
| `AT+VERSION?` | `v2.4.1.3-41928, app:0`（**app:0** 不是 `-WNB.bin` 标识） |
| `AT+KEY?` | `7d78879ee30cdf2cf0beb34298dfd810`（写入的 32 ASCII） |
| `AT+PSK?` | `92bd1578a8fcbab38db2242fbc40d2c4aab61c10e00352f9ff2cb7a153111b`（**64 hex 派生 PMK**） |

### 15.7 用户三个怀疑点的验证

1. **PSK 解释方式不一致** — **部分确认**。小黑盒把 32 ASCII 当作 WPA passphrase 重新派生 PMK（`AT+KEY?`/`AT+PSK?` 两态并存），但机器人 `hgicf.conf` 究竟把同一字符串当 ASCII 还是 raw PMK 本轮**没有验证**。这是一个独立的失败维度，但被固件本身的 WNB 缺失掩盖。
2. **`pair:1` / `forward:1`** — **无法验证**。`AT+WNBCFG` 不响应，`AT+STA_INFO=ERROR`，`AT+PAIR=1` 之后 `PAIR?` 仍报 "Stop pairing!"，pair 状态不能持久，未观察到 PAIR SUCCESS 事件。
3. **固件不是 `-WNB.bin`** — **强支持**。`v2.4.1.3-41928, app:0` 与公开 Mode 2 教程要求的 `-WNB.bin` 标识不符；实测对 `CONN_STATE`/`WNBCFG`/`STA_LIST`/`FDB_LIST`/`SYSCFG`/`SYSDBG?` 全部无响应或 ERROR，是基础 AP 固件而不是 WNB 桥接固件。

PC arping 后再次过滤小黑盒 MAC（排除 DHCP）：

~~~text
ether src ea:5c:43:5f:7f:b8 and not port 67 and not port 68  →  0 帧
~~~

完全符合"WNB 接收计数不增长"路径 → **Ethernet 口未绑定到 WNB 数据面**。

### 15.8 公开资源与官方 Mode 2 流程参考

公开工具与文档（用户提供）：

- LilyGO T-Halow AT 命令列表：`https://github.com/Xinyuan-LilyGO/T-Halow/blob/master/docs/AT_cmd.md`
- Mode 2 教程（含 PAIR、`-WNB.bin` 要求）：`https://github.com/Xinyuan-LilyGO/T-Halow/blob/master/docs/mode2_test.md`
- RJ45 盒硬件：`https://github.com/Xinyuan-LilyGO/T-Halow-RJ45`
- Qiita 实操：`https://qiita.com/furufuru/items/1bbb70490a91f3925371`
- 已知现象报告：`https://github.com/Xinyuan-LilyGO/T-Halow/issues/38`

公开成功案例要求看到 `+CONNECTED`、`forward:1`、`pair:1`、`STA0:[机器人MAC, pair:1, encrypt:1, connect:1]`、Ethernet 互 ping。本轮所有这些字段都拿不到，原因落在固件而不是配置。

### 15.9 当前概率排序

1. **固件不是 -WNB.bin（最确定）** — `v2.4.1.3 app:0` 是基础 AP 固件
2. PSK 解释方式（次要）— 32 ASCII vs 派生 PMK 一致性，本轮未验证
3. WNB 配对未完成（无法验证）— 命令存在但 PAIR 不能持久

### 15.10 本轮现场清理

按用户要求执行（**不动本机 PC 代码**）：

| 操作 | 位置 | 结果 |
|---|---|---|
| `SetTaixinToCommon` | 机器人（经 probe RPC） | `status=suc`，`taixin_mdu` 回到 `10.42.2.50/24`，默认路由 `via 10.42.10.10 dev eth_mdu` |
| 删除自建 | HDU `/home/agi/mc_direct_test` | 已删 |
| 删除自建 | HDU `/home/agi/mc_verify` | 已删 |
| 删除自建 | HDU `/home/agi/run_model3396_real.sh` | 已删 |
| 删除自建 | HDU `/home/agi/a3_deploy_example` | 已删 |
| 删除自建 | HDU `/agibot/data/user_deploy/model3396` | 已删 |
| 删除自建 | MDU `/agibot/data/user_deploy/mc` | 已删 |
| 删除自建 | MDU `/agibot/data/user_deploy/mc_verify` | 已删 |
| 删除自建 | MDU `/agibot/data/user_deploy/model3396` | 已删 |
| 删除自建 | MDU `/tmp/mdu_taixin_mdu.pcap`、`/tmp/mdu_tcpdump.log`、`/tmp/mdu_tcpdump.pid` | 已删 |
| 删除自建 | HDU `/tmp/a3_mqtt_relay_final.log` | 已删 |
| 保留不动 | HDU `/opt/{agibot,ros,custom_ros,coscene,gcc-13.3,zy}` | 官方软件，未触碰 |
| 保留不动 | MDU `/etc/hgicf.conf` | 文件 mtime 仍为 `6月18日 20:52`（出厂日） |
| 保留不动 | MDU `/etc/rc.local` | 未触碰（含 Taixin PSK 派生脚本） |
| 保留不动 | PC `/home/gzy/a3_deploy_example/mc/` | 用户要求保留本机代码 |
| 保留不动 | PC `/home/gzy/a3_deploy_example/mc.zip` | 同上 |

PC 网络配置还原：

~~~text
A3 profile                     → 已 down
A3-HaLow-BlackBox shared IP    → 192.168.77.1/24 已移除
小黑盒 profile                 → 重新激活，enp12s0 = 10.42.10.50/24
ip -br addr show enp12s0
  enp12s0 UP 10.42.10.50/24 fe80::e339:c903:23ff:a479/64
~~~

机器人最终状态：

~~~text
taixin_mdu      UNKNOWN  10.42.2.50/24 fe80::6bfc:eb4:e001:6a7a/64
default via 10.42.10.10 dev eth_mdu proto static metric 103
10.42.2.0/24 dev taixin_mdu proto kernel scope link src 10.42.2.50
/proc/hgicf/status: RADIO:1, FW_STATE:0, ALIVE_TMR:2323, TX_DATA:1573
/etc/hgicf.conf: mode=sta, key_mgmt=WPA-PSK, bss_bw=4, chan_list=六信道, auto_save=1
iptables -t nat POSTROUTING: SNAT 64515 / MASQUERADE 10.42.2.10 (来自 /etc/rc.local，未变)
~~~

### 15.11 下一步真正所需

要让 `.230 ↔ .30` 数据面打通，必须先取得（按优先级）：

1. **智元 A3 专用 WNB 小黑盒固件**（带 `-WNB.bin` 标识、`app:1` 或等价，能响应 `AT+CONN_STATE`/`AT+WNBCFG`）
2. **官方对 `SetTaixinToGCS.wpa_psk` 字段的解释**（ASCII passphrase 还是 raw 32-byte PMK）
3. **官方 PAIR 配对时序**（两侧同时 `PAIR=1`，单边是否生效）
4. **小黑盒 LAN IP / 网关 / NAT 的官方配置方法**（如果走 L3 而非 L2 bridge）
5. **明确小黑盒 `192.168.50.1` 是盒子 LAN 地址，还是由 PC 客户端虚拟提供**

在以上任一项得到明确答复前，不应再次执行 GCS 切换或重写小黑盒配置。

本轮产物：

~~~text
/home/gzy/a3_taixin_bridge_final_20260728_085726/
├── captures/
│   ├── pc_enp12s0.pcap          # 11.5 KB，含 16 次 ARP/PC 端 DHCP/mDNS
│   ├── pc_enp12s0_v2.pcap       # 760 B，WNB 探测后 arping 测试
│   └── mdu_taixin_mdu.pcap      # 20 KB，仅机器人自身 IGMP/mDNS + 318 ARP for .1
├── blackbox/                    # 切换前/后 NetAT 审计记录（已纠正 PSK 输出）
├── wnb_probe_results.txt        # 本轮 WNB 状态决策表逐项实测
├── get_taixin_info.log
├── hdu_mqtt_relay.pid
└── .taixin_psk                  # 32 字符冻结 PSK
~~~

**历史实验记录结束；当前有效状态见第 16 章。**
如有更新，优先复核第 15 章的"当前状态"和"结论分级"，再执行本文档中的审计命令。

---

## 十六、2026-07-29 最终固化状态（当前有效）

> 本章覆盖前文旧实验状态。前文出现的 `10.42.2.50`、旧 SSID 和“数据面不转发”均为历史记录。

### 16.1 最终链路

```text
PC enp12s0: 192.168.50.230/24
  -> Ethernet -> 小黑盒 AP: pingpong / 4 MHz
  -> HaLow 802.11ah
  -> MDU taixin_mdu: 192.168.50.30/24
  -> eth_mdu / HDU / agent / TTS
```

小黑盒管理身份：`192.168.77.65`（DHCP client，地址可能变化），以太网侧 MAC 为 `ea:5c:43:5f:7f:b8`，管理协议为 UDP NetAT/56789，固件为 `v2.4.1.3-41928, app:0`。

### 16.2 小黑盒配置

```ini
WIFIMODE=ap
SSID=pingpong
BSS_BW=4
CHAN_LIST=9060,9100,9140,9180,9220,9260
KEY=7d78879ee30cdf2cf0beb34298dfd810
```

固定 KEY 未再更换。

### 16.3 MDU 永久网络配置

`eth_mdu` 完全保持不动：

```text
eth_mdu：10.42.10.12/24
默认路由：via 10.42.10.10
```

`taixin_mdu` 为专网地址，且没有默认路由：

```text
taixin_mdu：192.168.50.30/24
网关：无
默认路由：无
自动启用：是
```

对应永久配置文件：

```text
/etc/NetworkManager/system-connections/nm-eth_taixin.nmconnection
ipv4.addresses=192.168.50.30/24
ipv4.never-default=true
connection.autoconnect=true
```

### 16.4 开机固化和等待逻辑

已安装并启用：

```text
/etc/systemd/system/taixin-default.service
/usr/local/sbin/taixin-default.sh
```

启动时会等待 HGIC 驱动、写入 `sta / pingpong / WPA-PSK / 4 MHz / 六信道 / auto_save=1`，激活 `nm-taixin_mdu`，并持续等待目标小黑盒 MAC `ea:5c:43:5f:7f:b8` 出现在 `sta_list`。小黑盒晚开机时会等待并重试，因此不再需要 `SetTaixinToGCS`、`hgpriv set` 或 ADB 恢复脚本。

### 16.5 自动语音提示

已安装并启用：

```text
/etc/systemd/system/taixin-voice-on-connect.service
/usr/local/sbin/taixin-voice-on-connect.sh
```

该服务持续等待 `taixin-default.service` 成功（即目标小黑盒真正关联成功），然后通过 MDU→HDU 的既有 `agi` SSH 通道调用 HDU 本地 TTS，播放已验证存在的缓存短句：

```text
好的，给您唱一首儿歌
```

小黑盒未连接时不会播放；关联成功后每次启动播放一次。

### 16.6 完整重启验收

2026-07-29 完全重启后，TTS 日志观察到：

```text
domain=taixin_connected_voice
text="好的，给您唱一首儿歌"
state=playing
playback completed
state=finished
error_msg=""
```

PC 侧已验证：

```bash
sudo ip neigh flush dev enp12s0
sudo arping -I enp12s0 -c 5 192.168.50.30   # 5/5 reply
ping -I enp12s0 -c 5 192.168.50.30          # 5/5 success
nc -zvw3 192.168.50.30 22                  # success
nc -zvw3 192.168.50.30 1883                # success
```

MDU `taixin_mdu` 同时能抓到来自 PC `192.168.50.230` 的 ARP/IP 帧；PC 端 ARP、ICMP、SSH、MQTT 和 MDU 抓包共同证明 Ethernet↔HaLow 数据面已经打通。

### 16.7 注意事项

- 当前系统默认只保留 `pingpong / 192.168.50.30` 这一套专网模式，不再保留 Common/GCS 开机切换逻辑。
- 不得删除 `eth_mdu` 的 `10.42.10.12/24`，也不得把系统默认路由改为 `via 192.168.50.1`。
- 任意新 TTS 文本仍可能因 `offline_cache_miss` 失败；当前自动提示使用的是已验证缓存短句。
- 小黑盒完全断电时机器人会等待关联，不会误播；小黑盒恢复后等待逻辑会继续完成关联和语音触发。
