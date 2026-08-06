# Third-Party Notices

This repository is licensed under Apache-2.0 at the root. Some starter
materials are adapted from or designed to interoperate with third-party
software and robot assets. Keep the notices below when redistributing this
starter.

## Agibot A3 Reference Materials

Path:

```text
agibot/
```

This directory contains Agibot-provided A3 reference materials, including URDF
packages, the MuJoCo/AimRT simulation reference, and the A3 deployment example.
These files retain their in-tree copyright notices and license declarations;
the repository root Apache-2.0 license does not remove or replace those
notices.

The Isaac Lab quickstart uses:

```text
agibot/URDF/A3T2.5-URDF-std-pingpang/
```

That package metadata declares license `BSD` in `package.xml`.

## Isaac Lab / BeyondMimic Starter

Path:

```text
hope_training/whole_body_tracking/
```

This starter package is adapted from Isaac Lab-style whole-body tracking code
and carries the MIT notice in `hope_training/whole_body_tracking/LICENCE`.
The file spelling follows the upstream package. HOPE-specific A3, table-tennis,
quickstart, and smoke-test changes are part of this public starter branch.

Related upstreams:

- `HybridRobotics/whole_body_tracking`
- NVIDIA Isaac Lab

## ROS / Mocap Workspace

Path:

```text
hope_ws/
```

The HOPE ROS workspace skeleton is provided for optional mocap/planner
integration. It depends on ROS 2 packages and can use `vrpn_mocap`, but the
upstream `vrpn_mocap` package is not vendored in this starter. Teams that need
live VRPN should install or clone it separately in their ROS 2 workspace and
follow that package's own license.

## Agibot A3 Deployment Example

Path:

```text
agibot/code_deployment/
```

This directory contains the Agibot-provided A3 deployment example for HOPE,
including source code, configs, runtime examples, and the third-party material
that Agibot provided with the example. The `thirdparty/joint_msgs/` ROS message
package declares `Mulan PSL v2` in its `package.xml`.

The deployment example also includes or references small third-party helper
components:

- `src/TRTInference/picosha2.h`: MIT license notice in file.
- `src/a3/a3_deploy_onnx_ref/include/xml.h`: MIT license notice in file.
- `src/a3/a3_deploy_onnx_ref/include/cnpy.h` and `src/.../src/cnpy.cpp`: MIT
  license notice in file.
- `cmake/FindTensorRT.cmake`: MIT-style permission notice in file.

Some build paths can fetch optional dependencies such as AimRT, fmt, GoogleTest,
or MuJoCo. Review those upstream build files before enabling them in CI.

## Agibot MuJoCo / AimRT Reference

Path:

```text
agibot/A3_MuJoCo_Sim/aimrt_mujoco_sim/
```

This Agibot-provided reference project includes its own `LICENSE`, which is
Mulan Permissive Software License, Version 2. It also contains nested CI and
build files from the upstream-style project. The upstream workflow files are
preserved as reference material under:

```text
agibot/A3_MuJoCo_Sim/aimrt_mujoco_sim/github_workflows_reference/
```

They are intentionally not placed under a `.github/workflows/` path in this
repository. Public network-fetched CMake dependencies in the Agibot reference
paths are pinned with `URL_HASH` where they are fetched by this tree.

The Agibot bundle keeps separate URDF and MuJoCo mesh layouts. Some visual
meshes are therefore duplicated across those package layouts so each upstream
path can still be used in its native form. Do not replace them with symlinks
unless the MuJoCo XML and Isaac/URDF loading paths are verified on the target
platforms.
