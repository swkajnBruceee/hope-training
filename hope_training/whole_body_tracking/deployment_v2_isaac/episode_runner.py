"""Single high-level action episode runner for SAC."""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.utils.math import quat_rotate_inverse

from .low_level_isaac_executor import (
    Model21800IsaacExecutor,
    CONTROL_DT,
    DECIMATION,
)

from .fixed_ball_smoke import (
    BALL_INITIAL_POSITION,
    BALL_INITIAL_VELOCITY,
    BALL_INITIAL_SPIN,
    predict_fixed_intercept,
)

from .predictor_observation import PredictorCommand

from .v2_one_ball_env import OneDecisionCommandAssembler

from .return_metrics import (
    detect_outgoing_net_crossing,
    detect_opponent_table_bounce,
    legal_return,
)


CONTACT_FORCE_THRESHOLD = 0.05


class OneBallEpisodeRunner:

    def __init__(self):

        self.ex = Model21800IsaacExecutor()

        self.ball = self.ex.scene["ball"]

        self.sensor = self.ex.scene.sensors[
            "racket_ball_contact"
        ]

        self.assembler = OneDecisionCommandAssembler()

        self.pred = predict_fixed_intercept()


        self.pose = torch.zeros(
            (1,7),
            device=self.ball.device
        )

        self.pose[:,3] = 1.


        self.velocity = torch.zeros(
            (1,6),
            device=self.ball.device
        )


    def set_ball(
        self,
        position,
        velocity,
        angular=(0.,0.,0.)
    ):

        self.pose[:,:3] = (
            self.ex.scene.env_origins
            +
            torch.tensor(
                position,
                device=self.ball.device
            )
        )

        self.velocity[:,:3] = torch.tensor(
            velocity,
            device=self.ball.device
        )

        self.velocity[:,3:] = torch.tensor(
            angular,
            device=self.ball.device
        )


        self.ball.write_root_pose_to_sim(
            self.pose
        )

        self.ball.write_root_velocity_to_sim(
            self.velocity
        )


    def reset_ball(self):

        self.set_ball(
            BALL_INITIAL_POSITION,
            (0,0,0),
            (0,0,0)
        )


        self.ex.scene.write_data_to_sim()



    def step(self, action):

        action=np.asarray(
            action,
            dtype=np.float64
        )


        assert action.shape==(3,)


        self.ex.reset()


        self.reset_ball()


        cmd=PredictorCommand(
            tuple(self.pred.position),
            tuple(self.pred.velocity),
            10.0,
            1.20
        )


        target=self.assembler.build(
            flight_id=1,
            revision_id=1,
            command_seq=1,
            predictor_command=cmd,
            source_now_s=10.0,
            producer_wall_s=100.0,
            current_base_xy=self.ex.initial_base[:2],
            normalized_action=action,
        )


        self.ex.set_target_command(
            target.position_world,
            target.velocity_world,
            target.control_tts_s,
            target.swing_sign,
        )


        contact=False
        cross_net=False
        bounce=False

        contact_time=None
        ball_velocity_at_contact=None
        max_contact_force=0.0
        final_ball_velocity=None

        previous=None
        net_clearance=None


        launch_delay = (
            1.20
            -
            self.pred.flight_time_s
        )


        for tick in range(220):

            elapsed=tick*CONTROL_DT


            state=self.ex._state()


            if (
                elapsed>=launch_delay
                and np.allclose(
                    self.ball.data.root_lin_vel_w[0]
                    .detach()
                    .cpu()
                    .numpy(),
                    0
                )
            ):
                self.set_ball(
                    BALL_INITIAL_POSITION,
                    BALL_INITIAL_VELOCITY,
                    BALL_INITIAL_SPIN,
                )


            target_cmd=self.ex.lifecycle.update(
                None,
                state
            )


            from a3_deploy_onnx_ref_pingpong.observation import build_observation


            obs=build_observation(
                state,
                target_cmd,
                self.ex.last_action,
                self.ex.adapter.default_q,
                self.ex.base_target_xy,
            )


            raw=np.asarray(
                self.ex.policy.infer_target(
                    obs,
                    target_cmd.time_to_strike,
                    self.ex.lifecycle.swing_sign,
                    CONTROL_DT,
                ),
                dtype=np.float32,
            ).reshape(31)


            q=self.ex.adapter.decode(raw)


            self.ex.robot.set_joint_position_target(
                torch.tensor(
                    q,
                    device=self.ex.device,
                    dtype=self.ex.dtype
                ).reshape(1,31),
                joint_ids=self.ex.joint_ids
            )


            self.ex.last_action=raw.copy()


            for _ in range(DECIMATION):

                self.ex.scene.write_data_to_sim()

                self.ex.sim.step(
                    render=False
                )

                self.ex.scene.update(
                    CONTROL_DT/DECIMATION
                )


                bp=self.ball.data.root_pos_w[0]

                bv=self.ball.data.root_lin_vel_w[0]


                p=(
                    bp
                    -
                    self.ex.scene.env_origins[0]
                ).detach().cpu().numpy()


                v=bv.detach().cpu().numpy()


                force=float(
                    torch.linalg.vector_norm(
                        self.sensor.data.net_forces_w,
                        dim=-1
                    ).max()
                )


                if force>CONTACT_FORCE_THRESHOLD:

                    if not contact:
                        contact_time = elapsed
                        ball_velocity_at_contact = v.copy()

                    contact=True

                    max_contact_force=max(
                        max_contact_force,
                        force
                    )


                if previous is not None:

                    event=detect_outgoing_net_crossing(
                        previous,
                        p,
                        v,
                        contact_seen=contact,
                        net_seen=cross_net,
                    )

                    if event:
                        cross_net=event.clears_net
                        net_clearance=event.clearance


                bounce=(
                    bounce
                    or
                    detect_opponent_table_bounce(
                        p,
                        v,
                        contact_seen=contact,
                        bounce_seen=bounce,
                    )
                )


                final_ball_velocity=v.copy()

                previous=p.copy()


            if bounce:
                break



        legal=legal_return(
            contact,
            cross_net,
            bounce
        )


        reward=0

        if contact:
            reward+=20

        if cross_net:
            reward+=30

        if bounce:
            reward+=100

        if legal:
            reward+=200


        return {
            "reward":reward,
            "contact":contact,
            "cross_net":cross_net,
            "bounce":bounce,
            "legal_return":legal,
            "net_clearance":net_clearance,

            "contact_time":contact_time,
            "ball_velocity_at_contact":ball_velocity_at_contact,
            "max_contact_force":max_contact_force,
            "final_ball_velocity":final_ball_velocity,
        }
