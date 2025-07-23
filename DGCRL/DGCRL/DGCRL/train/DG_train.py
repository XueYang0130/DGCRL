import random

import gym
import argparse
from DGTD3 import DGTD3
from utils import create_directory, plot_learning_curve, scale_action
from gym.envs.registration import register
from util.dataclass import DemoInfo
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import math

os.add_dll_directory("C://Users//19233436//.mujoco//mjpro150//bin")
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/')
parser.add_argument('--env', type=str, default='hopper')
parser.add_argument('--curve_mode', type=str, default="normal")
parser.add_argument('--demo_num', type=int, default=50)
parser.add_argument('--reset_mode', type=str, default="ac")
args = parser.parse_args()
register(
    id='navigation_v1',
    entry_point='environments.navigation_:Navigation2DEnvV1',
)
register(
    id='navigation_v2',
    entry_point='environments.navigation_:Navigation2DEnvV2',
)
register(
    id='navigation_v3',
    entry_point='environments.navigation_:Navigation2DEnvV3',
)
register(
    id='hopper',
    entry_point='environments.hopper_vel:HopperVelEnv',
)

register(
    id='ant',
    entry_point='environments.ant:AntVelEnv',
)

register(
    id='half_cheetah',
    entry_point='environments.half_cheetah:HalfCheetahVelEnv',
)


def scale_action(action, low, high):
    action = np.clip(action, -1, 1)
    weight = (high - low) / 2
    bias = (high + low) / 2
    action_ = action * weight + bias

    return action_


def check_env_available(task):
    small_puddle = task[:2]
    medium_puddle = task[2:4]
    large_puddle = task[4:6]
    goal = task[6:8]
    distance = math.sqrt((small_puddle[0] - goal[0]) ** 2 + (small_puddle[1] - goal[1]) ** 2)
    if distance < 0.11:
        return False
    distance = math.sqrt((medium_puddle[0] - goal[0]) ** 2 + (medium_puddle[1] - goal[1]) ** 2)
    if distance < 0.16:
        return False
    distance = math.sqrt((large_puddle[0] - goal[0]) ** 2 + (large_puddle[1] - goal[1]) ** 2)
    if distance < 0.21:
        return False
    distance = math.sqrt((small_puddle[0]) ** 2 + (small_puddle[1]) ** 2)
    if distance < 0.11:
        return False
    distance = math.sqrt((medium_puddle[0]) ** 2 + (medium_puddle[1]) ** 2)
    if distance < 0.16:
        return False
    distance = math.sqrt((large_puddle[0]) ** 2 + (large_puddle[1]) ** 2)
    if distance < 0.21:
        return False
    return True


if __name__ == '__main__':
    seeds = [
        6,
        14,
        20,
        202,
        405
    ]
    curve_mode = args.curve_mode
    for seed in seeds:
        print(f"SEED is:{seed}")
        demo_num = args.demo_num
        env_name = args.env
        env = gym.make(env_name)
        eval_env = gym.make(env_name)
        reset_mode = args.reset_mode

        if "navigation" in env_name:
            # ---------------- Agent for navigation -------------------------#
            agent = DGTD3(alpha=3e-4, beta=3e-4, state_dim=env.observation_space.shape[0],
                          action_dim=env.action_space.shape[0], actor_fc1_dim=400, actor_fc2_dim=300,
                          critic_fc1_dim=400, critic_fc2_dim=300, ckpt_dir=args.ckpt_dir, gamma=0.99,
                          tau=0.005, action_noise=0.05, policy_noise=0.02, policy_noise_clip=0.02,
                          delay_time=2, max_size=1_000_000, batch_size=128, env=env, seed=seed)

            for file_name in ["rand_demo/DGTD3/" + env_name[-2:] + "/" + str(i) + "_demo_info_" + env_name[-2:] + ".pkl"
                              for i in range(demo_num)]:
                with open(file_name, 'rb') as file:
                    for exp in pickle.load(file):
                        agent.demo_repository.append(exp)
        else:
            if env_name == "hopper":
                # --------------- Agent for Hopper ------------------- #
                agent = DGTD3(alpha=0.0003, beta=0.0003, state_dim=env.observation_space.shape[0],
                              action_dim=env.action_space.shape[0], actor_fc1_dim=400, actor_fc2_dim=300,
                              critic_fc1_dim=400, critic_fc2_dim=300, ckpt_dir=args.ckpt_dir, gamma=0.95,
                              tau=0.005, action_noise=0.05, policy_noise=0.1, policy_noise_clip=0.2,
                              delay_time=2, max_size=400000, batch_size=16, env=env)
            elif env_name == "ant":
                agent = DGTD3(alpha=0.0003, beta=0.0003, state_dim=env.observation_space.shape[0],
                              # 20 for half_cheetah
                              action_dim=env.action_space.shape[0], actor_fc1_dim=400, actor_fc2_dim=300,
                              critic_fc1_dim=400, critic_fc2_dim=300, ckpt_dir=args.ckpt_dir, gamma=0.95,
                              tau=0.005, action_noise=0.1, policy_noise=0.05, policy_noise_clip=0.1,
                              delay_time=2, max_size=400000, batch_size=16, env=env)
            elif env_name == "half_cheetah":
                agent = DGTD3(alpha=0.0003, beta=0.0003, state_dim=20,
                              action_dim=env.action_space.shape[0], actor_fc1_dim=400, actor_fc2_dim=300,
                              critic_fc1_dim=400, critic_fc2_dim=300, ckpt_dir=args.ckpt_dir, gamma=0.95,
                              tau=0.005, action_noise=0.1, policy_noise=0.05, policy_noise_clip=0.1,
                              delay_time=2, max_size=400000, batch_size=16, env=env)
            else:
                # print(env_name=="ant")
                print(f"No such env {env_name} implemented. Please choose from "
                      "'navigation_v1', 'navigation_v2', 'navigation_v3', 'hopper', 'ant', 'half_cheetah'")
                raise NotImplementedError
            for file_name in ["rand_demo/DGTD3/" + env_name + "/" + str(i) + "_demo_info_" + env_name + ".pkl" for i in
                              range(demo_num)]:  # 61 for half_cheetah
                with open(file_name, 'rb') as file:
                    for exp in pickle.load(file):
                        agent.demo_repository.append(exp)

        sum_disc_ret = []
        mean_rews = []
        disc_returns = []
        tasks = np.load("../task_info/" + str(env_name) + "/task_info.npy")

        task_idx = 0
        for task in tasks:  # line 2 in the algorithm

            env.reset_task(task=task)
            eval_env.reset_task(task=task)
            print(f"Task starts @ goal {task}")
            agent.env = env
            # -------------------- navigation ------------------- #
            # agent.reset_critic(alpha=3e-4, beta=3e-4, state_dim=env.observation_space.shape[0],
            #                    action_dim=env.action_space.shape[0], actor_fc1_dim=400, actor_fc2_dim=300,
            #                    critic_fc1_dim=400, critic_fc2_dim=300, max_size=1_000_000,
            #                    batch_size=128, env=env)

            # --------------------- Hopper ------------------------ #
            # agent.reset_critic(alpha=3e-4, beta=3e-4,
            #                    state_dim=env.observation_space.shape[0],
            #                    # state_dim=20,  # only for half_cheetah
            #                    action_dim=env.action_space.shape[0], actor_fc1_dim=400, actor_fc2_dim=300,
            #                    critic_fc1_dim=400, critic_fc2_dim=300, max_size=1_000_000,
            #                    batch_size=128, env=env)
            agent.global_step = 0

            # line 5 in the algorithm
            agent.renew_horizon()

            # line 4 in the algorithm
            demo_info, disc_ret_thres, s_a_traj = agent.get_max_disc_ret(eval_env, no_gamma=True)

            print(f"disc_ret_thres:{disc_ret_thres}")
            # --------------------- uncomment this for navigation ------------------------#
            # if curve_mode == "normal":
            #     if len(demo_info.demo) > 100:
            #         demo_info = DemoInfo(horizon=0,
            #                              disc_ret=-1000)
            #     if disc_ret_thres > -25:
            #         _, disc_rets = agent.js_train_navi(eval_env=eval_env,
            #                                            roll_back_step=3, # v2 1 steps # v3 5 steps
            #                                            demo_info=demo_info,
            #                                            goal=task,
            #                                            disc_ret_thres=disc_ret_thres)
            #     else:
            #         _, disc_rets = agent.train(eval_env=eval_env,
            #                                    goal=task)
            # if curve_mode == "reference":
            #     print("use plain TD3")
            #     _, disc_rets = agent.train(eval_env=eval_env,
            #                                goal=task, seed=seed)
            # disc_returns.append(disc_rets)
            # print(f"average disc return:{np.average(disc_rets)}\t"
            #       f"demo_rep_capacity:{len(agent.demo_repository)}")
            # print("-----------------------------------------")
            # sum_disc_ret += disc_rets
            # else:
            #     print("use plain TD3")
            #     _, disc_rets = agent.train(eval_env=eval_env,
            #                                goal=task)

            # --------------------- uncomment for Hopper&Ant&Half-Cheetah ------------------------#
            # if disc_ret_thres > -np.inf:
            if curve_mode == "normal":
                _, disc_rets = agent.js_train(eval_env=eval_env,
                                              roll_back_step=10,
                                              # navigation is 5 # 15 also works for Hopper, Ant and Halfcheetah
                                              demo_info=demo_info,
                                              disc_ret_thres=disc_ret_thres, seed=seed)
            #     if curve_mode == "reference":
            #         print("use plain TD3")
            #         _, disc_rets = agent.train(eval_env=eval_env,
            #                                    goal=task, seed=seed)
            disc_returns.append(disc_rets)
            print(f"average disc return:{np.average(disc_rets)}\t"
                  f"demo_rep_capacity:{len(agent.demo_repository)}")
            print("-----------------------------------------")
            sum_disc_ret += disc_rets
        #
        sum_performance = 0
        performances_T = []
        for task in tasks:
            demo_info, disc_ret_thres, _ = agent.get_max_disc_ret(eval_env, no_gamma=True, start_from=60)
            env.reset_task(task=task)
            eval_env.reset_task(task=task)
            print(f"Task starts @ goal {task}")
            sum_performance += disc_ret_thres
            performances_T.append(disc_ret_thres)
        sum_performance /= len(tasks)
        print(f"Average Performance:{sum_performance}")

        print(f"Overall tasks' average return: {np.mean(sum_disc_ret)}")
        mean_rews.append(np.mean(sum_disc_ret))
        np.save(
            file="rand_demo/DGTD3/_DGCRL_mean_rews_" + env_name + "_" + curve_mode + "_" + str(seed) + "demo_num" + str(
                demo_num) + ".npy",
            arr=mean_rews)
        np.save(
            file="rand_demo/DGTD3/_DGCRL_returns_" + env_name + "_" + curve_mode + "_" + str(seed) + "demo_num" + str(
                demo_num) + ".npy",
            arr=disc_returns)
        np.save(file="rand_demo/DGTD3/_DGCRL_performances_T_" + env_name + "_" + curve_mode + "_" + str(
            seed) + "demo_num" + str(demo_num) + ".npy",
                arr=performances_T)
