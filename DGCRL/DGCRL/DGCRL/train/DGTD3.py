import torch as T
import torch.nn.functional as F
import numpy as np
from networks import ActorNetwork, CriticNetwork
from buffer import ReplayBuffer
from util.dataclass import DemoInfo
from utils import create_directory, plot_learning_curve, scale_action
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
r_small = 0.1
r_medium = 0.15
r_large = 0.2
r_goal = 0.01
r_start = 0.01


class DGTD3:
    def __init__(self, alpha, beta, state_dim, action_dim, actor_fc1_dim, actor_fc2_dim,
                 critic_fc1_dim, critic_fc2_dim, ckpt_dir, gamma=0.99, tau=0.005, action_noise=0.05,
                 policy_noise=0.1, policy_noise_clip=0.1, delay_time=2, max_size=1000000,
                 batch_size=256, device="auto", env=None, self_evolution=True, seed=42):
        self.gamma = gamma
        self.tau = tau
        self.action_noise = action_noise
        self.policy_noise = policy_noise
        self.policy_noise_clip = policy_noise_clip
        self.delay_time = delay_time
        self.update_time = 0
        self.checkpoint_dir = ckpt_dir

        self.actor = ActorNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                  fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim)
        self.critic1 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                     fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)
        self.critic2 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                     fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)

        self.target_actor = ActorNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                         fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim)
        self.target_critic1 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                            fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)
        self.target_critic2 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                            fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)

        self.memory = ReplayBuffer(max_size=max_size, state_dim=state_dim, action_dim=action_dim,
                                   batch_size=batch_size)

        self.update_network_parameters(tau=1.0)
        self.device = device
        self.demo_repository = []
        self.env = env
        self.global_step = 0
        self.self_evolution = self_evolution
        # 设置PyTorch的随机种子
        T.manual_seed(seed)

        # 如果你在使用CUDA（GPU），也需要设置以下这行来确保CUDNN的行为一致
        T.backends.cudnn.deterministic = True
        T.backends.cudnn.benchmark = False

        # 设置numpy的随机种子（因为某些操作可能使用numpy）
        np.random.seed(seed)

    def reset_weights(self, alpha, beta, state_dim, action_dim, actor_fc1_dim, actor_fc2_dim,
                      critic_fc1_dim, critic_fc2_dim, max_size=1000000,
                      batch_size=256, env=None):
        self.actor = ActorNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                  fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim)
        self.critic1 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                     fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)
        self.critic2 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                     fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)

        self.target_actor = ActorNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                         fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim)
        self.target_critic1 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                            fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)
        self.target_critic2 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                            fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)

        self.memory = ReplayBuffer(max_size=max_size, state_dim=state_dim, action_dim=action_dim,
                                   batch_size=batch_size)
        self.env = env

    def reset_critic(self, alpha, beta, state_dim, action_dim, actor_fc1_dim, actor_fc2_dim,
                     critic_fc1_dim, critic_fc2_dim, max_size=1000000,
                     batch_size=256, env=None):
        self.actor = ActorNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                  fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim)
        # only uncomment for reference curve
        # self.critic1 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
        #                              fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)
        # self.critic2 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
        #                              fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)

        self.target_actor = ActorNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                         fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim)
        # only uncomment for reference curve
        # self.target_critic1 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
        #                                     fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)
        # self.target_critic2 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
        #                                     fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)

        self.memory = ReplayBuffer(max_size=max_size, state_dim=state_dim, action_dim=action_dim,
                                   batch_size=batch_size)
        # self.environments = environments

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for actor_params, target_actor_params in zip(self.actor.parameters(),
                                                     self.target_actor.parameters()):
            target_actor_params.data.copy_(tau * actor_params + (1 - tau) * target_actor_params)

        for critic1_params, target_critic1_params in zip(self.critic1.parameters(),
                                                         self.target_critic1.parameters()):
            target_critic1_params.data.copy_(tau * critic1_params + (1 - tau) * target_critic1_params)

        for critic2_params, target_critic2_params in zip(self.critic2.parameters(),
                                                         self.target_critic2.parameters()):
            target_critic2_params.data.copy_(tau * critic2_params + (1 - tau) * target_critic2_params)

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, observation, train=True):
        self.actor.eval()
        state = T.tensor([observation], dtype=T.float).to(device)
        action = self.actor.forward(state)
        # action /=10
        if train:
            # exploration noise
            noise = T.tensor(np.random.normal(loc=0.0, scale=self.action_noise),
                             dtype=T.float).to(device)
            # print(f"pre_a:{action}")
            action = T.clamp(action + noise, -1, 1)

            # action = T.clamp(action + noise, -0.1, 0.1)
            # print(action)
        self.actor.train()

        return action.squeeze().detach().cpu().numpy()

    def learn(self):
        if not self.memory.ready():
            return

        states, actions, rewards, states_, terminals = self.memory.sample_buffer()
        states_tensor = T.tensor(states, dtype=T.float).to(device)
        actions_tensor = T.tensor(actions, dtype=T.float).to(device)
        rewards_tensor = T.tensor(rewards, dtype=T.float).to(device)
        next_states_tensor = T.tensor(states_, dtype=T.float).to(device)
        terminals_tensor = T.tensor(terminals).to(device)

        with T.no_grad():
            next_actions_tensor = self.target_actor.forward(next_states_tensor)
            action_noise = T.tensor(np.random.normal(loc=0.0, scale=self.policy_noise),
                                    dtype=T.float).to(device)
            # smooth noise
            action_noise = T.clamp(action_noise, -self.policy_noise_clip, self.policy_noise_clip)
            next_actions_tensor = T.clamp(next_actions_tensor + action_noise, -1, 1)
            q1_ = self.target_critic1.forward(next_states_tensor, next_actions_tensor).view(-1)
            q2_ = self.target_critic2.forward(next_states_tensor, next_actions_tensor).view(-1)
            q1_[terminals_tensor] = 0.0
            q2_[terminals_tensor] = 0.0
            critic_val = T.min(q1_, q2_)
            target = rewards_tensor + self.gamma * critic_val
        q1 = self.critic1.forward(states_tensor, actions_tensor).view(-1)
        q2 = self.critic2.forward(states_tensor, actions_tensor).view(-1)

        critic1_loss = F.mse_loss(q1, target.detach())
        critic2_loss = F.mse_loss(q2, target.detach())
        critic_loss = critic1_loss + critic2_loss
        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        self.update_time += 1
        if self.update_time % self.delay_time != 0:
            return

        new_actions_tensor = self.actor.forward(states_tensor)
        q1 = self.critic1.forward(states_tensor, new_actions_tensor)
        actor_loss = -T.mean(q1)
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def save_models(self, episode):
        self.actor.save_checkpoint(self.checkpoint_dir + 'Actor/TD3_actor_{}.pth'.format(episode))
        # print('Saving actor network successfully!')
        self.target_actor.save_checkpoint(self.checkpoint_dir +
                                          'Target_actor/TD3_target_actor_{}.pth'.format(episode))
        # print('Saving target_actor network successfully!')
        self.critic1.save_checkpoint(self.checkpoint_dir + 'Critic1/TD3_critic1_{}.pth'.format(episode))
        # print('Saving critic1 network successfully!')
        self.target_critic1.save_checkpoint(self.checkpoint_dir +
                                            'Target_critic1/TD3_target_critic1_{}.pth'.format(episode))
        # print('Saving target critic1 network successfully!')
        self.critic2.save_checkpoint(self.checkpoint_dir + 'Critic2/TD3_critic2_{}.pth'.format(episode))
        # print('Saving critic2 network successfully!')
        self.target_critic2.save_checkpoint(self.checkpoint_dir +
                                            'Target_critic2/TD3_target_critic2_{}.pth'.format(episode))
        # print('Saving target critic2 network successfully!')
        # print("model saved!")

    def load_models(self, episode):
        self.actor.load_checkpoint(self.checkpoint_dir + 'Actor/TD3_actor_{}.pth'.format(episode))
        # print('Loading actor network successfully!')
        self.target_actor.load_checkpoint(self.checkpoint_dir +
                                          'Target_actor/TD3_target_actor_{}.pth'.format(episode))
        # print('Loading target_actor network successfully!')
        self.critic1.load_checkpoint(self.checkpoint_dir + 'Critic1/TD3_critic1_{}.pth'.format(episode))
        # print('Loading critic1 network successfully!')
        self.target_critic1.load_checkpoint(self.checkpoint_dir +
                                            'Target_critic1/TD3_target_critic1_{}.pth'.format(episode))
        # print('Loading target critic1 network successfully!')
        self.critic2.load_checkpoint(self.checkpoint_dir + 'Critic2/TD3_critic2_{}.pth'.format(episode))
        # print('Loading critic2 network successfully!')
        self.target_critic2.load_checkpoint(self.checkpoint_dir +
                                            'Target_critic2/TD3_target_critic2_{}.pth'.format(episode))
        # print('Loading target critic2 network successfully!')

    def renew_horizon(self):
        for demo_info in self.demo_repository:
            demo_info.horizon = len(demo_info.demo)

    def get_max_disc_ret(self, eval_env, no_gamma=False, start_from=0):
        max_demo_info = None
        max_disc_ret = -np.inf
        for i in range(len(self.demo_repository)):
            disc_ret, ret, new_demo, _ = self.play_one_episode(eval_env=eval_env,
                                                               demo_info=self.demo_repository[i],
                                                               verbose=False,
                                                               no_gamma=no_gamma)
            if disc_ret > max_disc_ret:
                max_demo_info = self.demo_repository[i]
                max_disc_ret = disc_ret

        s_a_traj = []


        return max_demo_info, max_disc_ret, s_a_traj

    def remove_obsolete_demos(self):
        max_disc_ret = self.get_max_disc_ret()
        for i in reversed(range(len(self.demo_repository))):
            updated = self.demo_repository[i].updated

            is_dominated = self.demo_repository[i].disc_ret < max_disc_ret
            if updated and is_dominated:
                print(
                    # f"demo:{self.demo_repository[i].demo} "
                    f"- DEMO LEN{len(self.demo_repository[i].demo)}\twith vec:{np.round_(self.demo_repository[i].disc_ret, 4)} is removed")
                self.demo_repository.pop(i)

    def add_new_demo_info(self, new_demo, disc_ret, horizon):
        if horizon is None:
            d_info = DemoInfo(demo=new_demo, disc_ret=disc_ret, updated=False, horizon=len(new_demo))
        else:
            d_info = DemoInfo(demo=new_demo, disc_ret=disc_ret, updated=False, horizon=horizon)
        self.demo_repository.append(d_info)
        return d_info

    def play_one_episode(self, eval_env, demo_info=DemoInfo, verbose=False, no_gamma=False, horizon_limit=100):

        new_demo = []
        path = []
        action_pointer = 0
        steps = 0
        terminated = False
        # for Hopper
        # obs, _ = eval_env.reset(seed=42)

        # for navigation
        obs = eval_env.reset()
        path.append(obs)
        while not terminated and steps < horizon_limit:
            steps += 1
            if action_pointer < demo_info.horizon:
                action = demo_info.demo[action_pointer]
                action_pointer += 1
                # print(action)
            else:
                action = self.choose_action(obs, train=False)
            obs_, reward, terminated, _ = eval_env.step(action)
            obs = obs_
            new_demo.append(action)
            path.append(obs)

        if verbose:
            print(f"eval action traj:{new_demo}")

        disc_ret, ret = self.evaluate_demo(demo=new_demo, env=eval_env, no_gamma=no_gamma)

        return disc_ret, ret, new_demo, path

    def evaluate_demo(self, demo, env, no_gamma=False):
        gamma = 1
        disc_ret = 0
        ret = 0
        if no_gamma:
            _gamma = self.gamma
            self.gamma = 1
        # for Hopper
        # obs, _ = env.reset(seed=42)
        # obs = env.reset()
        # for navigation
        obs = env.reset()
        for action in demo:
            # action = scale_action(action, low=env.action_space.low, high=env.action_space.high)
            obs_, reward, terminated, _ = env.step(action)
            disc_ret += gamma * reward
            ret += reward
            gamma *= self.gamma
            if terminated:
                break
        if no_gamma:
            self.gamma = _gamma
        return disc_ret, ret

    def train(self, eval_env, goal, seed=None):
        episodes = 0
        disc_rets = []
        paths = []
        while episodes < 100:
            step = 0
            episodes += 1
            disc_ret = 0
            gamma = 1
            # for locomotion
            # obs, _ = self.env.reset(seed=seed)
            # for navigation
            obs = self.env.reset()
            done = False
            # path = [obs]
            while not done and step < 100:  # while episode not end
                step += 1
                self.global_step += 1
                action = self.choose_action(obs, train=True)  # returned action
                obs_, reward, done, _ = self.env.step(action)
                # path.append(obs_)
                self.remember(obs, action, reward, obs_, done)
                self.learn()
                obs = obs_
                disc_ret += reward * gamma
                # gamma *= self.gamma
            # paths.append(path)

            # disc_ret, ret, new_demo, path = self.play_one_episode(eval_env=eval_env,
            #                                                       demo_info=DemoInfo(horizon=0),
            #                                                       verbose=False,
            #                                                       no_gamma=True)
            # self.action_noise = 0.08 - 0.07 * episodes / 100
            print(f"episode:{episodes}\tdisc_ret:{disc_ret}")
            disc_rets.append(disc_ret)
            # if episodes % 20 == 0:
            #     paths.append(path)
        # self.visualize_path(goal=goal, paths=paths)
        return np.mean(disc_rets), disc_rets

    def js_train(self,
                 eval_env=None,
                 roll_back_step=2,
                 demo_info=None,
                 disc_ret_thres=0,
                 seed=42):
        episode_num = 500
        demo_info.horizon -= max(roll_back_step, 0)  # line 7 in the algorithm
        episode = 1
        disc_rets = []
        # Show the best demo so far
        print(f"best disc reward:{demo_info.disc_ret}\tlen:{demo_info.horizon}\n")
        pi_g_pointer = 0
        obs, _ = self.env.reset()
        paths = []
        step = 0
        departed = False  # small trick to keep the exploration on track
        while episode < episode_num + 1:
            self.global_step += 1
            if departed == False:
                # line 8 in the algorithm
                if demo_info.horizon > 0 and pi_g_pointer < demo_info.horizon:
                    action = demo_info.demo[:demo_info.horizon][pi_g_pointer]
                    pi_g_pointer += 1
                else:
                    action = self.choose_action(obs, train=True)
            else:
                action = self.choose_action(obs, train=True)
            obs_, reward, terminated, info = self.env.step(action)

            self.remember(obs, action, reward, obs_, terminated)

            self.learn()  # line 9 in the algorithm
            obs = obs_
            step += 1
            if self.global_step % 100 == 0:
                # line 10 in the algorithm
                disc_ret, ret, new_demo, _ = self.play_one_episode(eval_env=eval_env,
                                                                   demo_info=demo_info,
                                                                   verbose=False,
                                                                   no_gamma=True)
                if disc_ret_thres > 0:
                    # passing threshold factor, when the discounted return threshold is greater than 0
                    # the passing threshold factor starts from 0.7
                    factor = min(1.0, 0.7 + 0.4 * episode / episode_num)
                else:
                    # if the discounted return threshold is less than 0
                    # the passing threshold factor starts from 1.3 and gradually reduce to 1
                    factor = max(1.0, 1.3 - 0.4 * episode / episode_num)

                # line 11 in the algorithm
                if np.round_(disc_ret, 4) >= np.round_(disc_ret_thres, 4) * factor:
                    # for navigation the threshold factor is 1.2 because the reward is negative

                    # line 13 in the algorithm
                    demo_info.horizon = max(demo_info.horizon - roll_back_step, 0)

                    if self.self_evolution:  # only with the self-evolving mechanism the demo repo will be updated
                        if np.round_(disc_ret, 4) >= np.round_(disc_ret_thres, 4):
                            if demo_info.horizon <= 0:  # for hopper
                                disc_ret_thres = disc_ret
                                self.renew_horizon()
                            print(f"replace old demo {len(demo_info.demo)} with better demo:{len(new_demo)} "
                                  f"\nhorizon/len:{demo_info.horizon}/{len(demo_info.demo)} ")

                            # line 12 in the algorithm
                            demo_info = self.add_new_demo_info(new_demo=new_demo,
                                                               disc_ret=disc_ret,
                                                               horizon=len(new_demo))

                print(f"episode:{episode}\tguide policy horizon:{demo_info.horizon}\t"
                      f"guided_disc_ret:{np.round_(disc_ret, 4)}\t"
                      f"disc_ret_threshold:{np.round_(disc_ret_thres, 4)}\t")

            if terminated or step >= 100:
                if not departed:
                    departed = True
                if departed:
                    departed = False
                step = 0
                obs, _ = self.env.reset()
                g = self.gamma
                self.gamma = 1
                eval_result, _, _, path = self.play_one_episode(eval_env=eval_env,
                                                                demo_info=demo_info,
                                                                no_gamma=True)
                self.gamma = g
                if episode % 20 == 0:
                    paths.append(path)
                disc_rets.append(eval_result)
                pi_g_pointer = 0
                episode += 1

        print(f"eval result:{disc_rets}\tdisc_ret_thres:{disc_ret_thres}")
        return self.demo_repository, disc_rets

    def js_train_navi(self,
                      eval_env=None,
                      roll_back_step=2,
                      demo_info=None,
                      goal=None,
                      disc_ret_thres=0,
                      seed=42):
        episode_num = 100
        self.action_noise = 0.08  # to encourage the early stage exploration
        # line 7 in the algorithm
        demo_info.horizon -= max(roll_back_step, 0)
        episode = 1
        disc_rets = []
        print(f"best disc reward:{disc_ret_thres}\tlen:{demo_info.horizon}\n")
        pi_g_pointer = 0
        obs = self.env.reset()
        paths = []

        step = 0
        departed = False
        while episode < episode_num + 1:
            self.global_step += 1
            # line 8 in the algorithm
            if demo_info.horizon > 0 and pi_g_pointer < demo_info.horizon:
                action = demo_info.demo[:demo_info.horizon][pi_g_pointer]
                pi_g_pointer += 1
            else:
                action = self.choose_action(obs, train=True)

            obs_, reward, terminated, info = self.env.step(action)
            self.remember(obs, action, reward, obs_, terminated)

            self.learn() # line 9 in the algorithm
            obs = obs_
            step += 1
            if self.global_step % 100 == 0:
                # line 10 in the algorithm
                disc_ret, ret, new_demo, _ = self.play_one_episode(eval_env=eval_env,
                                                                   demo_info=demo_info,
                                                                   verbose=False,
                                                                   no_gamma=True)
                # line 11 in the algorithm
                if np.round_(disc_ret, 4) >= np.round_(disc_ret_thres, 4) * 1.2:
                    # line 13 in the algorithm
                    demo_info.horizon = max(demo_info.horizon - roll_back_step, 0)

                    # only with the self-evolving mechanism the demo repo will be updated
                    if self.self_evolution:
                        if np.round_(disc_ret, 4) >= np.round_(disc_ret_thres, 4):
                            print(f"replace old demo {len(demo_info.demo)} with better demo:{len(new_demo)} "
                                  f"\nhorizon/len:{demo_info.horizon}/{len(demo_info.demo)} ")
                            if len(new_demo) < len(demo_info.demo):
                                horizon = len(new_demo)
                            else:
                                horizon = demo_info.horizon
                            if len(new_demo) < 100:
                                # line 12 in the algorithm
                                demo_info = self.add_new_demo_info(new_demo=new_demo,
                                                                   disc_ret=disc_ret,
                                                                   horizon=horizon)
                                disc_ret_thres = disc_ret
                                if disc_ret_thres == -1000:
                                    # as the target of navigation can sometime be some unreachable point for all
                                    # demos in the repository, we use the discounted return of the current return as the
                                    # threshold
                                    disc_ret_thres = demo_info.disc_ret

                print(f"episode:{episode}\tguide policy horizon:{demo_info.horizon}\t"
                      f"guided_disc_ret:{np.round_(disc_ret, 4)}\t"
                      f"disc_ret_threshold:{np.round_(disc_ret_thres, 4)}\t")

            if terminated or step >= 100:
                if not departed:
                    departed = True
                if departed:
                    departed = False
                step = 0
                self.action_noise = 0.08 - 0.07 * episode / 100
                # for navigation
                obs = self.env.reset()
                g = self.gamma
                self.gamma = 1
                eval_result, _, demo_, path = self.play_one_episode(eval_env=eval_env,
                                                                    demo_info=demo_info,
                                                                    no_gamma=True)
                self.gamma = g
                if episode % 20 == 0:
                    paths.append(path)
                disc_rets.append(eval_result)
                pi_g_pointer = 0
                episode += 1

        print(f"eval result:{disc_rets}\tdisc_ret_thres:{disc_ret_thres}")
        return self.demo_repository, disc_rets

    def visualize_path(self, goal, paths, start_pos=(0, 0)):
        fig, ax = plt.subplots(figsize=(8, 8))

        # 设定圆的数据：圆心和半径
        circles = [
            (tuple(goal[:2]), r_small),  # (圆心x, 圆心y), 半径
            (tuple(goal[2:4]), r_medium),
            (tuple(goal[4:6]), r_large),
            (tuple(goal[6:8]), r_goal),
            (start_pos, r_start)
        ]

        # 为每个圆添加一个Circle对象到图形中
        for i in range(len(circles)):
            center = circles[i][0]
            radius = circles[i][1]
            print(f"i:{i}\tcenter:{center}\tradius:{radius}")
            if i == 3:
                color = "green"
                facecolor = "none"
            elif i == 4:
                color = "red"
                facecolor = "none"
            else:
                color = "grey"
                facecolor = "grey"
            circle = Circle(center, radius, edgecolor=color, facecolor=facecolor, linewidth=2)
            ax.add_patch(circle)

        cmap = plt.cm.viridis
        colors = cmap(np.linspace(0, 1, len(paths)))
        for i in range(len(paths)):
            path = paths[i]
            ax.plot(*zip(*path),
                    marker='o',
                    color=colors[i],
                    linestyle='-',
                    linewidth=2,
                    markersize=2,
                    label=f"epi. {str((i + 1) * 20)} | path len {len(path)} | end @ {np.round_(path[-1], 2)}")

        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(-0.7, 0.7)
        ax.set_aspect('equal', adjustable='datalim')
        plt.title(f"DGTD3 action noise:{0.02}  lr:{3e-4}  truncated at:{100}")
        plt.legend()
        plt.show()
