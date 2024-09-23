import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py

from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env
from sim_env import make_sim_env, BOX_POSE
from scripted_policy import PickAndTransferPolicy, InsertionPolicy

import IPython

e = IPython.embed


def main(args):
    """
    Generate demonstration data in simulation.
    First rollout the policy (defined in ee space) in ee_sim_env. Obtain the joint trajectory.
    Replace the gripper joint positions with the commanded joint position.
    Replay this joint trajectory (as action sequence) in sim_env, and record all observations.
    Save this episode of data, and continue to next episode of data collection.
    生成模拟演示数据。
    首先在 ee_sim_env 中执行定义在末端执行器（ee）空间中的策略，获取关节轨迹。
    将夹爪的关节位置替换为指令中的关节位置。
    在 sim_env 中重新播放这个关节轨迹（作为动作序列），并记录所有观察数据。
    保存这一集的数据，然后继续收集下一集的数据。
    """

    # -------------------------------------------------------------------------
    task_name = args['task_name']  # task_name: 'sim_transfer_cube_scripted'
    dataset_dir = args['dataset_dir']  # dataset_dir: 'data_save_dir'
    num_episodes = args['num_episodes']  # num_episodes: 50
    onscreen_render = args['onscreen_render']  # onscreen_render: False
    inject_noise = False
    render_cam_name = 'top'

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']  # episode_len: 400
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']  # camera_names: ['top', 'left_wrist', 'right_wrist']
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    """
    策略选择
    """
    if task_name == 'sim_transfer_cube_scripted':
        policy_cls = PickAndTransferPolicy
    elif task_name == 'sim_insertion_scripted':
        policy_cls = InsertionPolicy
    elif task_name == 'sim_transfer_cube_scripted_mirror':
        policy_cls = PickAndTransferPolicy
    else:
        raise NotImplementedError
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    """
    开始生成数据
    """
    success = []  # 初始化成功列表：success 列表用于记录每个 episode 是否成功
    for episode_idx in range(num_episodes):
        print(f'{episode_idx=}')
        print('Rollout out EE space scripted policy')
        # setup the environment  设定环境并执行策略
        env = make_ee_sim_env(task_name)  # 创建环境：通过 make_ee_sim_env 函数创建一个 EE 仿真环境
        ts = env.reset()  # 重置环境：重置环境并获取初始时间步 (ts)
        """
        ts={TimeStep:4},
            'reward':None; discount:None; 0,1,2,3
            'objservation':
                'qpos':{ndarray:(14,)}
                'qvel':{ndarray:(14,)}
                'env_state':{ndarray:(7,)}
                'images':{dict:1}{'top':ndarray(480,640,3)}
                'mocap_pose_left':{ndarray:(7,)}，应该是[x,y,z,w,x,y,z]，即pose+wxyz四元数
                'mocap_pose_right':{ndarray:(7,)}，[x,y,z,w,x,y,z]，即pose+wxyz四元数
                'gripper_ctrl':{ndarray:(4,)}
        """
        episode = [ts]  # 初始化 episode 列表：将初始时间步添加到 episode 列表中
        policy = policy_cls(inject_noise)  # 初始化策略：创建策略实例
        # setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()
        for step in range(episode_len):
            action = policy(ts)  # action是当前step的左右的pose+四元数+gripper状态，其中gripper状态是0-1的数值，总共16个数值，即ndarray(16)
            ts = env.step(action)  # 此行代码在执行 ee_sim_env.py 中的各种Task，
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt.pause(0.002)
        plt.close()

        # -------------------------------------------------------------------------
        """
        计算回报和成功判断
            计算回报：计算 episode 中所有时间步的总回报 (episode_return) 和最大回报 (episode_max_reward)。
            判断成功：如果最大回报等于任务的最大回报，则认为该集成功。
        """
        episode_return = np.sum([ts.reward for ts in episode[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode[1:]])
        if episode_max_reward == env.task.max_reward:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")

        # -------------------------------------------------------------------------
        """
        处理关节轨迹
            获取关节轨迹：从 episode 中提取出每个时间步的关节位置 (qpos) 和夹爪控制 (gripper_ctrl)。
            替换关节位置：用归一化的夹爪控制值替换原始的关节位置。PUPPET_GRIPPER_POSITION_NORMALIZE_FN函数是进行归一化的
            保存子任务信息：复制环境状态中的初始盒子位姿到变量subtask_info中。
        """
        joint_traj = [ts.observation['qpos'] for ts in episode]
        # replace gripper pose with gripper control
        gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
        for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
            left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
            right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[2])
            joint[6] = left_ctrl
            joint[6 + 7] = right_ctrl

        subtask_info = episode[0].observation['env_state'].copy()  # box pose at step 0

        # -------------------------------------------------------------------------
        """
        清理变量和重播关节指令
            清理变量：删除环境、策略和 episode 列表，释放内存。
            重播关节指令：重新创建环境并重播之前记录的关节轨迹，记录重播过程中的所有状态和观察值。
        """
        # clear unused variables
        del env
        del episode
        del policy

        # setup the environment
        print('Replaying joint commands')
        env = make_sim_env(task_name)
        BOX_POSE[0] = subtask_info  # make sure the sim_env has the same object configurations as ee_sim_env
        ts = env.reset()

        episode_replay = [ts]

        # -------------------------------------------------------------------------
        """
        绘图设置和执行策略
            绘图设置：如果 onscreen_render 为 True，设置绘图的轴并初始化显示第一帧。
            策略执行：在每个时间步中，策略根据当前的状态 (ts) 生成一个动作 (action)，并将其应用到环境中 (env.step(action))，然后将新的时间步添加到 episode 列表中。
            实时更新图像：如果启用了渲染，每次迭代都会更新显示的图像。
            关闭绘图：完成后关闭绘图。
        """
        # setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()
        for t in range(len(joint_traj)):  # note: this will increase episode length by 1
            action = joint_traj[t]
            ts = env.step(action)
            episode_replay.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt.pause(0.02)

        # -------------------------------------------------------------------------
        """
        计算回报和成功判断
            计算回报：计算 episode_replay 中所有时间步的总回报 (episode_return) 和最大回报 (episode_max_reward)。
            判断成功：如果最大回报等于任务的最大回报，则认为该集成功。
        """
        episode_return = np.sum([ts.reward for ts in episode_replay[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode_replay[1:]])
        if episode_max_reward == env.task.max_reward:
            success.append(1)
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            success.append(0)
            print(f"{episode_idx=} Failed")

        plt.close()

        # -------------------------------------------------------------------------
        """
        将仿真中的每一步数据记录下来，并保存到一个 HDF5 文件中。数据包括观察到的机器人状态、采取的动作以及相机捕捉的图像
        For each timestep:
        observations
        - images
            - each_cam_name     (480, 640, 3) 'uint8'      图像数据，尺寸为 (480, 640, 3)，数据类型为 uint8
        - qpos                  (14,)         'float64'    机器人的关节位置，共有 14 个浮点数
        - qvel                  (14,)         'float64'    机器人的关节速度，共有 14 个浮点数

        action                  (14,)         'float64'    机器人在该时间步采取的动作，由 14 个浮点数表示
        """

        # -------------------------------------------------------------------------
        """
        数据字典的初始化
        data_dict 初始化：
            '/observations/qpos'：存储每个时间步的关节位置。
            '/observations/qvel'：存储每个时间步的关节速度。
            '/action'：存储每个时间步的动作。!!!这里着重看action里数值的物理含义!!!
        图像数据初始化：
            遍历 camera_names，为每个相机名称初始化一个空列表，用于存储该相机在每个时间步捕捉到的图像。
        """
        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        # -------------------------------------------------------------------------
        """
        调整轨迹长度
        原因：在重播时，实际存储的动作比时间步多一个，而时间步数比动作多两个。为了确保数据的一致性，这里对 joint_traj 和 episode_replay 进行了截断。
            joint_traj[:-1]：删除最后一个动作。
            episode_replay[:-1]：删除最后一个时间步。
        """
        # because the replaying, there will be eps_len + 1 actions and eps_len + 2 timesteps
        # truncate here to be consistent
        joint_traj = joint_traj[:-1]
        episode_replay = episode_replay[:-1]

        # -------------------------------------------------------------------------
        """
        数据收集循环
        最大时间步数：max_timesteps 表示剩余的最大时间步数。
        数据收集：
            action = joint_traj.pop(0)：从 joint_traj 中弹出第一个动作。
            ts = episode_replay.pop(0)：从 episode_replay 中弹出第一个时间步的状态。
            添加关节位置和速度：将当前时间步的关节位置（qpos）和速度（qvel）添加到 data_dict 中。
            添加动作：将当前时间步的动作添加到 data_dict 中。
            添加图像数据：遍历所有相机，将相机图像添加到对应的 data_dict 键下。
        """
        # len(joint_traj) i.e. actions: max_timesteps
        # len(episode_replay) i.e. time steps: max_timesteps + 1
        max_timesteps = len(joint_traj)
        while joint_traj:
            action = joint_traj.pop(0)
            ts = episode_replay.pop(0)
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/qvel'].append(ts.observation['qvel'])
            data_dict['/action'].append(action)
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

        # -------------------------------------------------------------------------
        """
        保存数据到 HDF5 文件
        计时：记录开始保存数据的时间 t0。
        设置文件路径：dataset_path 指定了保存 HDF5 文件的路径。
        创建 HDF5 文件：使用 h5py.File 创建 HDF5 文件，并指定文件模式为 'w'（写模式）。
        文件属性：设置文件的 sim 属性为 True，表示这是仿真数据。
        创建数据组：
            observations 组：创建一个用于存储观察数据的组 obs。
            images 组：在 observations 组中创建一个子组 image，用于存储图像数据。
            图像数据集：为每个相机创建一个数据集，用于存储图像，数据类型为 uint8，并指定数据块大小（chunks）为 (1, 480, 640, 3)。
            关节位置和速度数据集：创建 qpos 和 qvel 数据集，数据类型为 float64。
            动作数据集：创建 action 数据集，数据类型为 float64。
        TODO：数据的物理含义！！
        """
        # HDF5
        t0 = time.time()
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
            # compression='gzip',compression_opts=2,)
            # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
            qpos = obs.create_dataset('qpos', (max_timesteps, 14))
            qvel = obs.create_dataset('qvel', (max_timesteps, 14))
            action = root.create_dataset('action', (max_timesteps, 14))

            """
            填充数据并完成保存
                填充数据：遍历 data_dict，将数据填充到对应的 HDF5 数据集中。
                打印保存时间：打印保存数据所花费的时间。
            """
            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')

    print(f'Saved to {dataset_dir}')
    print(f'Success: {np.sum(success)} / {len(success)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset saving dir', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False)
    parser.add_argument('--onscreen_render', action='store_true')

    main(vars(parser.parse_args()))
