import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from constants import SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env

import IPython

e = IPython.embed


class BasePolicy:
    """
    policy 基类，用于实现机器人控制策略。这个类包含了生成轨迹、在轨迹点之间插值、以及执行策略的基本逻辑
    """

    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise  # 存储是否注入噪声的标志
        self.step_count = 0  # 记录执行策略的步骤计数
        self.left_trajectory = None  # 左臂和右臂的轨迹
        self.right_trajectory = None

    def generate_trajectory(self, ts_first):  # 轨迹生成方法，基类未实现，等待子类实现
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        """
        interpolate 方法：这是一个静态方法，用于在两个轨迹点之间进行插值，以计算当前时间步的机械臂位置、姿态和夹爪状态。
            t_frac：                        计算当前时间步 t 相对于两个轨迹点之间时间的比例。
            curr_xyz, curr_quat, curr_grip：分别是当前轨迹点的位置、姿态（四元数）和夹爪状态。
            next_xyz, next_quat, next_grip：分别是下一个轨迹点的位置、姿态和夹爪状态。
            xyz, quat, gripper：            通过线性插值计算当前时间步的位置、姿态和夹爪状态。
            返回值：                         返回插值后的 xyz（位置）、quat（姿态）和 gripper（夹爪状态）
        """
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def __call__(self, ts):  # 定义子类的使用方法，__call__可以使得类的实例化像函数一样调用
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(ts)  # 在第一步执行时，调用 generate_trajectory 方法生成轨迹

        # obtain left and right waypoints  获取当前和下一个轨迹点
        if self.left_trajectory[0]['t'] == self.step_count:
            self.curr_left_waypoint = self.left_trajectory.pop(0)
        next_left_waypoint = self.left_trajectory[0]

        if self.right_trajectory[0]['t'] == self.step_count:
            self.curr_right_waypoint = self.right_trajectory.pop(0)
        next_right_waypoint = self.right_trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command 插值计算当前状态
        """
        使用 interpolate 方法在当前和下一个轨迹点之间进行插值，计算出当前时间步左臂和右臂的位置、姿态和夹爪状态
        """
        left_xyz, left_quat, left_gripper = self.interpolate(self.curr_left_waypoint, next_left_waypoint,
                                                             self.step_count)
        right_xyz, right_quat, right_gripper = self.interpolate(self.curr_right_waypoint, next_right_waypoint,
                                                                self.step_count)

        # Inject noise  注入噪声
        """
        如果 inject_noise 为 True，则在左臂和右臂的位置上加入一定范围的随机噪声。噪声的范围由 scale 参数决定，这里设置为 0.01
        """
        if self.inject_noise:
            scale = 0.01  # 噪声的随机数范围为xyz方向 -1cm 到 1cm
            left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)
            right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)

        # 构建动作向量并返回
        """
        构建动作向量：
            action_left：将左臂的插值结果（位置、姿态、夹爪状态）拼接成一个动作向量。
            action_right：同样，将右臂的插值结果拼接成一个动作向量。
            最后将左臂和右臂的动作向量组合成一个完整的动作返回。
            更新步骤计数：增加 step_count，以便下一步使用。
            返回动作：返回包含左臂和右臂动作的完整动作向量
        """
        action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])
        action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])

        self.step_count += 1
        return np.concatenate([action_left, action_right])


class PickAndTransferPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        """
        使用初始的时间步 ts_first 来生成左右机械臂的轨迹
        """
        # -------------------------------------------------------------------------
        """
        初始位姿：从 ts_first 的观测值中提取左右机械臂的初始位姿。mocap_pose_right 和 mocap_pose_left 分别表示右臂和左臂的初始位置和姿态（四元数）。
        """
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        # -------------------------------------------------------------------------
        """
        盒子信息：从环境状态中提取目标物体（如盒子）的位置信息 box_xyz 和姿态信息 box_quat。
        """
        box_info = np.array(ts_first.observation['env_state'])
        box_xyz = box_info[:3]
        box_quat = box_info[3:]
        # print(f"Generate trajectory for {box_xyz=}")

        # -------------------------------------------------------------------------
        """
        计算目标位姿和四元数
        右臂末端执行器的目标四元数：获取右臂末端执行器的初始姿态，并在此基础上进行旋转操作，将其绕 Y 轴旋转 -60 度，以生成抓取物体时的目标姿态。
        """
        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        # -------------------------------------------------------------------------
        """
        左臂的目标四元数：定义一个左臂的目标四元数，使其绕 X 轴旋转 90 度，这是在与右臂配合时的姿态。
        """
        meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)

        # -------------------------------------------------------------------------
        """
        目标位置：定义一个机械臂在空间中移动的目标位置 meet_xyz，通常用于双臂之间的协调操作（例如交接物体）
        """
        meet_xyz = np.array([0, 0.5, 0.25])

        # -------------------------------------------------------------------------
        """
        左臂轨迹：定义了一个包含多个时间步的轨迹，每个时间步定义了时间 t、位置 xyz、四元数 quat 和夹爪状态 gripper。具体动作如下：
            t=0：  保持初始位置，夹爪闭合。
            t=100：向交接位置靠近。
            t=260：移动到交接位置。
            t=310：关闭夹爪（假设抓取物体）。
            t=360：左移一段距离。
            t=400：保持在该位置不动。
        """
        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
            {"t": 100, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1},
            # approach meet position
            {"t": 260, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1},
            # move to meet position
            {"t": 310, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 0},
            # close gripper
            {"t": 360, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0},
            # move left
            {"t": 400, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0},
            # stay
        ]

        # -------------------------------------------------------------------------
        """
        右臂轨迹：与左臂类似，定义了右臂的多个时间步，动作如下：
            t=0：  保持初始位置，夹爪闭合。
            t=90： 向目标物体（如盒子）靠近。
            t=130：向下移动，接近物体表面。
            t=170：关闭夹爪，抓取物体。
            t=200：向交接位置靠近。
            t=220：移动到交接位置。
            t=310：打开夹爪，释放物体。
            t=360：右移一段距离。
            t=400：保持在该位置不动。
        """
        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
            {"t": 90, "xyz": box_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 1},
            # approach the cube
            {"t": 130, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 1},
            # go down
            {"t": 170, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 0},
            # close gripper
            {"t": 200, "xyz": meet_xyz + np.array([0.05, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 0},
            # approach meet position
            {"t": 220, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 0},  # move to meet position
            {"t": 310, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 1},  # open gripper
            {"t": 360, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1},
            # move to right
            {"t": 400, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1},
            # stay
        ]


class InsertionPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        peg_info = np.array(ts_first.observation['env_state'])[:7]
        peg_xyz = peg_info[:3]
        peg_quat = peg_info[3:]

        socket_info = np.array(ts_first.observation['env_state'])[7:]
        socket_xyz = socket_info[:3]
        socket_quat = socket_info[3:]

        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        gripper_pick_quat_left = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60)

        meet_xyz = np.array([0, 0.5, 0.15])
        lift_right = 0.00715

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
            {"t": 120, "xyz": socket_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_left.elements,
             "gripper": 1},  # approach the cube
            {"t": 170, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements,
             "gripper": 1},  # go down
            {"t": 220, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements,
             "gripper": 0},  # close gripper
            {"t": 285, "xyz": meet_xyz + np.array([-0.1, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0},
            # approach meet position
            {"t": 340, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements,
             "gripper": 0},  # insertion
            {"t": 400, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements,
             "gripper": 0},  # insertion
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
            {"t": 120, "xyz": peg_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 1},
            # approach the cube
            {"t": 170, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements,
             "gripper": 1},  # go down
            {"t": 220, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements,
             "gripper": 0},  # close gripper
            {"t": 285, "xyz": meet_xyz + np.array([0.1, 0, lift_right]), "quat": gripper_pick_quat_right.elements,
             "gripper": 0},  # approach meet position
            {"t": 340, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements,
             "gripper": 0},  # insertion
            {"t": 400, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements,
             "gripper": 0},  # insertion

        ]


def test_policy(task_name):
    # example rolling out pick_and_transfer policy
    onscreen_render = True
    inject_noise = False

    # setup the environment
    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    if 'sim_transfer_cube' in task_name:
        env = make_ee_sim_env('sim_transfer_cube')
    elif 'sim_insertion' in task_name:
        env = make_ee_sim_env('sim_insertion')
    else:
        raise NotImplementedError

    for episode_idx in range(2):
        ts = env.reset()
        episode = [ts]
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images']['angle'])  # plt_img = ax.imshow(ts.observation['images']['top'])
            plt.ion()

        policy = PickAndTransferPolicy(inject_noise)
        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images']['angle'])  # plt_img.set_data(ts.observation['images']['top'])
                plt.pause(0.02)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        if episode_return > 0:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")


if __name__ == '__main__':
    test_task_name = 'sim_transfer_cube_scripted'
    test_policy(test_task_name)
