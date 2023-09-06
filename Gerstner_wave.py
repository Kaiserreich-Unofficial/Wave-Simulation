import numpy as np

# 导入matplotlib和mpl_toolkits模块，用于绘制3D图形
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from time import time
import torch


class gerstner_wave():
    def __init__(self, wave_length=50, steepness=0.05, shape=400):
        self.wave_length = wave_length
        self.direct = torch.tensor([1, 1], device='cuda', dtype=torch.float)
        self.freq = 2 * torch.pi / self.wave_length
        self.steepness = steepness
        self.shape = shape

        self.speed = np.sqrt(9.8 / self.freq)  # 波速
        self.amplitude = self.steepness/self.freq  # 波高
        self.a = self.steepness/self.freq
        # 创建一个边长为100的正方形网格
        grid = torch.meshgrid(torch.arange(0, shape), torch.arange(0, shape), indexing="xy")
        # 将网格转换为三维空间中的点，y坐标为0
        points = torch.stack(grid).reshape(2, -1).T.to('cuda')
        points = torch.cat([points[:, 0].reshape(-1, 1),
                        torch.zeros((points.shape[0], 1), device='cuda'), points[:, 1].reshape(-1, 1)], dim=1)
        self.points = points

        self.points_norm = torch.zeros_like(self.points)
        # 在__init__方法中定义sharpness变量，可以根据需要调整它的值
        self.sharpness = 10
        # 初试时间戳
        self.init_time = time()

    def update(self):
        time_update = time() - self.init_time
        f = self.freq*(torch.matmul(self.points[:, [0, 2]], self.direct) - self.speed * time_update)

        self.points[:, 0] = self.points[:, 0] + self.direct[0] * self.amplitude * torch.cos(f) * (1 + 1 / self.sharpness)
        self.points[:, 1] = self.amplitude * torch.sin(f) * (1 + 1 / self.sharpness)
        self.points[:, 2] = self.points[:, 2] + self.direct[1] * self.amplitude * torch.cos(f) * (1 + 1 / self.sharpness)

        # 主次切线叉乘得法线
        tangent = torch.zeros_like(self.points_norm)
        tangent[:, 0] = 1 - self.direct[0] * \
            self.direct[0] * self.steepness * torch.sin(f)
        tangent[:, 1] = self.direct[0] * self.steepness * torch.cos(f)
        tangent[:, 2] = - self.direct[1] * self.direct[0] * torch.sin(f)

        binormal = torch.zeros_like(self.points_norm)
        binormal[:, 0] = - self.direct[0] * self.direct[1] * torch.sin(f)
        binormal[:, 1] = self.direct[1] * self.steepness * torch.cos(f)
        binormal[:, 2] = 1 - self.direct[1] * \
            self.direct[1] * self.steepness * torch.sin(f)

        self.points_norm = torch.cross(tangent, binormal)
        # 法线标准化
        norm = torch.norm(self.points_norm, dim=1)

        norm[norm == 0] = 1e-15  # 避免除以零
        norm = norm.reshape(-1, 1)

        self.points_norm /= norm

    def plot(self):
        self.y_range = 6

        # 创建一个figure对象，设置大小为10x10
        fig = plt.figure(figsize=(10, 10))

        # 创建一个Axes3D对象，用于绘制3D图形
        self.ax = fig.add_subplot(111, projection='3d')

        # 设置坐标轴的范围和标签
        self.ax.set_xlim(0, self.shape)
        self.ax.set_ylim(-self.y_range, self.y_range)
        self.ax.set_zlim(0, self.shape)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # 设置标题
        self.ax.set_title('Gerstner Wave Simulation')
        # 创建一个用于动画的函数，设置帧数为100，间隔为50毫秒，重复为True
        ani = animation.FuncAnimation(
            fig, self.draw, frames=60, interval=10, repeat=True)

        # 显示图形
        plt.show()

    def draw(self, frame):
        # 清除之前的图形
        self.ax.clear()

        # 调用类的update方法，获取点云和法向量
        self.update()

        # 将点云转换为三维数组，用于绘制mesh
        x = np.asarray(self.points[:, 0].reshape(self.shape, self.shape).cpu())
        y = np.asarray(self.points[:, 1].reshape(self.shape, self.shape).cpu())
        z = np.asarray(self.points[:, 2].reshape(self.shape, self.shape).cpu())

        # 绘制mesh，设置颜色为蓝色，透明度为0.8
        self.ax.plot_surface(x, y, z, color='b', alpha=0.8)

        # 设置坐标轴的范围和标签
        self.ax.set_xlim(0, self.shape)
        self.ax.set_ylim(-self.y_range, self.y_range)
        self.ax.set_zlim(0, self.shape)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # 设置标题
        self.ax.set_title('Gerstner Wave Simulation')


a = gerstner_wave()
a.plot()
