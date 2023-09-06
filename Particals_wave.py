import random
import numpy as np
import open3d as o3d
from math import pi,exp
import torch

# Sigmoid 函数
def sigmoid(x):
    return 1/(1 + exp(-x))

# 波浪粒子的数量
NUM_PARTICLES = 100000

# 生成随机波浪粒子函数
def create_wave_particle():
    x = random.uniform(-100, 100)  # 随机x坐标
    y = random.uniform(-100, 100)  # 随机y坐标
    z = 0  # z坐标为零平面
    amplitude = random.uniform(-10, 10)  # 随机振幅
    wavelength = random.uniform(25, 50)  # 随机波长
    speed_x = random.uniform(-10, 10)  # 随机x方向的速度
    speed_y = random.uniform(-10, 10)  # 随机y方向的速度
    phase = random.uniform(0, 2 * pi)  # 随机相位
    wave_particle = np.array(
        [x, y, z, amplitude, wavelength, speed_x, speed_y, phase])  # 创建波浪粒子数组
    return wave_particle

# 创建波浪粒子数组
wave_particles = np.array([create_wave_particle() for i in range(NUM_PARTICLES)])

# 创建一个散点图对象，用于表示波浪粒子，设置点的大小为10，颜色为深蓝色（透明度随z值变化）
points = wave_particles[:, :3] # 获取波浪粒子的位置数组（前三列）
colors = np.array([[0.0,0.0,sigmoid(wave_particle[2])] for wave_particle in wave_particles]) # 设置颜色为浅蓝到深蓝的渐变，根据z值计算sigmoid函数的值作为蓝色分量

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors) # 添加颜色属性

# 创建一个XYZ轴对象，用于显示三维坐标轴，设置轴的长度和颜色
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=200, origin=[0, 0, 0])

# 将散点图和XYZ轴添加到可视化窗口中
vis = o3d.visualization.Visualizer()
vis.create_window(width=600, height=600)
vis.add_geometry(pcd)
vis.add_geometry(axis)

print(f"Created {NUM_PARTICLES} wave particles and added them to the visualization window.")

# 定义更新函数，用于每隔一段时间更新波浪粒子的位置和图形显示
def update(vis):
    global wave_particles

    # 更新波浪粒子的位置和相位，假设每帧时间间隔为0.1秒
    dt = 0.1

    # 使用numpy的array操作来实现向量化计算，提高运行效率
    wave_particles = update_wave_particles(wave_particles, dt) # 使用自定义函数直接对wave_particles数组进行更新

    # 获取波浪粒子的位置列表和颜色列表（深蓝色）
    points = wave_particles[:, :3] # 获取波浪粒子的位置数组（前三列）
    colors = np.array([[0.0,0.0,sigmoid(wave_particle[2])] for wave_particle in wave_particles]) # 根据z值计算sigmoid函数的值作为蓝色分量

    # 更新散点图的位置和颜色
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors) # 更新颜色属性

    # 返回True表示继续更新，返回False表示停止更新
    return True

# 定义一个函数，用于接收一个波浪粒子数组和一个时间间隔作为参数，并更新波浪粒子数组中的每个元素的位置和相位，返回更新后的波浪粒子数组
def update_wave_particles(wave_particles, dt):
    # 将波浪粒子数组转换为pytorch张量，并将其移动到CUDA设备上
    wave_particles = torch.tensor(wave_particles).to("cuda")
    # 更新波浪粒子的位置和相位
    wave_particles[:, 0] += wave_particles[:, 5] * dt  # x方向移动
    wave_particles[:, 1] += wave_particles[:, 6] * dt  # y方向移动
    wave_particles[:, 0] = torch.clamp(wave_particles[:, 0], -200, 200) # x坐标限制在-200到200之间
    wave_particles[:, 1] = torch.clamp(wave_particles[:, 1], -200, 200) # y坐标限制在-200到200之间
    wave_particles[:, 5] *= torch.where(torch.abs(wave_particles[:, 0]) == 200, -1, 1) # x方向速度反向
    wave_particles[:, 6] *= torch.where(torch.abs(wave_particles[:, 1]) == 200, -1, 1) # y方向速度反向
    wave_particles[:, 7] += 2 * pi * \
        torch.sqrt(wave_particles[:, 5]**2 + wave_particles[:, 6]**2) / \
        wave_particles[:, 4] * dt # 相位变化
    wave_particles[:, 2] = wave_particles[:, 3] * torch.sin(wave_particles[:, 7]) # z方向振动
    # 将pytorch张量转换为numpy数组，并返回
    wave_particles = np.array(wave_particles.cpu())
    return wave_particles

# 注册更新函数，实现动态显示效果
vis.register_animation_callback(update)

# 运行程序
vis.run()
