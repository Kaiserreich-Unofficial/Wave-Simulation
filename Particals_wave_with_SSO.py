from math import pi,exp,inf
import random
import numpy as np
import open3d as o3d
import torch
import open3d.core as o3c
import open3d.geometry as o3g

# Sigmoid 函数
def sigmoid(x):
    return 1/(1+exp(-x))

# 创建一个CUDA设备对象
device = o3c.Device("cuda:0")

# 波浪粒子的数量
NUM_PARTICLES = 10000

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

# 定义SSO算法类
class SSO:
    def __init__(self, num_agents, num_iterations, num_inliers, wave_particles):
        self.num_agents = num_agents  # SSO算法中搜索点（代理）的数量
        self.num_iterations = num_iterations  # SSO算法中迭代次数
        self.num_inliers = num_inliers  # 每个时刻波浪最上层表面的粒子数量
        self.wave_particles = wave_particles  # 波浪粒子数组，每一行包含8个元素，分别表示x坐标、y坐标、z坐标、振幅、波长、x方向速度、y方向速度和相位
        self.agents = []  # 搜索点（代理）列表，每个元素是一个二元组，包含一个索引和一个适应度值
        self.best_agent = None  # 最佳搜索点（代理）的索引
        self.best_fitness = -inf  # 最佳适应度值
        self.inliers = []  # 内点集，每个元素是一个索引

        # 将波浪粒子数组转换为pytorch张量，并将其移动到CUDA设备上
        self.wave_particles_tensor = torch.tensor(self.wave_particles).to("cuda")

    # 初始化搜索点（代理）列表
    def initialize_agents(self):
        for i in range(self.num_agents):
            # 随机选择一个波浪粒子的索引作为搜索点（代理）
            index = random.randint(0, len(self.wave_particles) - 1)
            # 计算搜索点（代理）的适应度值
            fitness = self.calculate_fitness(index)
            # 将搜索点（代理）的索引和适应度值添加到列表中
            self.agents.append((index, fitness))
            # 如果当前搜索点（代理）的适应度值大于最佳适应度值，则更新最佳搜索点（代理）的索引和最佳适应度值
            if fitness > self.best_fitness:
                self.best_agent = index
                self.best_fitness = fitness

    # 计算搜索点（代理）的适应度值
    def calculate_fitness(self, index):
        # 假设适应度函数为：f(agent) = z**3 * amplitude，这个函数考虑了波浪粒子的z值和振幅等属性，使得表面的粒子具有较高的适应度值
        fitness = self.wave_particles[index][2]**3 * \
            self.wave_particles[index][3]
        return fitness

    # 更新搜索点（代理）的位置和适应度值
    def update_agents(self):
        for i in range(self.num_agents):
            # 获取当前搜索点（代理）的索引和适应度值
            index, fitness = self.agents[i]
            # 随机选择一个方向向量，使其与最佳搜索点（代理）的位置向量垂直
            direction = torch.randn(3).to("cuda",dtype=torch.double)
            direction = direction - torch.dot(direction, torch.tensor(
                [self.wave_particles[self.best_agent][0], self.wave_particles[self.best_agent][1], self.wave_particles[self.best_agent][2]]).to("cuda")) * torch.tensor([self.wave_particles[self.best_agent][0], self.wave_particles[self.best_agent][1], self.wave_particles[self.best_agent][2]]).to("cuda")
            direction = direction / torch.norm(direction)
            # 随机选择一个步长，使其在0到最佳搜索点（代理）的振幅之间
            step = random.uniform(0, self.wave_particles[self.best_agent][3])
            # 沿着方向向量移动步长，得到新的位置
            new_position = torch.tensor([self.wave_particles[index][0], self.wave_particles[index][1], self.wave_particles[index][2]]).to(
                "cuda") + step * direction
            # 根据新的位置，找到最近的波浪粒子作为新的搜索点（代理）
            nearest_particle = None

            # 计算新位置与所有波浪粒子的距离，并找到最小距离对应的索引
            distances = torch.norm(
                self.wave_particles_tensor[:, :3] - new_position, dim=1)
            min_index = torch.argmin(distances).item()

            # 计算新的搜索点（代理）的适应度值
            new_fitness = self.calculate_fitness(min_index)
            # 如果新的适应度值大于原来的适应度值，则更新搜索点（代理）的索引和适应度值
            if new_fitness > fitness:
                self.agents[i] = (min_index, new_fitness)
                # 如果新的适应度值大于最佳适应度值，则更新最佳搜索点（代理）的索引和最佳适应度值
                if new_fitness > self.best_fitness:
                    self.best_agent = min_index
                    self.best_fitness = new_fitness

    # 寻找每个时刻波浪最上层表面的粒子并加入内点集
    def find_inliers(self):
        # 对搜索点（代理）列表按照适应度值降序排序
        # 先将agents列表转换为pytorch张量，并将其移动到CUDA设备上
        agents_tensor = torch.stack([torch.tensor([index, fitness]) for index, fitness in self.agents]).to("cuda")
        # 然后使用torch.sort()方法对张量的第二列（适应度值）进行降序排序，并返回排序后的索引
        sorted_indices = torch.sort(
            agents_tensor[:, 1], descending=True).indices
        # 最后根据索引重新排列agents列表
        self.agents = [self.agents[i.item()] for i in sorted_indices]
        # 取前num_inliers个搜索点（代理）作为内点集
        self.inliers = [index for index,
                        fitness in self.agents[:self.num_inliers]]

    # 执行SSO算法
    def run(self):
        # 初始化搜索点（代理）列表
        self.initialize_agents()
        # 迭代更新搜索点（代理）的位置和适应度值
        for i in range(self.num_iterations):
            print(f"SSO iteration {i+1}")
            self.update_agents()
        # 寻找每个时刻波浪最上层表面的粒子并加入内点集
        self.find_inliers()
        print(f"Found {len(self.inliers)} inliers")

# 创建一个SSO算法对象，设置搜索点（代理）数量为200，迭代次数为2，内点集数量为1000，波浪粒子列表为wave_particles
sso = SSO(200, 2, 3000, wave_particles)

# 创建一个散点图对象，用于表示波浪粒子，设置点的大小为10，颜色为深蓝色（透明度随z值变化）
points = wave_particles[:, :3] # 获取波浪粒子的位置数组（前三列）
colors = np.array([[0.0,0.0,sigmoid(wave_particle[2])] for wave_particle in wave_particles]) # 设置颜色为浅蓝到深蓝的渐变，根据z值计算sigmoid函数的值作为蓝色分量

pcd = o3g.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)  # 添加颜色属性

# 创建一个XYZ轴对象，用于显示三维坐标轴，设置轴的长度和颜色
axis = o3g.TriangleMesh.create_coordinate_frame(
    size=200, origin=[0, 0, 0])

# 将散点图和XYZ轴添加到可视化窗口中
vis = o3d.visualization.Visualizer()
vis.create_window(width=800, height=600)
vis.add_geometry(pcd)
vis.add_geometry(axis)

print(
    f"Created {NUM_PARTICLES} wave particles and added them to the visualization window.")

# 添加包络线的空占位
hull_ls = o3g.LineSet()
vis.add_geometry(hull_ls)

# 定义更新函数，用于每隔一段时间更新波浪粒子的位置和图形显示
def update(vis):
    global wave_particles, hull_ls

    # 更新波浪粒子的位置和相位，假设每帧时间间隔为0.1秒
    dt = 0.1
    # 使用numpy的array操作来实现向量化计算，提高运行效率
    wave_particles = update_wave_particles(wave_particles, dt) # 使用自定义函数直接对wave_particles数组进行更新

    # 获取波浪粒子的位置列表和颜色列表（深蓝色）
    points = wave_particles[:, :3] # 获取波浪粒子的位置数组（前三列）
    colors = np.array([[0.0,0.0,sigmoid(wave_particle[2])] for wave_particle in wave_particles]) # 根据z值计算sigmoid函数的值作为蓝色分量

    # 更新散点图的位置和颜色
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)  # 更新颜色属性

    # 执行SSO算法，寻找每个时刻波浪最上层表面的粒子并加入内点集
    sso.run()

    # 获取内点集的位置列表和颜色列表（红色）
    inlier_points = wave_particles[sso.inliers, :3] # 获取内点集中的粒子位置数组（前三列）

    inlier_colors = np.array([[1.0, 0.0, 0.0]
                             for _ in sso.inliers])  # 设置颜色为红色

    map_to_tensor = {}
    map_to_tensor["positions"] = o3c.Tensor(inlier_points,device=device)
    map_to_tensor["colors"] = o3c.Tensor(inlier_colors,device=device)

    cuda_inlier_pcd = o3d.t.geometry.PointCloud(map_to_tensor)

    cuda_inlier_pcd.estimate_normals(50, 10)  # 法线估计

    # 计算波浪表面粒子的包络线
    hull = cuda_inlier_pcd.compute_convex_hull()
    hull = o3g.LineSet.create_from_triangle_mesh(hull.to_legacy())
    hull_ls.points = hull.points
    hull_ls.lines = hull.lines
    hull_ls.paint_uniform_color([1, 0, 0])

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
