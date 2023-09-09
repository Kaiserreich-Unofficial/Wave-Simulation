# 导入matplotlib和mpl_toolkits模块，用于绘制3D图形
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cupy as cp


class FFT_wave:
    """该类使用快速傅立叶变换（FFT）模拟波的形成和变化。
    参数
    ----------
    L：方形域的长度，默认为 20。
    N：每个维度上的网格点数，默认为 64。
    T：波的周期，默认为 10。
    A：波的振幅，默认为 0.5。
    wind：风速，默认为 10 m/s。
    length：距离，默认为 100 km。
    """
    def __init__(self, L = 20, N = 64, T = 10, A = 0.5, wind = 10, length = 100):
        self.L = L
        self.N = N

        self.gravity = 9.81 # 重力加速度

        x = np.linspace(-L, L, N)
        y = np.linspace(-L, L, N)
        self.X, self.Y = np.meshgrid(x, y)


        kx = (2*cp.pi/(2*L)) * \
            cp.concatenate((cp.arange(0, N/2), cp.arange(-N/2, 0)))
        ky = kx.copy()

        self.KX, self.KY = cp.meshgrid(kx, ky)

        # 定义峰值增强因子和形状参数
        self.gamma = 3.3
        self.sigma = 0.08

        # 根据风速和距离计算峰值波数和与风相关的常数
        self.kp = 0.21 * self.gravity / wind**2
        self.alpha = 0.006 * cp.sqrt(wind * length / self.gravity)

        spectrum = self.jonswap()
        spectrum[cp.isnan(spectrum)] = 0

        self.amplitude = cp.sqrt(
            spectrum) * cp.random.randn(self.N, self.N)
        self.amplitude[0, 0] = 0

    def jonswap(self):
        k_mold = cp.sqrt(self.KX ** 2 + self.KY ** 2)
        k_mold = cp.where(k_mold == 0, 1e-3, k_mold)
        # 使用JONSWAP谱公式
        return self.alpha * self.gravity**2 / k_mold**5 * cp.exp(-5/4 * (self.kp / k_mold)**4) * \
            self.gamma**cp.exp(-(k_mold - self.kp)**2 / (2 * self.sigma**2 * self.kp**2))

    def dispersion(self, kx, ky):
        return cp.sqrt(self.gravity * cp.sqrt(kx**2 + ky**2))

    def calculate(self):
            # 计算每个波数对应的角频率
            omega = self.dispersion(self.KX, self.KY)
            # 计算每个波数对应的相位，使用欧拉公式
            phase = cp.exp(1j * omega * self.t)
            self.heights = np.real(cp.fft.ifft2(self.amplitude * phase).get())
            # 使用choppy wave方法对海浪进行偏移，使得波浪更尖锐
            choppy_wave = 0.5 * cp.sqrt(self.KX**2 + self.KY**2) / cp.sqrt(self.N)
            x_displacement = np.real(
                cp.fft.ifft2(-1j * choppy_wave * self.amplitude * phase * cp.sign(self.KX)).get())

            y_displacement = np.real(
                cp.fft.ifft2(-1j * choppy_wave * self.amplitude * phase * cp.sign(self.KY)).get())

            self.x_offset = self.X + x_displacement

            self.y_offset = self.Y + y_displacement

    def plot(self):
        # 创建一个figure对象，设置大小为10x10
        fig = plt.figure()

        # 创建一个Axes3D对象，用于绘制3D图形
        self.ax = fig.add_subplot(111, projection='3d')

        # 清除之前的图形
        self.ax.clear()
        # 设置坐标轴的范围和标签
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        # self.ax.set_xlim(-self.L, self.L)
        # self.ax.set_ylim(-self.L, self.L)
        # self.ax.set_zlim(0, 5)
        # 设置标题
        self.ax.set_title('FFT Wave Simulation')
        # 创建一个用于动画的函数，设置帧数为100，间隔为50毫秒，重复为True
        ani = animation.FuncAnimation(
            fig, self.draw, frames=60, interval=10, repeat=True)

        # 显示图形
        plt.show()

    def draw(self, frame):
        # 创建一个figure对象，设置大小为10x10
        fig = plt.figure(figsize=(10, 10))

        # 清除之前的图形
        self.ax.clear()
        # 计算当前帧对应的时间，假设每一帧间隔0.1秒
        self.t = frame * 0.1
        # 计算海浪的高度和法向量
        self.calculate()

        # 绘制mesh，设置颜色映射为'jet'，透明度为0.8
        surf = self.ax.plot_surface(self.x_offset, self.y_offset,
                             self.heights, cmap='viridis', alpha=0.8)

        # 添加颜色条
        # fig.colorbar(surf)

        # 设置坐标轴的范围和标签
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_xlim(-self.L, self.L)
        self.ax.set_ylim(-self.L, self.L)
        self.ax.set_zlim(0, 2)
        # 设置标题
        self.ax.set_title('FFT Wave Simulation')

        plt.close()


a = FFT_wave()
a.plot()
