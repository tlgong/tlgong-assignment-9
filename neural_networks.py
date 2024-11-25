import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle
from matplotlib import cm

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# 定义一个简单的 MLP 类
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # 学习率

        # 初始化权重和偏置
        self.W1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim)
        self.b2 = np.zeros((1, output_dim))

        # 定义激活函数及其导数
        if activation == 'tanh':
            self.activation = np.tanh
            self.activation_deriv = lambda x: 1 - np.tanh(x) ** 2
        elif activation == 'relu':
            self.activation = lambda x: np.maximum(0, x)
            self.activation_deriv = lambda x: (x > 0).astype(float)
        elif activation == 'sigmoid':
            self.activation = lambda x: 1 / (1 + np.exp(-x))
            self.activation_deriv = lambda x: self.activation(x) * (1 - self.activation(x))
        else:
            raise ValueError("Unsupported activation function.")
        self.output_activation = lambda x: np.tanh(x)
        self.output_activation_deriv = lambda x: 1 - np.tanh(x) ** 2

    def forward(self, X):
        # 前向传播
        self.X = X  # 存储输入以供后向传播使用
        self.Z1 = X.dot(self.W1) + self.b1
        self.A1 = self.activation(self.Z1)  # 隐藏层激活值
        self.Z2 = self.A1.dot(self.W2) + self.b2
        self.A2 = self.output_activation(self.Z2)  # 输出层激活值
        return self.A2

    def backward(self, X, y):
        # 使用链式法则计算梯度
        m = y.shape[0]
        dA2 = (self.A2 - y) * self.output_activation_deriv(self.Z2)  # 损失对 A2 的导数
        dW2 = self.A1.T.dot(dA2) / m
        db2 = np.sum(dA2, axis=0, keepdims=True) / m

        dA1 = dA2.dot(self.W2.T) * self.activation_deriv(self.Z1)
        dW1 = X.T.dot(dA1) / m
        db1 = np.sum(dA1, axis=0, keepdims=True) / m

        # 使用梯度下降更新权重
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def get_hidden_features(self):
        return self.A1

def generate_data(n_samples=100):
    np.random.seed(0)
    # 生成输入数据
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # 圆形边界
    y = y.reshape(-1, 1)
    return X, y

# 可视化更新函数
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # 执行训练步骤
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)

    # 绘制隐藏层特征
    hidden_features = mlp.get_hidden_features()
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2],
                      c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_hidden.set_title('Hidden Layer Features')
    ax_hidden.set_xlabel('Neuron 1')
    ax_hidden.set_ylabel('Neuron 2')
    ax_hidden.set_zlabel('Neuron 3')

    # 在隐藏层空间中添加决策超平面
    x_range = np.linspace(hidden_features[:, 0].min(), hidden_features[:, 0].max(), 10)
    y_range = np.linspace(hidden_features[:, 1].min(), hidden_features[:, 1].max(), 10)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    if mlp.W2[2, 0] != 0:
        Z_grid = (-mlp.W2[0, 0] * X_grid - mlp.W2[1, 0] * Y_grid - mlp.b2[0, 0]) / mlp.W2[2, 0]
        ax_hidden.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.3, color='green')

    # 绘制输入空间决策边界
    xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = mlp.forward(grid)
    Z = preds.reshape(xx.shape)
    ax_input.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.3, cmap='bwr')
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolors='k')
    ax_input.set_title('Input Space Decision Boundary')

    # 可视化特征和权值，使用颜色深浅表示权值大小
    # 神经元的位置用于可视化
    layer_sizes = [mlp.W1.shape[0], mlp.W1.shape[1], mlp.W2.shape[1]]
    neuron_positions = []
    x_positions = [0, 1, 2]
    for i, layer_size in enumerate(layer_sizes):
        y_positions = np.linspace(0.1, 0.9, layer_size)
        x = np.full(layer_size, x_positions[i])
        positions = np.column_stack([x, y_positions])
        neuron_positions.append(positions)

    # 绘制神经元
    for positions in neuron_positions:
        ax_gradient.scatter(positions[:, 0], positions[:, 1], s=500, facecolors='white', edgecolors='black', zorder=3)

    # 获取权值的最大绝对值，用于缩放颜色强度
    max_weight = max(np.max(np.abs(mlp.W1)), np.max(np.abs(mlp.W2)))

    # 使用灰度颜色映射表示权值大小
    cmap = cm.get_cmap('gray')

    # 从输入层到隐藏层的权值
    for i, (x0, y0) in enumerate(neuron_positions[0]):
        for j, (x1, y1) in enumerate(neuron_positions[1]):
            weight = mlp.W1[i, j]
            weight_norm = np.abs(weight) / max_weight  # 归一化权值大小
            color_intensity = cmap(1 - weight_norm)  # 映射到颜色（权值越大颜色越深）
            ax_gradient.plot([x0, x1], [y0, y1], color=color_intensity, linewidth=2)

    # 从隐藏层到输出层的权值
    for i, (x0, y0) in enumerate(neuron_positions[1]):
        for j, (x1, y1) in enumerate(neuron_positions[2]):
            weight = mlp.W2[i, j]
            weight_norm = np.abs(weight) / max_weight  # 归一化权值大小
            color_intensity = cmap(1 - weight_norm)  # 映射到颜色（权值越大颜色越深）
            ax_gradient.plot([x0, x1], [y0, y1], color=color_intensity, linewidth=2)

    ax_gradient.set_xlim(-0.5, 2.5)
    ax_gradient.set_ylim(0, 1)
    ax_gradient.axis('off')
    ax_gradient.set_title('Network Weights')

    # 添加当前迭代次数的文本标注
    iter_num = (frame + 1) * 10
    ax_hidden.text2D(0.05, 0.95, f"Iteration: {iter_num}", transform=ax_hidden.transAxes)
    ax_input.text(0.05, 0.95, f"Iteration: {iter_num}", transform=ax_input.transAxes)
    ax_gradient.text(0.05, 0.95, f"Iteration: {iter_num}", transform=ax_gradient.transAxes)

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # 设置可视化
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # 创建动画
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden,
                                     ax_gradient=ax_gradient, X=X, y=y),
                        frames=step_num // 10, repeat=False)

    # 将动画保存为 GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "sigmoid"  # 可选 'tanh', 'relu', 'sigmoid'
    lr = 0.1
    step_num = 2000
    visualize(activation, lr, step_num)
