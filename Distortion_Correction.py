import tkinter as tk
from tkinter import filedialog, messagebox
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from numpy.linalg import lstsq
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
from scipy.spatial import cKDTree  # 用于高效邻域查询

plt.ioff()  # 关闭交互模式

# 在全局变量中添加一个引用
canvas_widget = None


def run_code(param_file_path, xyz_file_path, rotation_angle_str, SEL, dx_str, dy_str):
    global canvas_widget
    try:
        rotation_angle = float(rotation_angle_str.strip() or 0.0)
    except ValueError:
        messagebox.showerror("错误", "旋转角度必须为数字")
        return

    try:
        dx = float(dx_str.strip() or 0)
    except ValueError:
        messagebox.showerror("错误", "X方向偏移量必须为数字")
        return

    try:
        dy = float(dy_str.strip() or 0)
    except ValueError:
        messagebox.showerror("错误", "Y方向偏移量必须为数字")
        return

    # 自定义颜色映射
    colors = [(0, 0, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]
    cmap_custom = LinearSegmentedColormap.from_list("custom_cmap", colors)

    # 读取 Excel 文件时，将空格也识别为 NaN
    distortion_params = pd.read_excel(param_file_path, na_values=[' '])
    # 删除所有列均为 NaN 的行，这样可以过滤掉有效数据行后的空白行
    distortion_params.dropna(how='all', inplace=True)

    # 从第三行开始取实际坐标和检测坐标（确保数据格式正确）
    actual_coords = distortion_params.iloc[2:, [0, 1]].dropna().to_numpy(dtype=float)
    detected_coords = distortion_params.iloc[2:, [2, 3]].dropna().to_numpy(dtype=float)

    # 定义旋转矩阵，根据旋转角度将检测值旋转
    theta = np.radians(rotation_angle)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta),  np.cos(theta)]])
    detected_coords = detected_coords @ rotation_matrix

    # 构造设计矩阵和目标向量，用于 SEL==1 和 SEL==2 分支
    if SEL == 1:
        A = np.column_stack([
            detected_coords[:, 0] ** 2,
            detected_coords[:, 1] ** 2,
            detected_coords[:, 0] * detected_coords[:, 1],
            detected_coords[:, 0],
            detected_coords[:, 1],
            np.ones(detected_coords.shape[0])
        ])
    elif SEL == 2:
        A = np.column_stack([
            detected_coords[:, 0] ** 2,
            detected_coords[:, 0],
            np.ones(detected_coords.shape[0])
        ])

    # 对于 SEL==1 和 SEL==2，目标向量分别为实际坐标的X、Y
    if SEL in (1, 2):
        B_x = actual_coords[:, 0]
        B_y = actual_coords[:, 1]

        # 求解系数
        coeff_x, _, _, _ = lstsq(A, B_x, rcond=None)
        coeff_y, _, _, _ = lstsq(A, B_y, rcond=None)

    # 读取XYZ文件（面型数据），其中前14行为头信息
    with open(xyz_file_path, 'r') as file:
        all_lines = file.readlines()
    header_lines = all_lines[:14]
    data_lines = all_lines[14:]

    # 读取数据部分（XYZ值）
    valid_data = []
    for line in data_lines:
        try:
            parts = list(map(float, line.split()))
            if len(parts) >= 3:
                valid_data.append(parts[:3])
        except ValueError:
            continue

    data_xyz = pd.DataFrame(valid_data, columns=['X', 'Y', 'Z'])
    X = data_xyz['X'].to_numpy()
    Y = data_xyz['Y'].to_numpy()
    Z = data_xyz['Z'].to_numpy()

    # 计算中心位置（面型中心）并加上用户输入偏移量
    center_x = np.mean(X) + dx
    center_y = np.mean(Y) + dy

    # 将XYZ数据平移到以中心为原点
    X -= center_x
    Y -= center_y

    # 根据不同的SEL选择不同的校正模型
    if SEL == 1:
        # 二维多项式校正（X和Y均校正）
        def map_coordinates(x, y, coeff):
            return (coeff[0] * x**2 + coeff[1] * y**2 + coeff[2] * x * y +
                    coeff[3] * x + coeff[4] * y + coeff[5])
        corrected_X = map_coordinates(X, Y, coeff_x) + center_x
        corrected_Y = map_coordinates(X, Y, coeff_y) + center_y

    elif SEL == 2:
        # 一维X方向校正
        def map_coordinates(x, coeff):
            return coeff[0] * x**2 + coeff[1] * x + coeff[2]
        corrected_X = map_coordinates(X, coeff_x) + center_x
        corrected_Y = Y + center_y

    elif SEL == 3:
        # 使用 Brown–Conrady 模型：先求径向畸变系数，再求切向畸变系数
        detected_rel = detected_coords
        actual_rel = actual_coords
        x_d = detected_rel[:, 0]
        y_d = detected_rel[:, 1]
        x_u = actual_rel[:, 0]
        y_u = actual_rel[:, 1]
        r_sq = x_d**2 + y_d**2
        r_fourth = r_sq**2

        # 1. 求径向系数：假设切向项为0时，
        # 对于非零 x_d 和 y_d，有： (x_u / x_d) - 1 = K1*r^2 + K2*r^4 和 (y_u / y_d) - 1 = K1*r^2 + K2*r^4
        A_rad = []
        b_rad = []
        thresh = 1e-6
        for i in range(len(x_d)):
            if abs(x_d[i]) > thresh:
                A_rad.append([r_sq[i], r_fourth[i]])
                b_rad.append((x_u[i] / x_d[i]) - 1)
            if abs(y_d[i]) > thresh:
                A_rad.append([r_sq[i], r_fourth[i]])
                b_rad.append((y_u[i] / y_d[i]) - 1)
        A_rad = np.array(A_rad)
        b_rad = np.array(b_rad)
        coeff_rad, _, _, _ = lstsq(A_rad, b_rad, rcond=None)
        K1, K2 = coeff_rad[0], coeff_rad[1]
        print(K1, K2)

        # 计算利用径向校正后的估计值
        scale = 1 + K1 * r_sq + K2 * r_fourth
        x_rad_est = x_d * scale
        y_rad_est = y_d * scale

        # 2. 求切向系数：残差由切向项引起
        # 模型：
        #   x_u - x_rad_est = 2*P1*x_d*y_d + P2*(r_sq + 2*x_d**2)
        #   y_u - y_rad_est = P1*(r_sq + 2*y_d**2) + 2*P2*x_d*y_d
        delta_x = x_u - x_rad_est
        delta_y = y_u - y_rad_est
        A_tan = []
        b_tan = []
        for i in range(len(x_d)):
            A_tan.append([2 * x_d[i] * y_d[i], (r_sq[i] + 2 * x_d[i]**2)])
            b_tan.append(delta_x[i])
            A_tan.append([(r_sq[i] + 2 * y_d[i]**2), 2 * x_d[i] * y_d[i]])
            b_tan.append(delta_y[i])
        A_tan = np.array(A_tan)
        b_tan = np.array(b_tan)
        coeff_tan, _, _, _ = lstsq(A_tan, b_tan, rcond=None)
        P1, P2 = coeff_tan[0], coeff_tan[1]

        # 3. 对 XYZ 数据应用 Brown–Conrady 校正
        # 此时 X, Y 已平移到以中心为原点（不含平移项）
        r_xyz = np.sqrt(X**2 + Y**2)
        r_xyz_sq = X**2 + Y**2
        r_xyz_fourth = r_xyz_sq**2
        scale_xyz = 1 + K1 * r_xyz_sq + K2 * r_xyz_fourth
        # 切向项
        tan_x = 2 * P1 * X * Y + P2 * (r_xyz_sq + 2 * X**2)
        tan_y = P1 * (r_xyz_sq + 2 * Y**2) + 2 * P2 * X * Y
        corrected_X = X * scale_xyz + tan_x + center_x
        corrected_Y = Y * scale_xyz + tan_y + center_y

    # 插值处理
    # 根据校正后的坐标生成连续的整数网格（完整覆盖校正数据范围）
    x_min = int(np.floor(min(corrected_X)))
    x_max = int(np.ceil(max(corrected_X)))
    y_min = int(np.floor(min(corrected_Y)))
    y_max = int(np.ceil(max(corrected_Y)))
    grid_x = np.arange(x_min, x_max + 1)
    grid_y = np.arange(y_min, y_max + 1)
    grid_x_mesh, grid_y_mesh = np.meshgrid(grid_x, grid_y)
    grid_points = np.column_stack((grid_x_mesh.ravel(), grid_y_mesh.ravel()))

    # 利用校正后散乱点对完整网格进行插值（线性法）
    interpolated_Z = griddata(np.column_stack((corrected_X, corrected_Y)), Z, grid_points, method='linear')
    interpolated_Z = interpolated_Z.reshape(grid_x_mesh.shape)

    # 额外判断：若某网格点周围（XY方向距离均小于 threshold）无数据点，则置为 NaN
    threshold = 2.0
    tree = cKDTree(np.column_stack((corrected_X, corrected_Y)))
    dists, _ = tree.query(grid_points, distance_upper_bound=threshold)
    dists = dists.reshape(grid_x_mesh.shape)
    interpolated_Z[dists > threshold] = np.nan

    # 更新头信息第4行：新X/Y起始坐标及网格尺寸
    new_X_min = grid_x[0]
    new_Y_min = grid_y[0]
    new_X_num = len(grid_x)
    new_Y_num = len(grid_y)
    header_lines[3] = f"{new_X_min} {new_Y_min} {new_X_num} {new_Y_num}\n"

    # 修改第九行：只更新前两个数字为显示宽度（若数据宽度超过原显示宽度，则更新为数据宽度+1），其他内容保持不变
    if len(header_lines) >= 9:
        line9 = header_lines[8].rstrip("\n")
        parts = line9.split()
        if len(parts) >= 2:
            try:
                disp_x = int(parts[0])
                disp_y = int(parts[1])
            except ValueError:
                disp_x, disp_y = new_X_num, new_Y_num
            if new_X_num > disp_x:
                disp_x = new_X_num + 1
            if new_Y_num > disp_y:
                disp_y = new_Y_num + 1
            parts[0] = str(disp_x)
            parts[1] = str(disp_y)
            header_lines[8] = " ".join(parts) + "\n"
        else:
            header_lines[8] = f"{new_X_num + 1} {new_Y_num + 1}\n"

    # 生成新的XYZ文件路径
    base_dir = os.path.dirname(xyz_file_path)
    file_name = os.path.basename(xyz_file_path)
    new_file_name = "corrected_" + file_name
    corrected_file_path = os.path.join(base_dir, new_file_name)

    # 写入新的XYZ文件：依照连续网格顺序写入
    with open(corrected_file_path, 'w') as file:
        for line in header_lines:
            file.write(line)
        for j, y in enumerate(grid_y):
            for i, x in enumerate(grid_x):
                z = interpolated_Z[j, i]
                z_str = 'No Data' if np.isnan(z) else f'{z:.10f}'
                file.write(f"{x} {y} {z_str}\n")
        file.write("#\n")

    # 绘图显示：原始数据、校正后散乱点和插值后的完整网格
    fig = Figure(figsize=(16, 6))
    z_min_val, z_max_val = np.percentile(Z, [5, 95])

    ax1 = fig.add_subplot(1, 3, 1)
    sc1 = ax1.scatter(X + center_x, Y + center_y, c=Z, cmap=cmap_custom, s=1, vmin=z_min_val, vmax=z_max_val)
    fig.colorbar(sc1, ax=ax1, label='Z')
    ax1.set_title('Original Data')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.axis('equal')

    ax2 = fig.add_subplot(1, 3, 2)
    sc2 = ax2.scatter(corrected_X, corrected_Y, c=Z, cmap=cmap_custom, s=1, vmin=z_min_val, vmax=z_max_val)
    fig.colorbar(sc2, ax=ax2, label='Z')
    ax2.set_title('Corrected Data')
    ax2.set_xlabel('X_corrected')
    ax2.set_ylabel('Y_corrected')
    ax2.axis('equal')

    ax3 = fig.add_subplot(1, 3, 3)
    sc3 = ax3.scatter(grid_x_mesh, grid_y_mesh, c=interpolated_Z, cmap=cmap_custom, s=1, vmin=z_min_val, vmax=z_max_val)
    fig.colorbar(sc3, ax=ax3, label='Z')
    ax3.set_title('Interpolated Data')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.axis('equal')

    fig.tight_layout(pad=3.0, w_pad=3.0)

    # 如果之前有canvas，则先销毁它
    if canvas_widget is not None:
        canvas_widget.get_tk_widget().destroy()

    canvas_widget = FigureCanvasTkAgg(fig, master=root)
    canvas_widget.draw()
    canvas_widget.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)


# 创建主窗口及界面
root = tk.Tk()
root.title("二次多项式畸变矫正工具")

# 添加顶部单选按钮，现增加第三个选项“回转畸变矫正”
top_frame = tk.Frame(root)
top_frame.pack(side=tk.TOP, padx=10, pady=5)
SEL = tk.IntVar(value=1)  # 默认选择选项1
radio_style = {'font': ('Arial', 10), 'padx': 5}
tk.Radiobutton(top_frame, text="二维畸变", variable=SEL, value=1, **radio_style).pack(side=tk.LEFT)
tk.Radiobutton(top_frame, text="一维X方向畸变", variable=SEL, value=2, **radio_style).pack(side=tk.LEFT)
tk.Radiobutton(top_frame, text="回转畸变矫正", variable=SEL, value=3, **radio_style).pack(side=tk.LEFT)

# 畸变参数文件选择
tk.Label(root, text="选择畸变参数文件(.xlsx):").pack()
param_file_var = tk.StringVar()
param_entry = tk.Entry(root, textvariable=param_file_var, state='readonly', width=50)
param_entry.pack()
tk.Button(root, text="选择文件", command=lambda: param_file_var.set(
    filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")]))).pack()

# 面型数据文件选择
tk.Label(root, text="选择面型数据文件(.xyz):").pack()
xyz_file_var = tk.StringVar()
xyz_entry = tk.Entry(root, textvariable=xyz_file_var, state='readonly', width=50)
xyz_entry.pack()
tk.Button(root, text="选择文件", command=lambda: xyz_file_var.set(
    filedialog.askopenfilename(filetypes=[("XYZ files", "*.xyz")]))).pack()

# 输入旋转角度
tk.Label(root, text="畸变参数检测值旋转角度(角度制,正数为逆时针):").pack()
rotation_entry = tk.Entry(root)
rotation_entry.pack()

# 输入X方向偏移量
tk.Label(root, text="面型中心X方向偏移量:").pack()
dx_entry = tk.Entry(root)
dx_entry.pack()

# 输入Y方向偏移量
tk.Label(root, text="面型中心Y方向偏移量:").pack()
dy_entry = tk.Entry(root)
dy_entry.pack()

# 运行按钮
tk.Button(root, text="运行", command=lambda: run_code(
    param_file_var.get(),
    xyz_file_var.get(),
    rotation_entry.get(),
    SEL.get(),
    dx_entry.get(),
    dy_entry.get()
)).pack()

root.mainloop()
