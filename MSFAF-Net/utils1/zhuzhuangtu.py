# 导入绘图模块
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.pyplot import  MultipleLocator
# 构建数据

x = [68.19
,89.63
,88.81
,90.75
,92.51
,89.13
,93.09
,94.11
,94.55
     ]  # 第一组数据

y = [39.11
,84.61
,84.35
,83.65
,86.28
,90.77
,91.41
,92.88
,91.84
     ]
# 第二组数据

label = ['0.1','0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']  # x轴


bar_width = 0.4  # 柱的宽度

# 中文乱码的处理
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 26, }

# 绘图
plt.figure(figsize=(8.5, 7.5), dpi=300, facecolor='w')
plt.bar(np.arange(9), x, label='Data-A', color='lime', alpha=0.8, width=bar_width)
plt.bar(np.arange(9) + bar_width, y, label='Data-B', color='blue', alpha=0.8, width=bar_width)
plt.rcParams.update({'font.size': 15})
plt.grid(visible=True, ls=':')

# 添加轴标签
plt.xlabel('TR', font)
plt.ylabel('F1 Score (%)', font)
# 添加标题
# plt.title('')
# 添加刻度标签
plt.xticks(np.arange(9) + bar_width, label)
# 设置Y轴的刻度范围
plt.ylim((30, 100))



#刻度范围
# ax = plt.gca()
# y_major_locator=MultipleLocator(1)
# ax.yaxis.set_major_locator(y_major_locator)
plt.tick_params(labelsize=22)

# 为每个条形图添加数值标签
for data_x, data_y in enumerate(x):
    #print(data_x)
    plt.text(data_x - 0.3, data_y, '%s' % data_y,fontsize=10)

for data_x1, data_y1 in enumerate(y):
    plt.text(data_x1 + bar_width - 0.2, data_y1, '%s' % data_y1,fontsize=10)

# 显示图例
plt.legend(fontsize=13,loc='upper left')
# 显示图形
#plt.show()
# 保存图像
plt.savefig('D:\\ImageProcess\\10_MBFNet\\experiment-4\\F1-Score.png')
