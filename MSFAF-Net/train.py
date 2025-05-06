from utils1.dataset import ISBI_Loader
from torch import optim
import torchvision.transforms as Transforms
import torch.utils.data as data
import time
import torch
import torch.nn as nn
import logging
import datetime
from model1.models2_rcb import RcbUnet
ModelName = 'wxcd_MSFAFNet'


def train_net(net, device, data_path, epochs=150, lr=0.0001,batch_size=8, is_Transfer=False):
    print('Currently, Training Model is :::::' + ModelName + ':::::')
    # 定义优化器
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                      milestones=[10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90],
                                                      gamma=0.9)
    # 定义loss
    criterion = nn.BCEWithLogitsLoss()

    # 输出损失函数 和 时间
    f_loss = open('train_wxcd_MSFAFNet_loss.txt', 'w')
    f_time = open('train_wxcd_MSFAFNet_time.txt', 'w')
    startime = datetime.datetime.now().strftime('%m-%d')
    log_dir = 'logger/' + startime + '-1'
    # 训练epochs次
    best_loss = float('inf')
    epochbest = 1

    batch_sizes = [2, 4, 8, 16, 32]  # 定义批量大小列表

    for epoch in range(1, epochs + 1):
        # 循环选择batch size
        #batch_size = batch_sizes[(epoch - 1) % len(batch_sizes)]
        #随机选择
        # batch_size = random.choice([2,4,8,16,32])

        #批次改变batchsize
        # if epoch % 20 == 0:
        #     batch_size *= 2
        # 每 10 次迭代改变一次batch size
        #batch_size = batch_sizes[(epoch - 1) // 10 % len(batch_sizes)]


        isbi_dataset = ISBI_Loader(data_path=data_path, transform=Transforms.ToTensor())
        train_loader = data.DataLoader(dataset=isbi_dataset, batch_size=batch_size, shuffle=True)

        print('=============迭代次数 %d, 批量大小 %d=======================' % (epoch, batch_size))
        sums = 0
        logging.info('validation set: %d patches' % epoch)
        # 开始训练
        net.train()
        num = int(0)
        best_mIoU = float(0)

        starttime = time.time()  # 记录时间
        for image1, image2, label in train_loader:
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image1 = image1.to(device=device)
            image2 = image2.to(device=device)
            label = label.to(device=device)
            # 使用网络参数，输出预测结果
            pred1 = net(image1, image2)
            # 计算loss
            loss = criterion(pred1, label)
            total_loss = loss

            print("************", total_loss)
            sums += float(total_loss)

            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(net.state_dict(), 'best' + ModelName + '_model_final.pth')
            # 保存损失
            total_loss.backward()
            optimizer.step()
            num += 1
        # learning rate delay
        f_loss.write(str(float('%5f' % (sums / 20))) + '\n')
        scheduler1.step()
        endtime = time.time()
        # 保存时间
        if epoch == 0:
            f_time.write('each epoch time\n')
        f_time.write(str(epoch) + ',' + str(starttime) + ',' + str(endtime) + ',' + str(
            float('%2f' % (starttime - endtime))) + '\n')

    f_loss.close()
    f_time.close()


if __name__ == '__main__':
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道3，分类1(目标)
    net = RcbUnet(3, 1)

    # 将网络拷贝到device中
    net.to(device=device)
    # 指定训练集地址，开始训练
    data_path = "data/whu/train2"
    train_net(net, device, data_path)