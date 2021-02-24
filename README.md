# YOLO_V1_GPU  
个人博客YOLO v1算法详解地址：https://blog.csdn.net/qq_39304630/article/details/112394446  
Tensorboard功能有待完善、目前迭代正常  
环境要求：  
1.PyTorch >= 1.1.0  
2.tensorboardX包  
3.cuda >= 10.1  
项目说明：  
1.tensorboard功能需要PyTorch版本在1.1.0及以上，在项目目录下执行指令tensorboard --logdir=log即可启动(如果出现无法找到命令的错误，则可能需要安装tensorflow)  
2.相应需要查看自己的cuda版本是否支持对应的PyTorch版本  
3.本项目默认使用GPU运行，如需使用CPU，请将cuda()全部去除  
4.频繁读取图片、垃圾回收的GC算法运行不够及时，可能会导致程序因为内存不足退出，可以手动管理，将img_data转为tensor后，将原来在内存中的数据del掉，再使用gc.collect()回收，内存够用可以无视  
5.本项目为个人复现论文研究使用，禁止用在其他任何场合，谢谢配合  
使用说明:  
1.本项目默认使用的是YOLO v1原文的网络结构，如需换用全卷积结构，只需要在Train的文件中将网络的文件更换为YOLO V1_Full_Conv即可，训练的入口为YOLO V1_Train.py  
2.FromRecord用于从中断的训练中恢复  
3.默认采用的策略是每一个class训练2000个epoch  
4.项目的权重保存策略为 epoch < 1000 时每100个epoch保存一次，epoch >= 1000 时，每1000个epoch保存一次  

# 更新日志 2-18  
1.增加各种loss的tensorboard曲线图监控  
2.增加特征图输出功能--由于本人选择在每一个batch后都输出一次特征图，因此该功能严重影响训练速度，所以在代码中做了注释处理，如有需要请自行开启：# feature_map_visualize(batch_train[0], writer)，取消注释即可。当然如果想要看到特征图又怕影响训练速度，可以选择将特征图的显示放在每一个epoch而不是batch中。  
3.使用特征图功能可能会遇到TypeError: clamp_(): argument ‘min’ must be Number, not Tensor，此时需要修改torchvison.utils源码，将norm_ip(t, t.min(), t.max())改为norm_ip(t, float(t.min()), float(t.max()))  
4.因本项目初期为避免精度误差而将坐标误差采用回归到原图像尺度上的方式进行计算Loss，最终未回归至0-1区间，导致了坐标loss与其他loss的差距过大，在迭代过程中不能权衡训练方向，当前版本loss的坐标损失已经归一化到0-1区间。  

# 更新日志 2-20
1.设置学习率多步长衰减策略  
2.为网络模型的层设置初始化  
3.增加每一个epoch后输出梯度直方图的功能，用于检测网络迭代情况  
4.重构了部分loss_function代码  

# 更新日志 2-21  
1.修复loss的错误反传(在计算loss时，如果使用数学函数，需要将math换成torch，否则不支持反传，同时torch.sqrt的导数在0处无定义，需要加上一个偏置值1e-8，避免出现0处的导数nan的问题)；为了避免显存占用过大的问题(计算loss时生成了过多的中间节点导致)，对于各项损失，使用数学函数计算，而用于反传的loss不再拆开（拆开会在累加时，由于PyTorch动态图的特点，生成过多的中间节点，导致严重的显存泄露问题），而是采用标量配合数学函数库math进行计算统计  
2.增加了一个全卷积的YOLO v1结构，用来避免训练过程中由于reshape导致特征图错乱的问题，作为对原YOLO V1算法的优化拓展  

# 更新日志 2-23
1.修复了从训练中恢复时，代码的TensorboardX记录错乱的问题  
该问题的原因是，在训练过程中，每一个epoch都进行了记录，但是我们的网络权重文件是100个epoch或者1000个epoch才记录的，若我们在233个epoch停止了训练，此时event里记录了1-233个epoch，而我们恢复训练的时候，是从201开始的，201-233的数据会有两份，因此才会导致数据错乱的问题。  
实验代码：  
ex1.py:  
from tensorboardX import SummaryWriter  
writer = SummaryWriter(logdir='runs')  
writer.add_scalar('Train/Loss_sum', 1, 1)  
writer.add_scalar('Train/Loss_sum', 2, 2)  
writer.add_scalar('Train/Loss_sum', 3, 3)  
writer.close()  

ex2.py:  
writer = SummaryWriter(logdir='runs')  
writer.add_scalar('Train/Loss_sum', 4, 2)  
writer.add_scalar('Train/Loss_sum', 5, 3)  
writer.add_scalar('Train/Loss_sum', 6, 4)  
writer.close()  
分别执行上述两段代码，会生成两个event文件，用Tensorboard监控后会发现曲线图全部乱掉，epoch2、epoch3会存在两份数据，而不会用新的覆盖旧的。  
解决办法：SummaryWriter对event的保存与pth的保存同步就行了，当重新训练时，删除那些不与pth同步的event即可~  
2.增加了读取视频、摄像头、图片进行检测输出的功能

# 更新日志 2-24
1.鉴于本人的实验环境机器存在过热关机的情况，模型保存方式已经修改为100个epoch保存一次，如各位不存在该问题，请忽略  
