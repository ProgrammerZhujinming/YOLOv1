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
5.本项目为个人复现论文研究使用，禁止任何商业用途，如需转载，请附带地址：https://github.com/ProgrammerZhujinming/YOLO_V1_GPU  谢谢配合  
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
分别执行上述两段代码，会生成两个event文件，用Tensorboard监控后会发现曲线图全部乱掉，epoch2、epoch3会存在两份数据，而不会用新的覆盖旧的，而是会根据生成的时间强行拼接，因此才导致了错乱的问题 
解决办法：SummaryWriter对event的保存与pth的保存同步就行了，当重新训练时，删除那些不与pth同步的event即可~  
2.增加了读取视频、摄像头、图片进行检测输出的功能

# 更新日志 2-24
1.鉴于本人的实验环境机器存在过热关机的情况，模型保存方式已经修改为100个epoch保存一次，如各位不存在该问题，请忽略  

# 更新日志 3-14
1.将类别预测的输出通过softmax函数作用后变为更加可靠的可能性大小输出  
2.修复图片、摄像头、视频检测代码中的bug  

# 更新日志 4-25  
1.修复了在迭代后期，avg_iou超过1的Bug,事故代码为interSection = (CrossRX - CrossLX + 1) * (CrossDY - CrossUY + 1)，正确写法应为interSection = (CrossRX - CrossLX) * (CrossDY - CrossUY)，当时也不知道脑子抽的什么风=-=  

# 更新日志 5-6
1.在PyTorch1.6版本之后，对于网络参数的保存方式采用了ZIP格式压缩，因此在切换到较低版本的PyTorch环境中使用之前，需要将模型参数的保存方式进行修改，因此增加了一个可以用于格式转换的transform_pth_format.py  

# 日志 5-12
1.在当前版本，对于每个grid cell输出的bounding_box，认为两个box中置信度高的那个做了针对物体的预测，在计算置信度损失时，没有根据阈值判定box预测的是背景还是物体，而是统一做误差，这可能会导致在置信度这一块训练的困难，正考虑加入合适的阈值进行软化,小于0.3即代表预测为背景，大于0.8代表预测为物体，若预测正确则不加入损失计算，重点在于对预测置信度处于0.3-0.8的难例进行挖掘  
2.考虑应针对每一个类别单独运用NMS，否则产生覆盖的不同类别的显示会出问题，如书本上有杯子，若不以类别区分进行NMS，最终有可能只能显示一个物体    

# 更新日志 5-21
1.给LossFunction增加setWeight函数，用于在训练到达一定量的时候进行权重的调整,让网络不再只侧重定位,提高置信度和分类误差的相对权重。

# 更新日志 5-27
1.采用tqdm将batch的训练情况由print输出更改为进度条输出。

# 更新日志 5-29
1.对DataSet加入shuffle函数用以打乱数据集，并加入10折交叉验证，用以协助手动调整超参数。（目前仅针对YOLO_V1_Train_From_Record.py做了更变，YOLO_V1_Train.py不变）  
2.目前项目输出的各种标准值 大多都是 mean-batch的，正考虑下一步方案。  
3.将模型中 使用view拉平向量的方式 更改为 更加简洁的 faltten。  
4.由于git lfs免费使用只能上传不超过1G的文件，而权重文件已经达到了 1.01G，因此本人无法在此提供权重文件，如确有需要，可通过csdn私信我获取，谢谢。  
 
# 更新日志 7-6  
1.重构了部分代码，减少了显存占用  
2.使用余弦退火算法进行学习率的动态调整  
3.初步撰写了mAP计算脚本  
4.将BN层调整在ReLU层后面，有实验证明，对于类sigmod型的函数，先BN再激活比较好，会有将激活函数的输入与输出拉到梯度不饱和区间的位置的作用，可以加速训练；放置在ReLU后面是因为ReLU不存在梯度饱和问题，并且BN层接在下一层之前，效果相当于将下一层的输入做归一化，正如我们在训练之前对数据集中的数据进行归一化那样，会有更好的效果。  
5.与身边朋友沟通后认为，作者在文中指出的confidence=Pr(obj)*IoU，指的是让网络对于置信度的回归目标变为当前的bounding box与 ground_truth的IoU。
