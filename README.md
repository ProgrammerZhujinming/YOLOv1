# YOLO_V1_GPU  
Tensorboard功能有待完善、目前迭代正常  
环境要求：  
1.PyTorch >= 1.1.0  
2.tensorboardX包  
3.cuda >= 10.1  
项目说明：  
1.tensorboard功能需要PyTorch版本在1.1.0及以上，在项目目录下执行指令tensorboard --logdir=log即可启动  
2.相应需要查看自己的cuda版本是否支持对应的PyTorch版本  
3.本项目默认使用GPU运行，如需使用CPU，请将cuda()全部去除  
4.频繁读取图片、垃圾回收的GC算法运行不够及时，可能会导致程序因为内存不足退出，可以手动管理，将img_data转为tensor后，将原来在内存中的数据del掉，再使用gc.collect()回收，内存够用可以无视  
5.本项目为个人复现论文研究使用，禁止用在其他任何场合，谢谢配合  

更新日志 2-18  
1.增加各种loss的tensorboard曲线图监控  
2.增加特征图输出功能--由于本人选择在每一个batch后都输出一次特征图，因此该功能严重影响训练速度，所以在代码中做了注释处理，如有需要请自行开启：# feature_map_visualize(batch_train[0], writer)，取消注释即可。当然如果想要看到特征图又怕影响训练速度，可以选择将特征图的显示放在每一个epoch而不是batch中。  
3.使用特征图功能可能会遇到TypeError: clamp_(): argument ‘min’ must be Number, not Tensor，此时需要修改torchvison.utils源码，将norm_ip(t, t.min(), t.max())改为norm_ip(t, float(t.min()), float(t.max()))  
4.因本项目初期为避免精度误差而将坐标误差采用回归到原图像尺度上的方式进行计算Loss，最终未回归至0-1区间，导致了坐标误差与其他误差的差距过大。当前版本loss的坐标损失已经归一化到0-1区间  

更新日志 2-20
1.设置学习率多步长衰减策略  
2.为网络模型的层设置初始化  
3.增加每一个epoch后输出梯度直方图的功能，用于检测网络迭代情况  
4.重构了部分loss_function代码  

更新日志 2-21  
1.修复loss的错误反传(在计算loss时，如果使用数学函数，需要将math换成torch，否则不支持反传，同时torch.sqrt的导数在0处无定义，需要加上一个偏置值1e-8，避免出现0处的导数nan的问题)和显存占用过大的问题(计算loss时生成了过多的中间节点导致)  
2.增加了一个全卷积的YOLO v1结构，用来避免训练过程中由于reshape导致特征图错乱的问题  
