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
