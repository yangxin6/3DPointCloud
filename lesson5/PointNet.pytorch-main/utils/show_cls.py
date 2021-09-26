from __future__ import print_function
import argparse
import torch.utils.data
from torch.autograd import Variable
from pointnet.dataset import ShapeNetDataset, ModelNetDataset
from pointnet.model import PointNetCls
import torch.nn.functional as F
# 可视化
from torch.utils.tensorboard import SummaryWriter
import os

#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))
parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, required=True,  help='model path') #加载模型
parser.add_argument('--batchSize', type=int, default=32, help='input batch size') #终端键入batchsize
parser.add_argument('--num_points', type=int, default=2500, help='input batch size') #默认的每个点云点数2500
parser.add_argument('--dataset', type=str, required=True, help="dataset path") #数据集路径
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40") #数据集类型shapenet或者modelnet40
opt = parser.parse_args()
print(opt)

if opt.dataset_type == 'shapenet':
    test_dataset = ShapeNetDataset( #测试集为ShapeNetDataset
        root=opt.dataset,
        split='test',
        classification=True,
        npoints=opt.num_points,
        data_augmentation=False)
elif opt.dataset_type == 'modelnet40':
    test_dataset = ModelNetDataset(
        root=opt.dataset,
        split='modelnet40_test',
        npoints=opt.num_points,
        data_augmentation=False)

testdataloader = torch.utils.data.DataLoader( #加载测试集数据
    test_dataset, batch_size=opt.batchSize, shuffle=True)

classifier = PointNetCls(k=len(test_dataset.classes))
classifier.cuda()
classifier.load_state_dict(torch.load(opt.model))
classifier.eval()
log_dir = os.path.join(opt.dataset_type, 'tensorboard', 'classification', 'show_test')
test_writer = SummaryWriter(log_dir=log_dir)

for i, data in enumerate(testdataloader, 0):
    points, target = data
    points, target = Variable(points), Variable(target[:, 0])
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    pred, _, _ = classifier(points)
    loss = F.nll_loss(pred, target)

    pred_choice = pred.data.max(1)[1]
    print(pred_choice) #预测  tensor([ 4, 12, 15,  4, 15, 15, 12,  6], device='cuda:0')
    print(target) #真值 tensor([ 4, 15, 15,  4, 15, 15, 15,  8], device='cuda:0')
    correct = pred_choice.eq(target.data).cpu().sum() #计算正确率 
    print('i:%d  loss: %f accuracy: %f' % (i, loss.data.item(), correct / float(8))) #显存小，修改batch_size为8或者4  ，i:0  loss: 0.426616 accuracy: 0.750000
    # 绘制
    test_writer.add_scalar('Accuracy/test', correct.item() / float(opt.batchSize), i)
    test_writer.add_scalar('Loss/test', loss.data.item(), i)

test_writer.close()