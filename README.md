# FGSM-PointCloud
这是西电2024年专业基础实践我的作业，针对点云设计FGSM对抗性扰动以获得较好的攻击效果
具体来说我们对Pointnet++架构进行了攻击，后续的工作可以按照现有的模板展开。
此外，我们还针对cifar10数据集（2D任务）进行了对抗攻击。
需要感谢我的合作者@NorthSeaFish，他完成本实践的2D部分。
# 针对3D对抗攻击的补充
针对点云的对抗攻击研究相当少，且之前的基于扰动的3D攻击很多时候不是通用对抗扰动
往往是在周边合成点集。我本次的FGSM3D是为了探究通用的扰动能否嫁接到3D对象如点云
上，这也是整个对抗攻击领域第一次直接对3D对象扰动而不是增添点的对抗攻击。实验效果相当出乎意料但是又耐人寻味。
大家可以参考仓库里的报告（比较水）。报告里所有的身份信息出于隐私考虑已经隐去。
