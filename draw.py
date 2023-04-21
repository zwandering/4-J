import matplotlib.pyplot as plt
import numpy as np

# 数据
x = []
for i in range(15):
    x.append(f'{i}')
x.append('mean')

x_1 = np.array(list(range(16)))
svm = [0.378043912,0.242714571,
       0.306986028,
       0.53493014,
       0.326147705,
       0.312974052,
       0.41996008,
       0.286227545,
       0.421157685,
       0.435129741,
       0.479840319,
       0.29740519,
       0.308982036,
       0.28742515,
       0.311377246,

0.356620093
       ]
svm_std = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0732]
mlp = [0.553027279,
0.444311377,
0.520159681,
0.786427146,
0.445908184,
0.529740519,
0.641117764,
0.428343313,
0.689021956,
0.55489022 ,
0.662275449,
0.651497006,
0.566067864,
0.499401198,
0.368063872,
0.508183633,

]
mlp_std = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1101]
resnet = [
55.3027279, 50.21956087824351, 47.82435129740519, 51.69660678642715, 43.91217564870259, 43.952095808383234, 67.54491017964072, 45.02994011976048, 48.46307385229541, 47.10578842315369, 59.4810379241517 ,45.74850299401198 ,45.94810379241517, 66.46706586826348, 44.03193612774451 ,46.187624750499005,
]
for i in range(16):
       resnet[i] /= float(100)
resnet_std = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1023]
irm = [0.557312043 ,
0.437125749 ,
0.513373253 ,
0.769261477 ,
0.40758483  ,
0.554091816 ,
0.680638723 ,
0.463473054 ,
0.65988024  ,
0.56007984  ,
0.727345309 ,
0.717365269 ,
0.504590818 ,
0.485828343 ,
0.38003992  ,
0.499001996 ,


]
irm_std = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1199]
tca = [ 0.5114067232532068,0.40402392, 0.50373598, 0.77070047, 0.28310024, 0.49644045, 0.56367148,
 0.49547675, 0.51852277, 0.51949454, 0.68790042, 0.61080965, 0.49414018,
 0.54292694, 0.39354661, 0.38661044,]
tca_std =[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1078]
dann = [0.625256154 ,
0.584031936 ,
0.569261477 ,
0.805189621 ,
0.516966068 ,
0.65508982  ,
0.723353293 ,
0.522954092 ,
0.70499002  ,
0.660279441 ,
0.648303393 ,
0.68742515  ,
0.541716567 ,
0.620359281 ,
0.565269461 ,
0.573652695 ,


]
dann_std=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0796]
adda =[0.561429398 ,
0.435763889 ,
0.555555556 ,
0.805555556 ,
0.381076389 ,
0.549913194 ,
0.596354167 ,
0.568576389 ,
0.611979167 ,
0.569010417 ,
0.706597222 ,
0.621961806 ,
0.508246528 ,
0.573350694 ,
0.466579861 ,
0.470920139 ,


]
adda_std =[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1022]
prpl=[0.649753826 ,
0.552694611 ,
0.6500998   ,
0.742714571 ,
0.557884232 ,
0.570658683 ,
0.778642715 ,
0.588622754 ,
0.760279441 ,
0.591417166 ,
0.684431138 ,
0.723552894 ,
0.648902196 ,
0.541516966 ,
0.597005988 ,
0.757884232 ,

]
prpl_std = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0823]
plt.figure(figsize=(30, 8))
total_width, n = 0.8, 8
width = total_width / n
width1 = width-0.02
# x = np.array(x) - (total_width - width) / 2
# 绘制柱状图
svm.reverse()
mlp.reverse()
resnet.reverse()
irm.reverse()
tca.reverse()
dann.reverse()
adda.reverse()
prpl.reverse()


plt.bar(x_1+0*width, svm, width=width1, label=f"SVM", yerr=svm_std, align='center', alpha=1., ecolor='black', capsize=4, error_kw={'elinewidth': 2},tick_label = x)
plt.bar(x_1+1*width, mlp, width=width1, label=f"MLP", yerr=mlp_std, align='center', alpha=1., ecolor='black', capsize=4, error_kw={'elinewidth': 2},tick_label = x)
plt.bar(x_1+2*width, resnet, width=width1, label=f"ResNet", yerr=resnet_std, align='center', alpha=1., ecolor='black', capsize=4, error_kw={'elinewidth': 2},tick_label = x)
plt.bar(x_1+3*width, irm, width=width1, label=f"IRM", yerr=irm_std, align='center', alpha=1., ecolor='black', capsize=4, error_kw={'elinewidth': 2},tick_label = x)
plt.bar(x_1+4*width, tca, width=width1, label=f"TCA", yerr=tca_std, align='center', alpha=1., ecolor='black', capsize=4, error_kw={'elinewidth': 2},tick_label = x)
plt.bar(x_1+5*width, dann, width=width1, label=f"DANN", yerr=dann_std, align='center', alpha=1., ecolor='black', capsize=4, error_kw={'elinewidth': 2},)
plt.bar(x_1+6*width, adda, width=width1, label=f"ADDA", yerr=adda_std, align='center', alpha=1., ecolor='black', capsize=4, error_kw={'elinewidth': 2},)
plt.bar(x_1+7*width, prpl, width=width1, label=f"PR-PL", yerr=prpl_std, align='center', alpha=1., ecolor='black', capsize=4, error_kw={'elinewidth': 2},)
# 添加标题和标签
# plt.title('')
plt.legend(loc='upper center', prop={'size': 18}, ncol=8, bbox_to_anchor=(0.5, 1.10))

plt.xlabel('Subject Index', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)

# 显示图形
plt.savefig('acc.png')
