import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# 读取数据集
iris = pd.read_csv("../data/iris.data", header=0)

# 数据集分类，分训练集和测试集
iris_data = iris.values[0:150, 0:6]
iris_data = np.array(iris_data[0:150, 0:6])

# 使用 LabelEncoder 将标签转换为数值
label_encoder = LabelEncoder()
iris_labels = iris_data[:, 4]  # 原始标签列（例如 'Iris-setosa', 'Iris-versicolor' 等）
iris_labels = label_encoder.fit_transform(iris_labels)  # 将标签转换为数字

# 训练集
iris_train_data = iris_data[range(0, 30), 0:6]
iris_train_data = np.vstack((iris_train_data, iris_data[range(50, 80), 0:6]))
iris_train_data = np.vstack((iris_train_data, iris_data[range(100, 130), 0:6]))
iris_train_data = np.array(iris_train_data)
iris_train_label = iris_labels[range(0, 30)]  # 确保标签与训练数据一致

# 这里是切割训练集和标签的对应部分
iris_train_label = np.hstack((iris_labels[range(0, 30)], iris_labels[range(50, 80)]))
iris_train_label = np.hstack((iris_train_label, iris_labels[range(100, 130)]))

iris_train_data = iris_train_data[:, 0:4]  # 使用前4列作为特征
iris_train_data = iris_train_data.astype('float64')
iris_train_label = iris_train_label.astype('float64')

print(iris_train_data.shape)
print(iris_train_label.shape)

# 测试集
iris_test_data = iris_data[range(30, 50), 0:6]
iris_test_data = np.vstack((iris_test_data, iris_data[range(80, 100), 0:6]))
iris_test_data = np.vstack((iris_test_data, iris_data[range(130, 149), 0:6]))
iris_test_data = np.array(iris_test_data)

iris_test_label = iris_labels[range(30, 50)]  # 测试集的标签
iris_test_label = np.hstack((iris_labels[range(30, 50)], iris_labels[range(80, 100)]))
iris_test_label = np.hstack((iris_test_label, iris_labels[range(130, 149)]))

iris_test_data = iris_test_data[:, 0:4]  # 使用前4列作为特征
iris_test_data = iris_test_data.astype('float64')
iris_test_label = iris_test_label.astype('float64')

print(iris_test_data.shape)
print(iris_test_label.shape)

# 训练模型并评估
b = []
a = []
for num in range(1, 101):
    #Linear
    #clf = svm.SVC(C=num/10, kernel='linear', decision_function_shape='ovr')
    
    #RBF
    #clf = svm.SVC(C=num/10, kernel='rbf', decision_function_shape='ovr')
    
    #poly
    #clf = svm.SVC(C=num/10, kernel='poly', decision_function_shape='ovr')
    
    #sigmoid
    clf = svm.SVC(C=num/10, kernel='sigmoid', decision_function_shape='ovr')
    
    clf.fit(iris_train_data, iris_train_label)

    c = clf.score(iris_train_data, iris_train_label)
    print("训练次数:", num, "训练集准确率:", c)
    a.append(c)  # 交叉验证准确率

    c = clf.score(iris_test_data, iris_test_label)
    print("训练次数:", num, "测试集准确率:", c)
    b.append(c)  # 测试集准确率

# 绘制结果
plt.figure(4)
plt.subplot(1, 2, 1)
plt.plot(range(1, 101), a)
plt.grid()
plt.xlabel('C/10')
plt.ylabel('Train_Set_Accuracy')
plt.subplot(1, 2, 2)
plt.plot(range(1, 101), b)
plt.grid()
plt.xlabel('C/10')
plt.ylabel('Test_Set_Accuracy')

plt.savefig("./result/Iris_sigmoid.png")
plt.show()
