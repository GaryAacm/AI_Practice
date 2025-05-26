import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 使用支持中文的字体，如SimHei（黑体），要确保系统中安装了这个字体
plt.rcParams['font.sans-serif'] = ['SimHei']   # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False 

# 1. 加载MNIST数据集
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2. 数据预处理
# 将28x28的图像展开成784维的向量
X_train_flat = X_train.reshape(-1, 28*28)
X_test_flat = X_test.reshape(-1, 28*28)

# 归一化数据到0-1范围
X_train_flat = X_train_flat.astype('float32') / 255
X_test_flat = X_test_flat.astype('float32') / 255

# 3. 展示部分手写数字图像
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
axes = axes.ravel()
for i in range(10):
    axes[i].imshow(X_train[i], cmap='gray')
    axes[i].set_title(f"标签: {y_train[i]}")
    axes[i].axis('off')
plt.tight_layout()
plt.show()

# 4. 初始化KNN分类器
knn = KNeighborsClassifier(n_neighbors=5)

# 5. 训练模型（为了加快计算，只使用部分数据）
X_train_sample = X_train_flat[:5000]
y_train_sample = y_train[:5000]
knn.fit(X_train_sample, y_train_sample)

# 6. 预测测试集（也使用部分数据）
X_test_sample = X_test_flat[:1000]
y_test_sample = y_test[:1000]
y_pred = knn.predict(X_test_sample)

# 7. 计算准确率
accuracy = accuracy_score(y_test_sample, y_pred)
print(f"KNN分类器在MNIST测试集上的准确率为: {accuracy * 100:.2f}%")

# 8. 绘制混淆矩阵
cm = confusion_matrix(y_test_sample, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
disp.plot(cmap=plt.cm.Blues)
plt.title("KNN分类器的混淆矩阵")
plt.show()
