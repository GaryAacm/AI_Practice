import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd

# Load data
sonar = pd.read_csv("../data/sonar.all-data", header=None)

# Extract features and labels
sonar_data = sonar.values[0:208, 0:60]

sonar_labels = sonar.values[0:208, 60]
sonar_labels = np.where(sonar_labels == 'R', 0, 1)

# First class (Rocks) training set: 0 to 60
# Second class (Mines) training set: 97 to 180
sonar_train_data = sonar_data[range(0, 61), 0:61]
sonar_train_data = np.vstack((sonar_train_data, sonar_data[range(97, 180), 0:61]))
sonar_train_label = sonar_labels[range(0, 61)].tolist() + sonar_labels[range(97, 180)].tolist()
sonar_train_data = np.array(sonar_train_data)

# Test set
# First class (Rocks) test set: 61 to 97
# Second class (Mines) test set: 180 to 208
sonar_test_data = sonar_data[range(61, 97), 0:61]
sonar_test_data = np.vstack((sonar_test_data, sonar_data[range(180, 208), 0:61]))
sonar_test_label = sonar_labels[range(61, 97)].tolist() + sonar_labels[range(180, 208)].tolist()
sonar_test_data = np.array(sonar_test_data)

# Check data and labels
print(sonar_train_data)
print(sonar_train_label)

print(sonar_test_data)
print(sonar_test_label)

# SVM training with different C values
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
    
    clf.fit(sonar_train_data, sonar_train_label)

    # Training accuracy
    c = clf.score(sonar_train_data, sonar_train_label)
    print("Training iteration:", num, "Training accuracy:", c)
    a.append(c)   # Append training accuracy

    # Test accuracy
    c = clf.score(sonar_test_data, sonar_test_label)
    print("Training iteration:", num, "Test accuracy:", c)
    b.append(c)  # Append test accuracy

# Plot the results
plt.figure(1)
plt.plot(range(1, 101), a, label='Training Accuracy')
plt.plot(range(1, 101), b, label='Test Accuracy')
plt.grid()
plt.xlabel('C/10')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('./result/sigmoid.png')
plt.show()
