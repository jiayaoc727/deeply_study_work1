# 创建 analysis.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import settings
from models import my_densenet

# 加载模型
model = my_densenet()
model.load_weights(settings.MODEL_PATH)

# 加载测试数据
from data import test_db

# 获取预测结果
y_pred = model.predict(test_db)
y_true = np.concatenate([y for x, y in test_db], axis=0)

# 计算混淆矩阵
cm = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))

# 可视化
plt.figure(figsize=(12, 12))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks(np.arange(len(settings.CLASSES)), settings.CLASSES, rotation=90)
plt.yticks(np.arange(len(settings.CLASSES)), settings.CLASSES)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()