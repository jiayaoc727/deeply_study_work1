# 创建 debug_data.py
import data

print(f"数据目录: {data.settings.IMAGES_ROOT}")
print(f"类别数量: {data.settings.CLASS_NUM}")
print(f"类别列表: {data.settings.CLASSES}")
print(f"训练集样本数: {data.train_samples}")
print(f"验证集样本数: {data.dev_samples}")
print(f"测试集样本数: {data.test_samples}")