import tensorflow as tf
import pickle
import os
from data import train_db, dev_db, test_db  # 确保导入测试数据
import models
import settings
from models import show_loss_acc
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ==================== 配置 ====================
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用某些优化，提高兼容性

# ==================== 创建模型 ====================
print("=== 创建模型 ===")
model = models.my_densenet()
model.summary()

# ==================== 编译模型 ====================
print("\n=== 编译模型 ===")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=settings.LEARNING_RATE),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=['accuracy']
)

# ==================== 回调函数 ====================
print("\n=== 设置回调函数 ===")

# 保存最佳模型
model_check_point = tf.keras.callbacks.ModelCheckpoint(
    filepath=settings.MODEL_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# 早停法（防止过拟合）
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=20,
    restore_best_weights=True,
    mode='max',
    verbose=1
)

# 学习率调度（自动调整学习率）
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=8,
    min_lr=1e-7,
    verbose=1
)

# 训练日志
class TrainingLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            print(f"\n📊 Epoch {epoch+1} 总结:")
            print(f"   训练准确率: {logs['accuracy']:.4f}")
            print(f"   验证准确率: {logs['val_accuracy']:.4f}")
            print(f"   训练损失: {logs['loss']:.4f}")
            print(f"   验证损失: {logs['val_loss']:.4f}")

logger = TrainingLogger()

# ==================== 训练模型 ====================
print("\n=== 开始训练 ===")
history = model.fit(
    train_db,
    epochs=settings.TRAIN_EPOCHS,
    validation_data=dev_db,
    callbacks=[model_check_point, early_stopping, lr_scheduler, logger],
    verbose=1
)

# ==================== 保存训练历史 ====================
print("\n=== 保存训练结果 ===")
history_path = 'history.pkl'
with open(history_path, 'wb') as f:
    pickle.dump(history.history, f)
print(f"✅ 训练历史已保存到 {history_path}")

# ==================== 评估模型 ====================
print("\n=== 评估模型 ===")

# 加载最佳模型
best_model = tf.keras.models.load_model(settings.MODEL_PATH)

# 评估训练集
train_loss, train_acc = best_model.evaluate(train_db, verbose=0)
print(f"训练集 - 准确率: {train_acc:.4f}, 损失: {train_loss:.4f}")

# 评估验证集
val_loss, val_acc = best_model.evaluate(dev_db, verbose=0)
print(f"验证集 - 准确率: {val_acc:.4f}, 损失: {val_loss:.4f}")

# 评估测试集（如果有）
if 'test_db' in dir():
    test_loss, test_acc = best_model.evaluate(test_db, verbose=0)
    print(f"测试集 - 准确率: {test_acc:.4f}, 损失: {test_loss:.4f}")

# ==================== 可视化训练曲线 ====================
print("\n=== 可视化训练曲线 ===")
show_loss_acc(history)

# ==================== 输出最终结果 ====================
print("\n" + "="*60)
print("🎉 训练完成！")
print(f"📈 最高验证准确率: {max(history.history['val_accuracy']):.4f}")
print(f"📁 模型保存位置: {settings.MODEL_PATH}")
print(f"📊 历史记录位置: {history_path}")
print("="*60)