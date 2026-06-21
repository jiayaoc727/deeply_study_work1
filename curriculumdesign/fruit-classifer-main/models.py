import tensorflow as tf
import matplotlib.pyplot as plt
import settings

# 模型加载，指定图片处理的大小和是否进行迁移学习
def my_densenet():
    IMG_SHAPE = (224, 224, 3)
    
    # 使用更强的预训练模型
    base_model = tf.keras.applications.ResNet50(
        include_top=False, 
        weights='imagenet', 
        input_shape=IMG_SHAPE
    )
    
    # 解冻更多层
    base_model.trainable = True
    fine_tune_at = 100  # 从第100层开始训练
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((224, 224, 3)),
        
        # 添加数据增强层（在模型内部）
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        
        # 增加中间层
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Dense(settings.CLASS_NUM, activation=tf.nn.softmax)
    ])
    
    return model



# 展示训练过程的曲线
def show_loss_acc(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


if __name__ == '__main__':
    model = my_densenet()
    model.summary()