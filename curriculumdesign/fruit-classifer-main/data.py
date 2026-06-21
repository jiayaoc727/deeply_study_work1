import os
import random
import tensorflow as tf
import settings

# 每个类别选取的图片数量
samples_per_class = settings.SAMPLES_PER_CLASS
# 图片根目录
images_root = settings.IMAGES_ROOT
# 类别->编码的映射
class_code_map = settings.CLASS_CODE_MAP

# 我们准备使用经典网络在imagenet数据集上的预训练权重，所以归一化时也要使用imagenet的平均值和标准差
image_mean = tf.constant(settings.IMAGE_MEAN)
image_std = tf.constant(settings.IMAGE_STD)


def normalization(x):
    """
    对输入图片x进行归一化，返回归一化的值
    """
    return (x - image_mean) / image_std


def train_preprocess(x, y):
    """
    对训练数据进行预处理
    """
    # 读取图片
    x = tf.io.read_file(x)
    
    # 解码图片（兼容 JPEG、PNG，禁用动图）
    x = tf.image.decode_image(x, channels=3, expand_animations=False)
    
    # 确保是 3 维张量
    x = tf.ensure_shape(x, (None, None, 3))
    
    # 将图片缩放到[244,244]
    x = tf.image.resize(x, [244, 244])
    
    # 随机左右翻转
    if random.choice([0, 1]):
        x = tf.image.random_flip_left_right(x)
    
    # 随机裁剪
    x = tf.image.random_crop(x, [224, 224, 3])
    
    # 归一化到[0,1]
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = normalization(x)
    
    # 标签转 one-hot
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, settings.CLASS_NUM)
    
    return x, y


def dev_preprocess(x, y):
    """
    对验证集和测试集进行数据预处理
    """
    # 读取图片
    x = tf.io.read_file(x)
    x = tf.image.decode_image(x, channels=3, expand_animations=False)
    x = tf.ensure_shape(x, (None, None, 3))
    x = tf.image.resize(x, [224, 224])
    
    # 归一化
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = normalization(x)
    
    # 标签转 one-hot
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, settings.CLASS_NUM)
    
    return x, y


# (图片路径,标签)的列表
image_path_and_labels = []
sub_images_dir_list = sorted(list(os.listdir(images_root)))

# 遍历每一个子目录
for sub_images_dir in sub_images_dir_list:
    sub_path = os.path.join(images_root, sub_images_dir)
    
    if os.path.isdir(sub_path) and sub_images_dir in settings.CLASSES:
        current_label = class_code_map.get(sub_images_dir)
        
        # 只保留有效的图片格式
        images = []
        for img_name in sorted(os.listdir(sub_path)):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append(img_name)
        
        # 随机打乱并截取
        random.seed(settings.RANDOM_SEED)
        random.shuffle(images)
        images = images[:samples_per_class]
        
        # 构建(x,y)对
        for image_name in images:
            abs_image_path = os.path.join(sub_path, image_name)
            image_path_and_labels.append((abs_image_path, current_label))

# 计算各数据集样例数
total_samples = len(image_path_and_labels)
train_samples = int(total_samples * settings.TRAIN_DATASET)
dev_samples = int(total_samples * settings.DEV_DATASET)
test_samples = total_samples - train_samples - dev_samples

# 打乱数据集
random.seed(settings.RANDOM_SEED)
random.shuffle(image_path_and_labels)

# 将图片数据和标签数据分开
x_data = tf.constant([img for img, label in image_path_and_labels])
y_data = tf.constant([label for img, label in image_path_and_labels])

# 划分数据集
train_db = tf.data.Dataset.from_tensor_slices((x_data[:train_samples], y_data[:train_samples]))
train_db = train_db.shuffle(10000).map(train_preprocess).batch(settings.BATCH_SIZE)

dev_db = tf.data.Dataset.from_tensor_slices(
    (x_data[train_samples:train_samples + dev_samples], y_data[train_samples:train_samples + dev_samples]))
dev_db = dev_db.map(dev_preprocess).batch(settings.BATCH_SIZE)

test_db = tf.data.Dataset.from_tensor_slices(
    (x_data[train_samples + dev_samples:], y_data[train_samples + dev_samples:]))
test_db = test_db.map(dev_preprocess).batch(settings.BATCH_SIZE)

print('总样例数:', total_samples)
print('训练集样例数:', train_samples)
print('开发集样例数:', dev_samples)
print('测试集样例数:', test_samples)