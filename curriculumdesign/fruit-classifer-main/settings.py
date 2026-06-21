# 图片保存根目录（更新为新数据路径）
IMAGES_ROOT = '../data75/train'

# 每个类别选取的图片数量（根据数据量调整）
SAMPLES_PER_CLASS = 500

# 图片类别（中文名称 -> 目录名称）
FRUIT_CLASS = {
    '苹果': 'apple',
    '香蕉': 'banana',
    '桃子': 'peach',
    '葡萄': 'grape',
    '橘子': 'orange',
    '蓝莓': 'blueberry',
    '西瓜': 'watermelon',
    '草莓': 'strawberry',
    '芒果': 'mango',
    '菠萝': 'pineapple',
    '梨': 'pear',
    '樱桃': 'cherry',
    '柠檬': 'lemon',
    '柚子': 'grapefruit',
    '猕猴桃': 'kiwi',
    '石榴': 'pomegranate',
    '柿子': 'persimmon',
    '荔枝': 'lychee',
    '榴莲': 'durian',
    
}

# 参与训练的类别（目录名称列表）
CLASSES = [
    'apple', 'banana', 'peach', 'grape', 'orange', 'blueberry',
    'watermelon', 'strawberry', 'mango', 'pineapple',
    'pear', 'cherry', 'lemon', 'grapefruit', 'kiwi',
    'pomegranate', 'persimmon', 'lychee', 'durian'
]

# 参与训练的类别数量
CLASS_NUM = len(CLASSES)

# 类别->编号的映射
CLASS_CODE_MAP = {
    'apple': 0, 'banana': 1, 'peach': 2, 'grape': 3, 'orange': 4,
    'blueberry': 5, 'watermelon': 6, 'strawberry': 7, 'mango': 8, 'pineapple': 9,
    'pear': 10, 'cherry': 11, 'lemon': 12, 'grapefruit': 13, 'kiwi': 14,
    'pomegranate': 15, 'persimmon': 16, 'lychee': 17, 'durian': 18
}

# 编号->类别的映射（中文名称）
CODE_CLASS_MAP = {
    0: '苹果', 1: '香蕉', 2: '桃子', 3: '葡萄', 4: '橘子',
    5: '蓝莓', 6: '西瓜', 7: '草莓', 8: '芒果', 9: '菠萝',
    10: '梨', 11: '樱桃', 12: '柠檬', 13: '柚子', 14: '猕猴桃',
    15: '石榴', 16: '柿子', 17: '荔枝', 18: '榴莲'
}
# 随机数种子
RANDOM_SEED = 13  # 四个类别时样本较为均衡的随机数种子
# RANDOM_SEED = 19  # 三个类别时样本较为均衡的随机数种子

# 训练集比例
TRAIN_DATASET = 0.6
# 开发集比例
DEV_DATASET = 0.2
# 测试集比例
TEST_DATASET = 0.2

# mini_batch大小
BATCH_SIZE = 32

# imagenet数据集均值
IMAGE_MEAN = [0.485, 0.456, 0.406]
# imagenet数据集标准差
IMAGE_STD = [0.299, 0.224, 0.225]

# 学习率
LEARNING_RATE = 0.0001
# 训练epoch数
TRAIN_EPOCHS = 100
# 保存训练模型的路径
MODEL_PATH = 'model.h5'

# Web服务端口
WEB_PORT = 5000