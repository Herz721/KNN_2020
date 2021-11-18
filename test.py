#coding:utf-8
 
import operator
import struct
import numpy as np
import matplotlib.pyplot as plt

# 训练集
train_images_idx3_ubyte_file = 'MNIST_data/train-images-idx3-ubyte'
# 训练集标签
train_labels_idx1_ubyte_file = 'MNIST_data/train-labels-idx1-ubyte'
# 测试集
test_images_idx3_ubyte_file = 'MNIST_data/t10k-images-idx3-ubyte'
# 测试集标签
test_labels_idx1_ubyte_file = 'MNIST_data/t10k-labels-idx1-ubyte'
 
def decode_idx3_ubyte(idx3_ubyte_file):
    bin_data = open(idx3_ubyte_file, 'rb').read()
 
    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))
 
    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images
 
 
def decode_idx1_ubyte(idx1_ubyte_file):
    bin_data = open(idx1_ubyte_file, 'rb').read()
 
    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))
 
    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels
 
 
def classify(VerifySet,TrainSet,labels,k):
    Trainsize = TrainSet.shape[0]
    Diffset = np.tile(VerifySet,(Trainsize,1))-TrainSet
    sqDiffset = Diffset**2
    sqDistances = sqDiffset.sum(axis=1)
    #距离计算完毕
    distances = sqDistances ** 0.5
    #距离从小到大排序，返回距离的序号
    sortedDistIndicies = distances.argsort()
    #字典
    classCount = {}
    #前K个距离最小的
    for i in range(k):
        #labels[sortedDistIndicies[0]]为距离最小的数据样本的标签
        voteIlabel = labels[sortedDistIndicies[i]]
        #以标签为key,支持该标签+1
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #返回邻近最多的标签值
    return sortedClassCount[0][0]
 
 
if __name__ == '__main__':
    train_images = decode_idx3_ubyte(train_images_idx3_ubyte_file)
    train_labels = decode_idx1_ubyte(train_labels_idx1_ubyte_file)
    test_images = decode_idx3_ubyte(test_images_idx3_ubyte_file)
    test_labels = decode_idx1_ubyte(test_labels_idx1_ubyte_file)
 
    mTrain = 55000  #训练数据大小
    TrainSet = np.zeros((mTrain, 784))  
 
    for i in range(mTrain):
        for j in range(28):
            for k in range(28):
                TrainSet[i, 28*j+k] = train_images[i][j][k]
    

    knn_data = []
    miss_data = []

    for knn in range(1, 60, 3):  #K值大小
        errorCount = 0.0
        mTest = 1000  #测试数据大小
        for i in range(mTest):
            RightLabel = test_labels[i]
            VerifySet = np.zeros(784)
            for j in range(28):
                for k in range(28):
                    VerifySet[28*j+k] = test_images[i][j][k]  
 
            Result = classify(VerifySet, TrainSet, train_labels, knn)
            print("识别结果：%d 正确答案：%d" % (Result, RightLabel))
            if (Result != RightLabel):
                errorCount += 1.0
        missrate = errorCount / float(mTest)
        print("\n错误率： %f" % missrate)
        knn_data.append(knn)
        miss_data.append(missrate)
    plt.plot(knn_data,miss_data)
    plt.show()