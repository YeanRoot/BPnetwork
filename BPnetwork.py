import math
import random
import pandas as pd
import matplotlib.pyplot as plt


# 图解代码见readme
class Neure:  # 定义神经元类

    def __init__(self, input_data=None, input_data_weight=None, offset=None, activate_function='tanh',
                 layer=0):  # 每个神经元的输入参数1有输入信号，2每个输入信号的权重，
        if input_data_weight is None:  # 3神经元偏置，4激活函数种类，5所在层号
            input_data_weight = []
        if input_data is None:
            input_data = []
        self.input_data = input_data  # 列表
        self.input_data_weight = input_data_weight  # 列表
        self.offset = offset  # int
        self.activate_function = activate_function  # str
        self.layer = layer  # int

    def output(self):  # 得到神经元的输出
        output = self.getz()
        # for i in range(len(self.input_data)):
        #     output+=self.input_data[i]*self.input_data_weight[i]
        # output+=self.offset
        if self.activate_function == 'tanh':
            output = Function(output).tanh()
        elif self.activate_function == 'self':
            output = output
        return output

    def getz(self):  # 得到输入参数*输入参数权重+偏置的值
        z = 0
        for i in range(len(self.input_data)):
            z += self.input_data[i] * self.input_data_weight[i]
        z += self.offset
        return z


class Function:  # 本次项目所用到的函数类，名称前面带d表示是该函数的导数

    def __init__(self, x=None, x_=None):
        self.x = x  ##整形或列表
        self.x_ = x_

    def tanh(self):
        return math.tanh(self.x)

    def dtanh(self):
        return 1 - pow(math.tanh(self.x), 2)

    def softmax(self):
        sum = 0
        for i in range(len(self.x)):
            sum += math.exp(self.x[i])
        output = [math.exp(self.x[i]) / sum for i in range(len(self.x))]
        return output

    def dsoftmax(self):  # x预测值，x_真实类别序号
        doutput = []
        for i in range(len(self.x)):
            if self.x_ == i:
                doutput.append(self.x[i] * (1 - self.x[i]))
            else:
                doutput.append(-self.x[self.x_] * self.x[i])
        return doutput

    def cross_shang(self):  # x预测值，x_实际值
        sum = 0
        for i in range(len(self.x_)):
            sum += self.x_[i] * math.log(self.x[i])
        sum = -sum
        return sum

    def dcross_shang(self):  # x预测值，x_实际值
        for i in range(len(self.x_)):
            if self.x_[i] != 0:
                return [-1 / self.x[i], i]


def forward_prop(layer_num, last_num, networklayer, lastlayer):  # 前向传播算法。参数：本层神经元数目，上一层神经元数目，神经网络层，上层神经网络层
    for i in range(layer_num):
        inputdata = [lastlayer[j].output() for j in range(last_num)]  # 上一层神经元的输出整理成一个列表
        networklayer[i].input_data = inputdata
        # print(networklayer[i].input_data)
    return networklayer


def get_max_index(final_out):  # 得到预测最大值索引（也即输出层第几个神经元输出值最大）
    max_index = 0
    max = final_out[0]
    for i in range(len(final_out) - 1):
        if final_out[i + 1] > max:
            max = final_out[i + 1]
            max_index = i + 1
    return max_index


def get_grad(layer_num, last_num, delta, networklayer):  # 得到梯度值，参数：本层神经元数目，上一层神经元数目，神经网络误差，神经网络层
    grad = []
    for i in range(layer_num):
        grad_single = []
        for j in range(last_num):
            grad_single.append(delta[i] * networklayer[j].output())  # 输入信号权重的梯度＝神经网络误差*上一层神经元输出信号值
        grad.append(grad_single)
    # print(grad)
    return grad


def update_weight(layer_num, last_num, lr, networklayer, grad,
                  delta):  # 更新权重（SGD），参数：本层神经元数目，上一层神经元数目，学习率，神经网络层，梯度，神经网络误差
    for i in range(layer_num):
        update_weight = []
        for j in range(last_num):
            update_weight.append(networklayer[i].input_data_weight[j] + grad[i][j] * lr)  # 权重＝权重-梯度*学习率
        update_offset = networklayer[i].offset + delta[i] * lr  # 偏置 = 偏置-神经网络误差*学习率
        networklayer[i].input_data_weight = update_weight
        networklayer[i].offset = update_offset
        # print(networklayer[i].input_data_weight)
    return networklayer


def update_weight_mobp(layer_num, last_num, lr, networklayer, grad, delta, vdm, vdm_offset,
                       eta):  # 更新权重（mobp），参数：本层神经元数目，上一层神经元数目，学习率，神经网络层，梯度，神经网络误差，
    for i in range(layer_num):  # 动量梯度，偏置动量梯度，动量系数
        update_weight = []
        for j in range(last_num):
            vdm[i][j] = eta * vdm[i][j] + (1 - eta) * grad[i][j]  # 动量梯度＝动量系数*动量梯度+（1-动量系数）*梯度
            update_weight.append(networklayer[i].input_data_weight[j] + vdm[i][j] * lr)
        vdm_offset[i] = eta * vdm_offset[i] + (1 - eta) * delta[i]
        update_offset = networklayer[i].offset + vdm_offset[i] * lr
        networklayer[i].input_data_weight = update_weight
        networklayer[i].offset = update_offset
        # print(networklayer[i].input_data_weight)
    return networklayer, vdm, vdm_offset  # 返回了更新后的动量梯度，以便下一次传参


if __name__ == "__main__":
    # inputdata=[1]
    # weigh=[1,1,1,1]
    inputLayer_num = 4  # 输出层神经元个数
    hiddenLayer_num = 5  # 隐藏层神经元个数
    outputLayer_num = 3  # 输出层神经元个数
    lr = 0.02  # 学习率
    eta = 0.8  # 动量系数
    epoch = 50  # 学习轮数
    # update_way = "mobp" #优化算法，mobp
    update_way = "SGD"  # 优化算法，SGD（两者选其一）

    all_best_acc = []  # 使用不同学习率/隐藏层个数得到的最好测试集准确率集合
    xdata = []  # 画图时的x轴坐标值
    # for lr in range(1, 400, 5):  # 探究使用不同学习率（梯度更新步长），与以下两部分探究代码冲突
    #     lr = lr / 1000
    #     xdata.append(lr)
    # for hiddenLayer_num in range(1,30): #探究使用不同隐藏层个数，
    #     xdata.append(hiddenLayer_num)

    # 一堆画图代码
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.grid()
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train_acc")
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.grid()
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("test_acc")
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.grid()
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("train_loss")
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.grid()
    ax4.set_xlabel("epoch")
    ax4.set_ylabel("test_loss")

    for update_way in ["SGD", "mobp"]:  # 探究不同的优化方法的影响
        print(update_way)
        vdm_hidden = []  # 隐藏层动量梯度
        vdm_output = []  # 输出层动量梯度
        vdm_offset_hidden = []  # 隐藏层偏置的动量梯度
        vdm_offset_output = []  # 输出层偏置的动量梯度

        random.seed(1)  # 生成随机数种子，控制生成的权重和偏置
        inputLayer = [Neure(input_data_weight=[1],  # 生成输入层，输入权重为1，偏置为0，inputLayer_num个神经元
                            offset=0,
                            layer=0,
                            activate_function='self') for i in range(inputLayer_num)]

        hiddenLayer = [Neure(input_data_weight=[random.uniform(-1, 1) for i in range(inputLayer_num)],
                             # 生成隐藏层，输入权重和偏置在-1到1的范围内伪随机生成，每个神经元有inputLayer_num个个数的输入权重
                             offset=random.uniform(-1, 1),  # 有hiddenLayer_num个神经元
                             layer=1) for j in range(hiddenLayer_num)]

        outputLayer = [Neure(input_data_weight=[random.uniform(-1, 1) for i in range(hiddenLayer_num)],  # 生成输出层，方法同隐藏层
                             offset=random.uniform(-1, 1),
                             layer=2,
                             activate_function='self') for j in range(outputLayer_num)]

        train_reader = pd.read_csv("./dataset/train_data.csv", sep=',')  # 读取训练集
        test_reader = pd.read_csv("./dataset/test_data.csv", sep=',')  # 读取测试集
        train_len = len(train_reader)  # 训练集长度
        test_len = len(test_reader)  # 测试集长度
        train_acc = []  # 所有epoch的训练集准确率
        train_loss = []  # 所有epoch的训练集损失
        test_acc = []  # 所有epoch的测试集准确率
        test_loss = []  # 所有epoch的测试集损失
        best_acc = 0  # 所有epoch的测试集最佳准确率

        for t in range(epoch):  # 每一轮训练
            acc_sum = 0
            loss_sum = 0

            if update_way == "mobp":  # 初始化动量梯度
                vdm_hidden = [[0 for j in range(inputLayer_num)] for i in range(hiddenLayer_num)]
                vdm_output = [[0 for j in range(hiddenLayer_num)] for i in range(outputLayer_num)]
                vdm_offset_hidden = [0 for i in range(hiddenLayer_num)]
                vdm_offset_output = [0 for i in range(outputLayer_num)]

            for s in range(train_len):  # 输入每一张照片
                inputdata = train_reader.loc[
                    s, ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].tolist()  # 得到输入特征的列表
                trueclass = []  # 输出层实际正确输出
                fclass = train_reader.loc[s, ['Species']].tolist()[0]  # 花的类别
                if fclass == 0:
                    trueclass = [1, 0, 0]
                elif fclass == 1:
                    trueclass = [0, 1, 0]
                elif fclass == 2:
                    trueclass = [0, 0, 1]

                # ------------开始前向传播------------#
                for i in range(inputLayer_num):  # 把照片特征传入输入层
                    inputLayer[i].input_data = inputdata[i:i + 1]
                hiddenLayer = forward_prop(hiddenLayer_num, inputLayer_num, hiddenLayer, inputLayer)  # 进入隐藏层
                outputLayer = forward_prop(outputLayer_num, hiddenLayer_num, outputLayer, hiddenLayer)  # 进入输出层
                temp_out = [outputLayer[i].output() for i in range(outputLayer_num)]  # 输出层的临时输出值
                final_out = Function(temp_out).softmax()  # 对临时输出值作用于softmax激活函数，得到最终输出值

                max_index = get_max_index(final_out)  # 得到输出层最大输出是第几个神经元
                if max_index == fclass:  # 统计正确输出个数
                    acc_sum += 1
                # print(s, final_out,max_index)
                loss_sum += Function(final_out, trueclass).cross_shang()  # 统计损失值
                # print(s, loss)

                # ------------开始反向传播--------------#
                dshang = Function(final_out, trueclass).dcross_shang()  # 得到交叉熵的微分
                delta_output = Function(final_out, dshang[1]).dsoftmax()  # 神经网络误差=交叉熵微分*softmax函数微分
                delta_hidden = []
                for i in range(hiddenLayer_num):
                    delta = 0
                    for j in range(outputLayer_num):
                        delta += delta_output[j] * outputLayer[j].input_data_weight[i]
                    delta *= Function(hiddenLayer[i].getz()).dtanh()  # 利用递推关系式得到隐藏层的神经网络误差
                    delta_hidden.append(delta)

                grad_output = get_grad(outputLayer_num, hiddenLayer_num, delta_output, hiddenLayer)  # 得到输出层梯度
                grad_hidden = get_grad(hiddenLayer_num, inputLayer_num, delta_hidden, inputLayer)  # 得到输入层梯度

                if update_way == "SGD":  # 随机梯度下降
                    outputLayer = update_weight(outputLayer_num, hiddenLayer_num, lr, outputLayer, grad_output,
                                                delta_output)
                    hiddenLayer = update_weight(hiddenLayer_num, inputLayer_num, lr, hiddenLayer, grad_hidden,
                                                delta_hidden)
                elif update_way == "mobp":  # 动量法
                    outputLayer, vdm_output, vdm_offset_output = update_weight_mobp(outputLayer_num, hiddenLayer_num,
                                                                                    lr, outputLayer, grad_output,
                                                                                    delta_output, vdm_output,
                                                                                    vdm_offset_output, eta)
                    hiddenLayer, vdm_hidden, vdm_offset_hidden = update_weight_mobp(hiddenLayer_num, inputLayer_num, lr,
                                                                                    hiddenLayer, grad_hidden,
                                                                                    delta_hidden, vdm_hidden,
                                                                                    vdm_offset_hidden, eta)

            train_acc.append(acc_sum / train_len)
            train_loss.append(loss_sum / train_len)
            # print(train_acc,train_loss)
            acc_sum = 0
            loss_sum = 0
            for v in range(test_len):  # 测试集同理，少了反向传播梯度更新的步骤
                inputdata = test_reader.loc[
                    v, ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].tolist()
                trueclass = []
                fclass = test_reader.loc[v, ['Species']].tolist()[0]
                if fclass == 0:
                    trueclass = [1, 0, 0]
                elif fclass == 1:
                    trueclass = [0, 1, 0]
                elif fclass == 2:
                    trueclass = [0, 0, 1]
                for i in range(inputLayer_num):
                    inputLayer[i].input_data = inputdata[i:i + 1]
                hiddenLayer = forward_prop(hiddenLayer_num, inputLayer_num, hiddenLayer, inputLayer)
                outputLayer = forward_prop(outputLayer_num, hiddenLayer_num, outputLayer, hiddenLayer)
                temp_out = [outputLayer[i].output() for i in range(outputLayer_num)]
                final_out = Function(temp_out).softmax()

                max_index = get_max_index(final_out)
                if fclass == max_index:
                    acc_sum += 1
                loss_sum += Function(final_out, trueclass).cross_shang()

            test_acc.append(acc_sum / test_len)
            test_loss.append(loss_sum / test_len)
            if test_acc[t] > best_acc:
                best_acc = test_acc[t]
            print('epoch:%d' % t)
            print('train_acc:%f   test_acc:%f   best_acc%f' % (train_acc[t], test_acc[t], best_acc))
            print('train_loss:%f   test_loss:%f' % (train_loss[t], test_loss[t]))

        all_best_acc.append(best_acc)
        epochx = [i + 1 for i in range(epoch)]
        ax1.plot(epochx, train_acc)
        ax2.plot(epochx, test_acc)
        ax3.plot(epochx, train_loss)
        ax4.plot(epochx, test_loss)

    plt.show()

    # print(all_best_acc, len(all_best_acc))
    # plt.plot(xdata, all_best_acc)
    # plt.ylabel("best_acc")
    # plt.show()

# 到此结束，感谢观看（^_^）
