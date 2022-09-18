# BPnetwork
不使用工具库（如pytorch，TensorFlow，numpy等）实现bp网络
既然要构建神经网络，首先我的思路是构建一个神经元类(Neure)：
 ![image](https://github.com/YeanRoot/BPnetwork/tree/main/image/image.png)

对应代码如下：

 

接下来是构建神经网络层，我是直接生成了一个由Neure类构成的列表当做网络层，列表内第几个元素表示该层第几个神经元。输入层的话权重设为1，偏置为零。其它层偏置和权重都是伪随机生成（通过random.seed()函数控制）。
代码如下：
 

前向传播部分比较简单，调用Neure类的output（）方法，输入到下一层Neure的input_data即可。
难的地方在于反向传播，这一部分要注意一下三点
1、	把SGD和mobp的具体原理搞懂，了解每一层dC/dz的递推关系式。
2、	了解交叉熵和softmax函数的求导过程，尤其是softmax函数，还得分两种情况讨论。
3、	要非常小心链式求导的反向传递的过程，千万不能把某部分的导数乘到另一部分。
大致的思路和反向传播的代码如下：
 
 
 
 
 

剩下的就是统计正确率和损失以及画图了。
不足之处请多多指教！
