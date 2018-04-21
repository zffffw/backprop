#bp神经网络

##使用方法

命令行输入（macOS，Linux）:

```shell
./*.out train_data_path test_data_path EPOCHS
```

train_data_path: 训练集数据，格式csv

test_data_path: 测试集数据，格式csv

EPOCHS: 训练轮数



例如：

```shell
./backprop iris.csv iris_t.csv 2000
```



windows使用方法请参照以上修改。



## 介绍

1. Linear类：该类是全连接层的类，属性有w（权重）以及b（偏置）。

   > 使用高斯随机数初始化w和b

2. LinearModel类：这是一个神经网络模型，由一个输入层，一个隐含层，一个输出层组成。

   > 成员函数有：
   >
   >  	1. forward() ：前导， 求出输出层的结果z2
   >  	2. backward() ：反向传播并且更新输出层和隐含层的w和b
   >  	3. initModel()：初始化模型的各项数值，包括w和b

3. residual_g() 和 residual_e() 表示输出层和隐含层的残差。

4. train() 训练函数

5. test() 测试函数



## 总结 

1. 模型参数固定，可变的只有隐含层的数量，且模型只有三层。