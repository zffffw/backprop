#include <iostream>
#include <vector>
#include <math.h>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <fstream>
#include <sstream>  
#include <string> 
using namespace std;
#define hidden 7 //隐含层神经元个数
#define traindatasize 90  //训练集大小
#define testdatasize 60  //测试集大小
#define lrate 0.01  // 学习速率
#define input_size 4  // 训练集的数据规格，输入层神经元个数
#define output_size 3  // 输出层神经元个数
typedef std::vector<double> DVector;  
typedef std::vector<DVector> DVector2D; //一个二维vector

/* 函数以及模型声明 */
class LinearModel;
DVector2D mult(DVector2D input, DVector2D W);
DVector2D addbias(DVector2D input, DVector2D B);
DVector2D residual_g(DVector2D input_label, DVector2D output);
DVector2D residual_e(DVector2D b, DVector2D g, DVector2D w);
DVector2D update_b(DVector2D b, DVector2D g, double lr);
DVector2D update_w(DVector2D w, DVector2D g, DVector2D x, double lr);
DVector2D getRes(DVector2D data, LinearModel model);
void printV(DVector2D v);
double GaussianNoise(double mu, double sigma);
DVector2D sigmoid(DVector2D z);



//随机产生正太分布数
double GaussianNoise(double mu, double sigma)
{
    const double epsilon = std::numeric_limits<double>::min();
    const double two_pi = 2.0*3.14159265358979323846;

    static double z0, z1;
    static bool generate;
    generate = !generate;

    if (!generate)
       return z1 * sigma + mu;

    double u1, u2;
    do
     {
       u1 = rand() * (1.0 / RAND_MAX);
       u2 = rand() * (1.0 / RAND_MAX);
     }
    while ( u1 <= epsilon );

    z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
    z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
    return z0 * sigma + mu;
}

//打印vector
void printV(DVector2D v) {
    int row = v.size();
    int col = v[0].size();
    for(int i = 0; i < row; ++i) {
        for(int j = 0; j < col; ++j) {
            cout << v[i][j] << ' ' ;
        }
        cout << endl;
    }
    cout << endl;
}

//sigmoid函数
DVector2D sigmoid(DVector2D z) {
    int row = z.size();
    int col = z[0].size();
    for(int i = 0; i < row; ++i) {
        for(int j = 0; j < col; ++j) {
            z[i][j] = 1.0/(1 + exp(-z[i][j]));
        }
    }
    return z;
}

//Linear  参数 w：权重  b：偏置
class Linear {
    public:
        DVector2D w;  //wight
        DVector2D b;  //bias
        //高斯随机数初始化w和b
        void initwb(int input, int output) {
            w.resize(output);
            for(int i = 0; i < output; ++i) {
                w[i].resize(input);
            }
            for(int i = 0; i < output; ++i) {
                for( int j = 0; j < input; ++j) {
                    w[i][j] = GaussianNoise(0, 1);
                }
            }
            
            b.resize(output);
            for(int i = 0; i < output; ++i) {
                b[i].resize(1);
            }
            for(int i = 0; i < output; ++i) {
                b[i][0] = GaussianNoise(0, 1);
            }
            
        }
};

class LinearModel {
    public:
        Linear l1; // 隐含层
        Linear l2; // 输出层
        DVector2D z1;  // 隐含层神经元的数值
        DVector2D z2;  // 输出层神经元的数值
        double lr1;  //隐含层w和b学习速率
        double lr2; //输出层w和b学习速率
        //初始化模型各个参数
        void initLinearModel(double learning_rate1, double learning_rate2) {
            lr1 = learning_rate1;
            lr2 = learning_rate2;
            l1.initwb(input_size, hidden);
            l2.initwb(hidden, output_size);

            z1.resize(hidden);
            for(int i = 0; i < hidden; ++i) {
                z1[i].resize(1);
            }
            for(int i = 0; i < hidden; ++i) {
                z1[i][0] = 0;
            }

            z2.resize(3);
            for(int i = 0; i < output_size; ++i) {
                z2[i].resize(1);
            }
            for(int i = 0; i < output_size; ++i) {
                z1[i][0] = 0;
            }
        }
        //前导
        void forward(DVector2D input) {
            z1 = mult(input, l1.w);
            z1 = addbias(z1, l1.b);
            z1 = sigmoid(z1);
            // printV(z1);

            z2 = mult(z1, l2.w);
            z2 = addbias(z2, l2.b);
            z2 = sigmoid(z2);
            // printV(z2);

        }
        
        //反向传播
        void backward(DVector2D input_label, DVector2D input_data) {
            DVector2D g, e;
            g = residual_g(input_label, z2);
            e = residual_e(z1, g, l2.w);

            l2.w = update_w(l2.w, g, z1, lr1);
            l2.b = update_b(l2.b, g, lr1);

            l1.w = update_w(l1.w, e, input_data, lr2);
            l1.b = update_b(l1.b, e, lr2);


        }
};

LinearModel model; //定义全局神经网络模型

//计算损失数值，利用损失函数的算法
double loss(DVector2D input_label, DVector2D input_data) {
    int row = input_label.size();
    double los = 0;
    DVector2D res;
    res = getRes(input_data, model);
    for(int i = 0; i < row; ++i) {
        los += pow(input_label[i][0] - res[i][0], 2);
    }
    los /= 2.0;
    return los;
}

//矩阵相乘，返回结果output
DVector2D mult(DVector2D input, DVector2D W) {
    DVector2D output;
    int row = W.size();
    output.resize(row);
    for(int i = 0; i < row; ++i) {
        output[i].resize(1);
    }
    for(int i = 0; i < row; ++i) {
        output[i][0] = 0;
    }

    for(int i = 0; i < row; ++i) {
        double sum = 0;
        for(int j = 0; j < input.size(); ++j) {
            sum += W[i][j]*input[j][0];
        }
        output[i][0] = sum;
    }
    return output;
}

//加偏置
DVector2D addbias(DVector2D input, DVector2D B) {
    int row = input.size();
    for(int i = 0; i < row; ++i) {
        input[i][0] += B[i][0];
    }
    return input;
}

//计算残差1， g 详见周志华 机器学习的推导
DVector2D residual_g(DVector2D input_label, DVector2D output) {
    DVector2D g;
    int row = input_label.size();
    g.resize(row);
    for(int i = 0; i < row; ++i) {
        g[i].resize(1);
    }
    
    for(int i = 0; i < row; ++i) {
        g[i][0] = output[i][0]*(1 - output[i][0])*(input_label[i][0] - output[i][0]);
    }
    return g;
}

//计算残差2，e
DVector2D residual_e(DVector2D b, DVector2D g, DVector2D w) {
    DVector2D e;
    int row = b.size();
    e.resize(row);
    for(int i = 0; i < row; ++i) {
        e[i].resize(1);
    }
    for(int h = 0; h < row; ++h) {
        double wg = 0;
        int rr = g.size();
        for(int j = 0; j < rr; ++j) {
            wg += w[j][h]*g[j][0];
        }
        e[h][0] = b[h][0]*(1 - b[h][0])*wg;
    }
    return e;
}

//更新权重
DVector2D update_w(DVector2D w, DVector2D g, DVector2D x, double lr) {
    int row = w.size();
    int col = w[0].size();
    for(int i = 0; i < row; ++i) {
        for(int j = 0; j < col; ++j) {
            w[i][j] += lr*g[i][0]*x[j][0];
        }
    }
    return w;
}

//更新偏置
DVector2D update_b(DVector2D b, DVector2D g, double lr) {
    int row = b.size();
    for(int i = 0; i < row; ++i) {
        b[i][0] -= lr*g[i][0];
    }
    return b;
}




//获取结果
DVector2D getRes(DVector2D data, LinearModel model) {
    DVector2D label;
    label.resize(output_size);
    for(int i = 0; i < output_size; ++i) {
        label[i].resize(1);
    }

    //矩阵运算求出结果label
    label = mult(data, model.l1.w);
    label = addbias(label, model.l1.b);
    label = sigmoid(label);

    label = mult(label, model.l2.w);
    label = addbias(label, model.l2.b);
    label = sigmoid(label);

    return label;
}

//判断模型输出的结果是否与标签相近， 准则为判断最大数所在的行，label为3*1规格
bool compare_test_data(DVector2D test_res, DVector2D test_label) {
    int index_res = 0;
    int index_label = 0;
    double maxl = 0, maxr = 0;
    for(int i = 0; i < output_size; ++i) {
        if(test_label[i][0] > maxl) {
            index_label = i;
            maxl = test_label[i][0];
        }
        if(test_res[i][0] > maxr) {
            index_res = i;
            maxr = test_res[i][0];
        }
    }
    return index_res == index_label;
}

double test(double test_datas[][5], int test_labels[][3], int data_size=traindatasize) {
    DVector2D test_data;
    DVector2D test_label;
    DVector2D test_res; //获取model输出的数据
    test_data.resize(input_size);
    test_label.resize(output_size);
    for(int i = 0; i < input_size; ++i) {
        test_data[i].resize(1);
    }
    for(int i = 0; i < output_size; ++i) {
        test_label[i].resize(1);
    }
    int count = 0; //count记录测试集中判断正确的个数
    for(int i = 0; i < data_size; ++i) {
        for(int j = 0; j < input_size; ++j) {
            test_data[j][0] = test_datas[i][j];
        }
        for(int j = 0; j < output_size; ++j) {
            test_label[j][0] = test_labels[i][j];
        }
        test_res = getRes(test_data, model);
        if(compare_test_data(test_res, test_label)) {
            count += 1;
        } 
    }
    double acc = (double)(count)/(double)(data_size);
    return acc;
}
        


//训练  batch_size = 1, 
void train(int epochs, double train_datas[][5], int train_labels[][3], int data_size) {
    int i = 0;
    int idx;
    DVector2D train_data, train_label;
    train_data.resize(input_size);
    for(int i = 0; i < input_size; ++i) {
        train_data[i].resize(1);
    }
    train_label.resize(output_size);
    for(int i = 0; i < output_size; ++i) {
        train_label[i].resize(1);
    }
    //i为丢进训练的数据个数，且batch_size = 1， 每次只丢进一个数进行训练
    while(i <= epochs * data_size) {
        idx = i % data_size;


        for(int j = 0; j < input_size; ++j) {
            train_data[j][0] = train_datas[idx][j];
        }
        for(int j = 0; j < output_size; ++j) {
            train_label[j][0] = train_labels[idx][j];
        }
        model.forward(train_data);
        model.backward(train_label, train_data);
        //每100轮输出一次训练集的测试结果
        if (i % (100*data_size) == 0) {
            double los;
            los = loss(train_label, train_data);
            double acc = 0;
            acc = test(train_datas, train_labels, data_size);
            cout << "EPOCHS:" << i/data_size << ' ';
            cout << "loss:" << los << "  acc:" << acc << endl;
        }
        ++i;
    }
}

//读取csv文件
void read_data(double data[][5], int label[][3], int n=traindatasize, string train_path=0) {
	// n is the number of data we want read
    ifstream fin(train_path); 
    char temp_file[200];
    fin.getline(temp_file, sizeof(temp_file));
	string s;
    for(int i=0;i<n;i++)
    {
		int id;
        fin >> id;
        fin.ignore(1,',');
        fin >> data[i][0];

        fin.ignore(1,',');
        fin >> data[i][1];

        fin.ignore(1,',');
        fin >> data[i][2];

        fin.ignore(1,',');
        fin >> data[i][3];

        fin.ignore(1,',');
        fin >> s;

        if (s == "Iris-setosa") {

            label[i][0] = 1;
            label[i][1] = 0;
            label[i][2] = 0;
		} else if (s == "Iris-versicolor") {

            label[i][0] = 0;
            label[i][1] = 1;
            label[i][2] = 0;
		} else if (s == "Iris-virginica") {

            label[i][0] = 0;
            label[i][1] = 0;
            label[i][2] = 1;
		} else {
			cout << "Name Error!" << endl;
		}
    }

    fin.close();
    return;
}


int main(int argc, char* argv[]) {
    string train_path;
    string test_path;
    string E;
    int EPOCHS;
    if (argc > 1) {
        train_path = string(argv[1]);
        test_path = string(argv[2]);
        E = string(argv[3]);
    } else {
        cout << " 输入错误 ！！！" <<  "\n 退出程序。" << endl;
        exit(0);
    }
    EPOCHS = stoi(E);
    model.initLinearModel(lrate, lrate*10);
    double train_data[traindatasize][5], test_data[testdatasize][5];
    int train_label[traindatasize][3], test_label[testdatasize][3];
    read_data(train_data, train_label, traindatasize, train_path);
    cout << "成功读取训练数据！" << endl;
    read_data(test_data, test_label, testdatasize, test_path);
    cout << "成功读取测试数据！" << endl;
    cout << "开始训练！" << endl;

    train(EPOCHS, train_data, train_label, traindatasize);
    cout << "训练结束！" << endl;
    double acc = test(test_data, test_label, testdatasize);
    cout << "开始测试数据！ " << endl;
    cout << "测试数据集正确率:" << acc << endl;
    
}
