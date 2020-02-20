import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

np.random.seed(777)
he=tf.contrib.layers.variance_scaling_initializer()

os.environ['CUDA_VISIBLE_DEVICE']='1'

# preprocessing
def train_data(dir,k): # for X
    li=[]
    train_list = os.listdir(dir)
    train_list=train_list[100*k:100*(k+1):]
    for img in train_list:
        image=cv2.imread('/dog_and_cat/train/'+img, cv2.IMREAD_COLOR)
        image=cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
        li.append(image)
    img_li=np.array(li)
    return img_li/255


def convert_label(dir): # for Y
    train_list=os.listdir(dir)
    for i in train_list:
        if "dog" in i:
            idx=train_list.index(i)
            train_list[idx]=1
        elif "cat" in i:
            idx = train_list.index(i)
            train_list[idx]=0
    train_list=np.array(train_list)
    data=train_list.reshape(-1,1)
    one_hot=OneHotEncoder(sparse=False,categories='auto')
    one_hot.fit(data)
    label=one_hot.transform(data)
    np.random.shuffle(label)
    return label

def test_data(dir):
    test_list=os.listdir(dir)
    test_list=test_list[10000:10300:]+test_list[22500:22800:]
    size = len(test_list)
    n_arr = np.empty((size, 32, 32, 3), np.float32)
    for img in test_list:
        image=cv2.imread('/dog_and_cat/train/'+img, cv2.IMREAD_COLOR)
        image=cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        idx=test_list.index(img)
        n_arr[idx]=image
    np.random.shuffle(n_arr)
    return n_arr/255

def test_label(dir):
    test_list=os.listdir(dir)
    test_list = test_list[10000:10300:]+test_list[22500:22800:]
    for i in test_list:
        if "dog" in i:
            idx=test_list.index(i)
            test_list[idx]=1
        elif "cat" in i:
            idx = test_list.index(i)
            test_list[idx]=0
    data=np.array(test_list).reshape(-1,1)
    one_hot=OneHotEncoder(sparse=False,categories='auto')
    one_hot.fit(data)
    label=one_hot.transform(data)
    np.random.shuffle(label)
    return label


dir_test='/dog_and_cat/test1'
dir_train='/dog_and_cat/train' # 12500 cat, 12500 dog
batch_size=100
epoch_num=5
learning_rate=0.01
keep_prob=1


class Model:

    def __init__(self, sess, name):
        self.sess=sess
        self.name=name
        self.Process()

    def Process(self):
        with tf.variable_scope(self.name):

            self.X=tf.placeholder(tf.float32,[None,32,32,3])
            self.Y=tf.placeholder(tf.float32,[None,2])


            # Layer1
            L1=tf.layers.conv2d(self.X,32,3,(1,1),padding="SAME",activation=tf.nn.relu,kernel_initializer=he)
            L1=tf.nn.max_pool(L1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
            # L1=tf.nn.dropout(L1,keep_prob)
            print(L1.shape)

            # Layer2
            L2=tf.layers.conv2d(L1,64,3,(1,1),padding="SAME",activation=tf.nn.relu,kernel_initializer=he)
            L2=tf.nn.max_pool(L2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
            # L2 = tf.nn.dropout(L2, keep_prob)
            print(L2.shape)

            # Layer3
            L3=tf.layers.conv2d(L2,128,3,(1,1),padding="SAME",activation=tf.nn.relu,kernel_initializer=he)
            L3=tf.nn.max_pool(L3,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
            # L3 = tf.nn.dropout(L3, keep_prob)
            L3_re=tf.reshape(L3,[-1,128*4*4])
            print(L3.shape)

            # fc
            # layer4
            W4=tf.get_variable("W4",shape=[128*4*4,650],initializer=he) # Conv. layer에서도 초기값이 어디냐에 따라 목표값을 찾는 과정이 다르므로 해줘야 한다.
            b4=tf.Variable(tf.random_normal([650]))
            L4=tf.nn.relu(tf.matmul(L3_re,W4)+b4)
            print(L4.shape)

            # layer5
            W5=tf.get_variable("W5",shape=[650,2],initializer=he)
            b5=tf.Variable(tf.random_normal([2]))
            self.logits=tf.matmul(L4,W5)+b5 # 여기서는 그냥 output값 -> accuracy로 넘어갈 때 softmax 함수를 걸어줘야 함
            print(self.logits.shape)

            # calc cost func & optimizer
            self.cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.Y)) # softmax 포함
            self.optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)

            # accuracy
            predict=tf.equal(tf.argmax(tf.nn.softmax(self.logits),1), tf.argmax(self.Y,1)) # 확률값
            self.accuracy=tf.reduce_mean(tf.cast(predict,tf.float32))

    def get_accuracy(self, x,y):
                return self.sess.run(self.accuracy,feed_dict={self.X:x,self.Y:y})

    def train(self, x, y):
                return self.sess.run([self.cost,self.optimizer],feed_dict={self.X:x,self.Y:y})

# initializing
sess=tf.Session()
m1= Model(sess,'m1')
init=tf.global_variables_initializer()
sess.run(init)

print("learning start")
bat_y=convert_label(dir_train)
# seed=0
for epoch in range(epoch_num):
    # seed+=1
    # np.random.seed(seed)
    avg_cost=0
    total_batch=int(25000/batch_size)
    tmp=0
    for i in range(total_batch):
        if 0<=i<100 or 125<=i<225:
            batch_x = train_data(dir_train, i)
            # np.random.shuffle(batch_x)
            batch_y=bat_y[100*i:100*(i+1):]
            # np.random.shuffle(batch_y)
            c,_=m1.train(batch_x,batch_y)
            avg_cost+=c/total_batch
            tmp += 1
            if tmp%10==0:print(tmp)
        else:
            pass
    print("epoch:%04d"%(epoch+1)," cost:{:.9f}".format(avg_cost))

print("learning end")

x_test=test_data(dir_train)
y_test=test_label(dir_train)

print("accuracy:",m1.get_accuracy(x_test,y_test))