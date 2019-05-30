import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
tf.set_random_seed(10)

X = tf.placeholder(tf.float32, [None, 28,28,1])
Y_ = tf.placeholder(tf.float32, [None, 10])
lr = tf.placeholder(tf.float32)

K=4
L=8
M=12
N=200

W1 = tf.Variable(tf.truncated_normal([5,5,1,K], stddev=0.1))
B1 = tf.Variable(tf.ones([K])/10)

W2 = tf.Variable(tf.truncated_normal([5,5,K,L], stddev=0.1))
B2 = tf.Variable(tf.ones([L])/10)

W3 = tf.Variable(tf.truncated_normal([4,4,L,M], stddev=0.1))
B3 = tf.Variable(tf.ones([M])/10)

W4 = tf.Variable(tf.truncated_normal([7*7*M,N], stddev=0.1))
B4 = tf.Variable(tf.ones([N])/10)

W5 = tf.Variable(tf.truncated_normal([N,10], stddev=0.1))
B5 = tf.Variable(tf.ones([10])/10)

stride = 1
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
stride = 2
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
stride = 2
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])

Y4=tf.nn.relu(tf.matmul(YY,W4)+B4)
Ylogist = tf.matmul(Y4,W5)+B5

Y = tf.nn.softmax(Ylogist)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Ylogist))
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()

saver = tf.train.Saver()
saver.restore(sess, './model2.cpkt')

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

img = mpimg.imread('2.png')
gray = rgb2gray(img)
test = 1 - np.array(gray.reshape(-1, 28, 28, 1))

plt.imshow(gray, cmap='gray')
plt.show()

print(sess.run(tf.argmax(Y, 1), feed_dict={X: test}))
