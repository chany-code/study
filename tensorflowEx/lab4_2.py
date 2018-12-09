#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np

xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype = np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:,[-1]]

#Make sure the shape and data are OK
print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)


# In[11]:


import tensorflow as tf

tf.set_random_seed(777) # for reproducibility

xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype = np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:,[-1]]

#Make sure the shape and data are OK
print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

#Placeholder for a tensor will be always fed
X=tf.placeholder(tf.float32, shape=[None, 3])
Y=tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3,1]),name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X,W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess= tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001) :
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X:x_data, Y:y_data})
    if step %10 ==0 :
        print(step, "Cost :", cost_val, "\nPrediction : \n", hy_val)
        
#Ask my score
print("Your score will be ", sess.run(hypothesis, feed_dict={X:[[100,70,101]]}))
print("Other scores will be ", sess.run(hypothesis, feed_dict={X:[[60,70,110], [90,100,80]]}))


# In[1]:


#Queue 사용하기

"""
#1
filename_queue=tf.train.string_input_producer(['data01.csv', 'data02.csv', ...], shuffle=False, name='filename_queue')

#2
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

#3
record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

#batch : 펌프같은 역할
train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10) #'한번에 10개 가져와라'
sess = tf.Session()
...

#Start populating the filename queue.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    ...
    feed_dict = {X:x_batch, Y:y_batch}
    
coord.request_stop()
coord.join(threads)

"""


# In[ ]:




