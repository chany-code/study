#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf

# X and Y data
x_train = [1,2,3]
y_train = [1,2,3]

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# Our hypothesis WX_b
hypothesis = x_train * W + b

# cost/Loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.01)
train = optimizer.minimize(cost)

#Launch the graph in a session
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer()) # tensorflow variable사용시 필수

#Fit the Line
for step in range(2001):
    sess.run(train)
    if step%20 ==0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
        


# In[5]:


# x,y 값을 직접주는 대신, placeholder로 주기
# 나중에 feed dict로
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# Our hypothesis WX_b
hypothesis = X * W + b

# cost/Loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.01)
train = optimizer.minimize(cost)

#Launch the graph in a session
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer()) # tensorflow variable사용시 필수


for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost,W, b, train], feed_dict={X:[1,2,3,4,5], Y:[2.1,3.1,4.1,5.1,6.1]})
    if step%20 ==0:
        print(step, cost_val, W_val, b_val)
    


# In[9]:


print(sess.run(hypothesis, feed_dict={X:[5]}))
print(sess.run(hypothesis, feed_dict={X:[2.5]}))
print(sess.run(hypothesis, feed_dict={X:[1.5, 3.5]}))


# In[ ]:




