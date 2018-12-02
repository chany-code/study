#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import matplotlib.pyplot as plt

X =[1,2,3]
Y =[1,2,3]

W = tf.placeholder(tf.float32)
# Our hypothesis
hypothesis = X *W

# cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

sess = tf.Session()
#Initializes Globla Variables
sess.run(tf.global_variables_initializer())
# Variables for plotting CF
W_val=[]
cost_val=[]
for i in range(-30,50):
    feed_W=i*0.1
    curr_cost, curr_W = sess.run([cost,W],feed_dict={W:feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)
    
# Show the CF
plt.plot(W_val, cost_val)
plt.show()


# In[4]:


x_data = [1,2,3]
y_data = [1,2,3]
W = tf.Variable(tf.random_normal([1]), name='weight')
X = tf.placeholder(tf.float32)
Y= tf.placeholder(tf.float32)

hypothesis = X * W

cost = tf.reduce_sum(tf.square(hypothesis - Y))

#Minimize : 
learning_rate=0.1
gradient = tf.reduce_mean((W*X-Y)*X)
descent = W - learning_rate * gradient
update = W.assign(descent)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1) 

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(21):
    sess.run(update, feed_dict={X:x_data, Y:y_data})
    print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}),sess.run(W))


# In[7]:


X=[1,2,3]
Y=[1,2,3]

W =tf.Variable(-3.0)
hypothesis = X * W
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run(W))
    sess.run(train)


# In[ ]:




