#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf


# In[2]:


hello=tf.constant("Hello, Tensorflow!")

sess=tf.Session()

print(sess.run(hello))


# In[4]:


node1=tf.constant(3.0,tf.float32)
node2=tf.constant(4.0)#also tf.float 32 implicitly
node3=tf.add(node1,node2)#node3=node1+node2


# In[5]:


print("node1:",node1,"node2:",node2)
print("node3: ",node3)


# In[7]:


sess=tf.Session()
print("sess.run(node1,node2): ", sess.run([node1,node2]))
print("sess.run(node3): ", sess.run(node3))


# In[8]:


a=tf.placeholder(tf.float32)
b=tf.placeholder(tf.float32)
adder_node = a+b #provide a shortcut for tf.add(a,b)

print(sess.run(adder_node, feed_dict={a:3, b:4.5}))
print(sess.run(adder_node, feed_dict={a:[1,3],b:[2,4]}))


# In[ ]:


# tensor == array
# Tensor Ranks : 몇 차원 array이냐
# Tensor Shapes : 각각의 element에 몇개씩 들어있느냐 [3,3] 으로 나타냄 
# shape 굉장히 중요
# Tensor Types : 대부분 tf.float32많이씀. tf.int32도 많이 씀.
# Graph를 먼저 설계하고, 실행시키고, update하거나 리턴

