import tensorflow as tf
import numpy as np
import math
import sys
from datetime import datetime

def load_data(f):
	word_n_size=f.readline()
	words,size=word_n_size.split()
	words,size=int(words),int(size)

	vocab=[]
	feature=[]

	for i in range(words):
		temp_data=f.readline()
		w=temp_data.split(' ',1)[0]
		vocab.append(w)

		vec=temp_data.split(' ',1)[1].split()
		for i in range(len(vec)):
					vec[i]=float(vec[i])
		vec=np.asarray(vec,dtype=np.float32)
		length=np.sqrt((vec**2).sum())
		vec=vec/length
		feature.append(vec)

	feature=np.asarray(feature,dtype=np.float32)

	return vocab,feature

def load_matrix(f):
	W=[]
	for line in f.readlines():
		entry=line.split()
		for i in range(len(entry)):
			entry[i]=float(entry[i])
		W.append(entry)
	W=np.asarray(W,dtype=np.float32)

	return W

def load_bias(f):
	b=f.readline().split()
	for i in range(len(b)):
		b[i]=float(b[i])
	b=np.asarray(b,dtype=np.float32)
	
	return b


def weight_variable(W):
	return tf.Variable(W)

def bias_variable(b):
	return tf.Variable(b)

if __name__=='__main__':
	if len(sys.argv)<2:
		print("Usage: python ae.py <filename> ")
		sys.exit()

	filename=sys.argv[1]
	try:
		with open(filename) as f:
			vocab,feature=load_data(f)
			f.close()
	except IOError:
		print("File not exist")
		sys.exit(-1)


	try:
		with open('W_300_250','r') as f:
			W=load_matrix(f)
			f.close()
	except IOError:
		print("weight file doesn't exist")
		sys.exit(-1)


	try:
		with open('b_300_250','r') as f:
			b=load_bias(f)
			f.close()
	except IOError:
		print("bias file doesn't exist")
		sys.exit(-1)	

	try:
		with open('W_250_200','r') as f:
			_W1=load_matrix(f)
			f.close()
	except IOError:
		print("weight file doesn't exist")
		sys.exit(-1)


	try:
		with open('b_250_200','r') as f:
			_b1=load_bias(f)
			f.close()
	except IOError:
		print("bias file doesn't exist")
		sys.exit(-1)

	try:
		with open('W_200_150','r') as f:
			_W2=load_matrix(f)
			f.close()
	except IOError:
		print("weight file doesn't exist")
		sys.exit(-1)


	try:
		with open('b_200_150','r') as f:
			_b2=load_bias(f)
			f.close()
	except IOError:
		print("bias file doesn't exist")
		sys.exit(-1)

	

	try:
		with open('W_150_100','r') as f:
			_W3=load_matrix(f)
			f.close()
	except IOError:
		print("weight file doesn't exist")
		sys.exit(-1)

	try:
		with open('b_150_100','r') as f:
			_b3=load_bias(f)
			f.close()
	except IOError:
		print("bias file doesn't exist")
		sys.exit(-1)


	
	feature=feature.dot(W)+b
	feature=np.tanh(feature)
	feature=feature.dot(_W1)+_b1
	feature=np.tanh(feature)
	feature=feature.dot(_W2)+_b2
	feature=np.tanh(feature)

	x=tf.placeholder(tf.float32,[None,150],name='input')

	W1=weight_variable(_W3)
	b1=bias_variable(_b3)
	h=tf.nn.tanh(tf.matmul(x,W1)+b1)

	W2=tf.transpose(W1)
	b2=tf.Variable(tf.zeros(150))
	y=tf.nn.tanh(tf.matmul(h,W2)+b2)

	#later try regularizer
	loss=tf.reduce_sum(tf.square(y-x),1)
	loss=tf.reduce_mean(loss)

	optimizer=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)


	init=tf.initialize_all_variables()
	sess=tf.Session()
	sess.run(init)

	for i in range(200):
		sess.run(optimizer,feed_dict={x: feature})
		format_str=('%s: step  %d ---- loss : %f')
		print(format_str%(datetime.now(),i,loss.eval(feed_dict={x: feature},session=sess)))
	
	output_file=open("W_150_100","w")
	for i in range(150):
		for j in range(100):
			output_file.write(str((W1.eval(session=sess)[i][j])))
			output_file.write(' ')
		output_file.write('\n')
	output_file.close()

	out_file=open("b_150_100","w")
	for i in range(100):
		out_file.write(str(b1.eval(session=sess)[i]))
		out_file.write(' ')
	out_file.write('\n')
	out_file.close()
	print("bias matrix saved")
