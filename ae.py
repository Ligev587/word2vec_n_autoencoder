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

def weight_variable(shape):
	return tf.Variable(tf.random_uniform(shape,
									-1.0/math.sqrt(shape[0]),
									-1.0/math.sqrt(shape[1])))

def bias_variable(shape):
	return tf.Variable(tf.zeros(shape))

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

	x=tf.placeholder(tf.float32,[None,200],name='input')

	W1=weight_variable([200,150])
	b1=bias_variable(150)
	h=tf.nn.tanh(tf.matmul(x,W1)+b1)

	W2=tf.transpose(W1)
	b2=bias_variable(200)
	y=tf.nn.tanh(tf.matmul(h,W2)+b2)

	#later try regularizer
	loss=tf.reduce_sum(tf.square(y-x),1)
	loss=tf.reduce_mean(loss)

	optimizer=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)


	init=tf.initialize_all_variables()
	sess=tf.Session()
	sess.run(init)
	result_file=open("res","w")

	for i in range(12000):
		sess.run(optimizer,feed_dict={x: feature})
		format_str=('%s: step  %d ---- loss : %f')
		print(format_str%(datetime.now(),i,loss.eval(feed_dict={x: feature},session=sess)))
		result_file.write(str(i)+"   ")
		result_file.write(str(loss.eval(feed_dict={x: feature},session=sess)))
		result_file.write('\n')
		#if loss.eval(feed_dict={x: feature},session=sess)==0.106070:
			#break;
	
	print(feature[0])
	print(y.eval(session=sess,feed_dict={x: feature})[0])
	
	output_file=open("W1","w")
	for i in range(200):
		for j in range(150):
			output_file.write(str((W1.eval(session=sess)[i][j])))
			output_file.write(' ')
		output_file.write('\n')
	output_file.close()
	print("weight matrix saved")
