import os
import sys
import codecs
import math
import random
import numpy as np
import tensorflow as tf
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



#Read wn's .tab file
if __name__ == '__main__':
	

	if len(sys.argv)<2:
		print("Usage: python ae.py <filename>")
		sys.exit()

	filename=sys.argv[1]
	try:
		with open(filename) as f:
			vocab,feature=load_data(f)
			f.close()
	except IOError:
		print("File not exist")
		sys.exit(-1)


#==================================================================
	wn_file='wn-data-eng.tab'
	reader=codecs.open(wn_file,"r").readlines()
	wn={}
	wn_vocab=[]

	for l in reader:
		if l[0]=="#":
			continue;
		v=l.split("\t")[0]
		k=l.split("\t")[2][:-1]

		if len(k.split())>1:
			continue
		if "-" in k:
			continue
		k=k.lower()
		try:
			idx=vocab.index(k)
		except:
			continue

		try:
			temp=wn[k]
			wn[k]=temp+";"+v
		except KeyError:
			wn[k]=v
		
		try:
			idx=wn_vocab.index(k)
		except:
			wn_vocab.append(k)
	print(len(wn_vocab))
		
#==================================================================
	try:
		with open('W1','r') as f:
			_W1=load_matrix(f)
			f.close()
	except IOError:
		print("weight file doesn't exist")
		sys.exit(-1)

	try:
		with open('b1','r') as f:
			_b1=load_bias(f)
			f.close()
	except IOError:
		print("bias file doesn't exist")
		sys.exit(-1)

	try:
		with open('W2','r') as f:
			_W2=load_matrix(f)
			f.close()
	except IOError:
		print("weight file doesn't exist")
		sys.exit(-1)

	try:
		with open('b2','r') as f:
			_b2=load_bias(f)
			f.close()
	except IOError:
		print("bias file doesn't exist")
		sys.exit(-1)

	x=tf.placeholder(tf.float32,[None,200],name='input')
	#y=tf.placeholder(tf.float32,name='label')

	W1=weight_variable(_W1)
	b1=bias_variable(_b1)
	h1=tf.nn.tanh(tf.matmul(x,W1)+b1)
	
	norm=tf.sqrt(tf.reduce_sum(tf.square(h1),1,keep_dims=True))
	h1_norm=h1/norm
	_s0,_s1,_n0,_n1,_n2,_n3,_n4=tf.split(0,7,h1_norm)
	sim2=tf.matmul(_s0,_s1,transpose_b=True)

	W2=weight_variable(_W2)
	b2=bias_variable(_b2)
	h2=tf.nn.tanh(tf.matmul(h1,W2)+b2)

	norm=tf.sqrt(tf.reduce_sum(tf.square(h2),1,keep_dims=True))
	h2_norm=h2/norm

	s0,s1,n0,n1,n2,n3,n4=tf.split(0,7,h2_norm)
	sim=tf.matmul(s0,s1,transpose_b=True)
	p=tf.exp(tf.matmul(s0,s1,transpose_b=True))
	q=tf.exp(tf.matmul(s0,n0,transpose_b=True))+tf.exp(tf.matmul(s0,n1,transpose_b=True))+tf.exp(tf.matmul(s0,n2,transpose_b=True))+tf.exp(tf.matmul(s0,n3,transpose_b=True))+tf.exp(tf.matmul(s0,n4,transpose_b=True))

	
	#y_=tf.sigmoid(p)
	#loss=tf.nn.sigmoid_cross_entropy_with_logits(y_,y)
	#loss=-y*tf.log(y_)-(1-y)*tf.log(1-y_)
	loss=-tf.log(tf.div(p,q))
	optimizer=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
	
	init=tf.initialize_all_variables()

	sess=tf.Session()
	sess.run(init)

	random.shuffle(wn_vocab)
	K=10

	for i in range(K):
		test_set=wn_vocab[i*int(len(wn_vocab)/K):i*int(len(wn_vocab)/K)+int(len(wn_vocab)/K)+1]
		train_set=wn_vocab[0:i*int(len(wn_vocab)/K)]+wn_vocab[i*int(len(wn_vocab)/K)+int(len(wn_vocab)/K)+1:]

		for j in range(len(train_set)-1):
			idx1=vocab.index(train_set[j])
			list1=wn[train_set[j]].split(";")

			for k in range(j+1,len(train_set)):
				_x=[]
				list2=wn[train_set[k]].split(";")
				for element in list2:
					if element in list1:
						idx2=vocab.index(train_set[k])
						_x.append(feature[idx1])
						_x.append(feature[idx2])
						#sess.run(optimizer,feed_dict={x: _x,y: np.array([1])})
						#format_str=("%s: Fold: %d Positive --WordA: %s --WordB: %s")
						#print(format_str%(datetime.now(),i,train_set[j],train_set[k]))

						#negative sample 5:
						cnt=5
						neg=[]
						while cnt>0:
							randnum=random.randint(0,len(train_set)-1)
							if randnum==j:
								continue
							if randnum==k:
								continue
							if randnum in neg:
								continue
							list3=wn[train_set[randnum]].split(";")
							flag=False
							for element in list3:
								if element in list1:
									flag=True
									break

							if flag==False:
								_x.append(feature[vocab.index(train_set[randnum])])
								cnt=cnt-1
								neg.append(randnum)
							else:
								continue
						
						_x=np.asarray(_x,dtype=np.float32)
						sess.run(optimizer,feed_dict={x: _x})
						format_str=("Pos : %s %s Neg: %s %s %s %s %s")
						print(format_str%(train_set[j],train_set[k],train_set[neg[0]],train_set[neg[1]],train_set[neg[2]],train_set[neg[3]],train_set[neg[4]]))
						break
					else:
						continue

		output_file=open(str(i)+"-Fold",'a')

		for j in range(len(test_set)-1):
			for k in range(j+1,len(test_set)):
				list1=wn[test_set[j]].split(";")
				list2=wn[test_set[k]].split(";")

				for element in list2:
					if element in list1:

						idx1=vocab.index(test_set[j])
						idx2=vocab.index(test_set[k])

						threshold=feature[idx1].dot(feature[idx2])
						x_test=[]
						x_test.append(feature[idx1])
						x_test.append(feature[idx2])

						for r in range(5):
							x_test.append(feature[idx2])
		
						x_test=np.asarray(x_test,dtype=np.float32)
					
						output_file.write("positive "+test_set[j]+" "+test_set[k]+"\n")
						output_file.write(str(threshold)+" ")
						output_file.write(str(sim.eval(session=sess,feed_dict={x: x_test})))
						output_file.write(" ")
						output_file.write(str(sim2.eval(session=sess,feed_dict={x: x_test})))
						output_file.write("\n")

					else:
						continue

		output_file.close()


#=======================recording=======================================================

	output_file=open("W1_finetuned","w")
	for i in range(200):
		for j in range(150):
			output_file.write(str((W1.eval(session=sess)[i][j])))
			output_file.write(' ')
		output_file.write('\n')
	output_file.close()

	output_file=open("W2_finetuned","w")
	for i in range(150):
		for j in range(100):
			output_file.write(str((W2.eval(session=sess)[i][j])))
			output_file.write(' ')
		output_file.write('\n')
	output_file.close()

	out_file=open("b1_finetuned","w")
	for i in range(150):
		out_file.write(str(b1.eval(session=sess)[i]))
		out_file.write(' ')
	out_file.write('\n')
	out_file.close()

	out_file=open("b2_finetuned","w")
	for i in range(100):
		out_file.write(str(b2.eval(session=sess)[i]))
		out_file.write(' ')
	out_file.write('\n')
	out_file.close()				
