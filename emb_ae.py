import numpy as np
import sparse_autoencoder
import scipy.optimize

words=[]
vectors=[]

def load_embeddings(filename):
	
	with open(filename,"r") as f:
		try:
			line1=f.readline()

			line=f.readline()
			while line:
				yield line
				line=f.readline()

		finally:
			f.close()
			yield None

#initialize the parameters
visible_size=150
hidden_size=100

sparsity_param=0.1
_lambda=3e-3
beta=3

#read line by line
reader=load_embeddings("wiki_50")
line=reader.next()
words.append(line.split()[0])
vector=line.split()[1:]
for i in range(len(vector)):
	vector[i]=float(vector[i])
vector=np.asarray(vector,dtype=np.float32)
vectors.append(vector)
while line:
	line=reader.next()
	if line is None:
		break
	words.append(line.split()[0])
	vector=line.split()[1:]
	for i in range(len(vector)):
		vector[i]=float(vector[i])
	vector=np.asarray(vector,dtype=np.float32)
	vectors.append(vector)

#reshape and start train
vectors=np.asarray(vectors,dtype=np.float32).transpose()

theta=sparse_autoencoder.initialize(hidden_size,visible_size)
J=lambda x:sparse_autoencoder.sparse_autoencoder_cost(x,visible_size,hidden_size,_lambda,sparsity_param,beta,vectors)
_options={'maxiter':400,'disp':True}
result=scipy.optimize.minimize(J,theta,method='L-BFGS-B',jac=True,options=_options)
ae_opt_theta=result.x
ae_feature=sparse_autoencoder.sparse_autoencoder(ae_opt_theta,hidden_size,visible_size,vectors)
ae_feature=ae_feature.transpose()

#write in the results
output_filename="wiki_word2vec_ae"
output_file=open(output_filename,"w")
for i in range(len(words)):
	output_file.write(words[i]+' ')
	for j in range(len(ae_feature[i,:])):
		output_file.write(str(ae_feature[i,j])+' ')
	output_file.write('\n')
output_file.close()
