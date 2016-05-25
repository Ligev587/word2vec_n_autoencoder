import sys
import struct
import numpy as np

N=50 #number of closest words that will be shown

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


def calc_distance(target,voacb,feature):
	try:
		i=voacb.index(target)
		#should remeber this way
		#in numpy matrix*vec
		#and sum func
		rank=(feature*feature[i]).sum(axis=1)
	except:
		rank=None

	return rank


if __name__=='__main__':
	if len(sys.argv)<2:
		print("Usage: python distance.py <filename>")
		sys.exit()

	filename=sys.argv[1]

	try:
		with open(filename) as f:
			vocab,feature=load_data(f)
	except IOError:
			print("File not found")
			sys.exit(-1)

	while(True):
		target=raw_input("Enter word(EXIT to exit):")
		if target=="EXIT":
			break;
		rank=calc_distance(target,vocab,feature)
		if rank is None:
			print("out of dictionary")
			continue

		index_rank=dict()
		
		for i,r in enumerate(rank):
			index_rank[i]=r;
		sorted_rank=sorted(index_rank.items(),key=lambda d:d[1],reverse=True)
		for i in range(N):
			print(vocab[sorted_rank[i][0]]+"             "+str(sorted_rank[i][1]))

		print("")