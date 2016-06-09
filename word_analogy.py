import sys
import numpy as np

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


if __name__=='__main__':
	if len(sys.argv)<2:
		print("Usage: python word_analogy.py <filename>")
		sys.exit()

	filename=sys.argv[1]

	try:
		with open(filename) as f:
			vocab,feature=load_data(f)
			f.close()
	except IOError:
		print("File not found")
		sys.exit(-1)
	#initialization
	i=0
	cosadd_correct,cosadd_error=0,0;
	cosmul_correct,cosmul_error=0,0;
	with open("questions-words.txt") as f:
		#output_file1=open("analogy_result_3cosAdd",'w')
		output_file2=open("analogy_result_3cosMul_noae",'w')

		for line in f.readlines():
			i+=1
			print("processing line %d" %i)
			if line.split()[0]==':':
				continue

			a1=line.split()[0]
			a2=line.split()[1]
			b1=line.split()[2]
			b2=line.split()[3]

			try:
				a1_idx=vocab.index(a1)
			except:
				continue
			try:
				a2_idx=vocab.index(a2)
			except:
				continue
			try:
				b1_idx=vocab.index(b1)
			except:
				continue

			rank_a1=(feature*feature[a1_idx]).sum(axis=1)
			rank_a2=(feature*feature[a2_idx]).sum(axis=1)
			rank_b1=(feature*feature[b1_idx]).sum(axis=1)

			index_rank=dict()
			
			"""
			rank_3cosAdd=rank_b1-rank_a1+rank_a2
			#bb_index=rank_3cosAdd.argmax()
			for j,r in enumerate(rank_3cosAdd):
				index_rank[j]=r;
			sorted_rank=sorted(index_rank.items(),key=lambda d:d[1],reverse=True)
			output_file1.write(a1+" "+a2+" "+b1+" "+b2+"\n")
			for k1 in range(5):
				output_file1.write(vocab[sorted_rank[k1][0]]+" ")
			output_file1.write("\n")
			output_file1.write("\n")
			"""

			rank_3cosMul=rank_b1*rank_a2/(rank_a1+0.000001)
			#bb_index=rank_3cosMul.argmax()
			for k,r in enumerate(rank_3cosMul):
				index_rank[k]=r;
			sorted_rank=sorted(index_rank.items(),key=lambda d:d[1],reverse=True)
			output_file2.write(a1+" "+a2+" "+b1+" "+b2+"\n")
			for k2 in range(5):
				output_file2.write(vocab[sorted_rank[k2][0]]+" ")
			output_file2.write("\n")
			output_file2.write("\n")

		#output_file1.close()
		output_file2.close()
		f.close()