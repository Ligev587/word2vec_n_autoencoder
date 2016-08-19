import numpy as np
import sys

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


	try:
		with open("embeddings_150dim") as f:
			vocab1,feature1=load_data(f)
			f.close()
	except IOError:
		print("File not exist")
		sys.exit(-1)

	try:
		with open("embeddings_100dim") as f:
			vocab2,feature2=load_data(f)
			f.close()
	except IOError:
		print("File not exist")
		sys.exit(-1)


	for i in range(10):
		f=open(str(i)+"-Fold")

		right1=0
		wrong1=0
		right2=0
		wrong2=0
		right3=0
		wrong3=0
		right4=0
		wrong4=0
		right5=0
		wrong5=0
		right6=0
		wrong6=0

		line=f.readline()
		while line:
			label=line.split()[0]
			word1=line.split()[1]
			word2=line.split()[2]

			idx1=vocab1.index(word1)
			idx2=vocab1.index(word2)

			line=f.readline()
			threshold=float(line.split()[0])
			threshold1=feature1[idx1].dot(feature1[idx2])
			threshold2=feature2[idx1].dot(feature2[idx2])

			sim=line.split(" ",1)[1]			
			sim1=sim.split("[[")[1]
			sim2=sim.split("[[")[2]
			sim1=sim1.split("]]")[0]
			sim2=sim2.split("]]")[0]
			
			sim1=float(sim1)
			sim2=float(sim2)
			if sim1<threshold:
				wrong1=wrong1+1
			else:
				right1=right1+1

			if sim1<threshold1:
				wrong2=wrong2+1
			else:
				right2=right2+1

			if sim1<threshold2:
				wrong3=wrong3+1
			else:
				right3=right3+1

			if sim2<threshold:
				wrong4=wrong4+1
			else:
				right4=right4+1

			if sim2<threshold1:
				wrong5=wrong5+1
			else:
				right5=right5+1

			if sim2<threshold2:
				wrong6=wrong6+1
			else:
				right6=right6+1


			line=f.readline()
			format_str=("%s: %f %f")
			#print(format_str%(label,threshold,sim))
		accuracy=float(right4)/float(right4+wrong4)
		print(str(i)+" fold 150d to original 200d accuracy: "+str(accuracy))
		accuracy=float(right5)/float(right5+wrong5)
		print(str(i)+" fold 150d to original 150d accuracy: "+str(accuracy))
		accuracy=float(right6)/float(right6+wrong6)
		print(str(i)+" fold 150d to original 100d accuracy: "+str(accuracy))
		accuracy=float(right1)/float(right1+wrong1)
		print(str(i)+" fold 100d to original 200d accuracy: "+str(accuracy))
		accuracy=float(right2)/float(right2+wrong2)
		print(str(i)+" fold 100d to original 150d accuracy: "+str(accuracy))
		accuracy=float(right3)/float(right3+wrong3)
		print(str(i)+" fold 100d to original 100d accuracy: "+str(accuracy))
