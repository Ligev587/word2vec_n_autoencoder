import numpy as np
import csv

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

def calc_sim(target,source,vocab,feature):
	try:
		i=vocab.index(target)
		j=vocab.index(source)
		sim=feature[i].dot(feature[j])
	except:
		sim=-1

	return sim

if __name__=='__main__':
	csvfile=open("combined.csv","r",encoding="utf-8")
	reader=csv.reader(csvfile)
	with open("wiki_word2vec_0527_ae",encoding='utf8') as f:
		ae_vocab,ae_feature=load_data(f)
		f.close()
	with open("wiki_word2vec_0527",encoding='utf8') as f:
		noae_vocab,noae_feature=load_data(f)
		f.close()
	human=[]
	sim_noae=[]
	sim_ae=[]

	for line in reader:
		target=line[0]
		source=line[1]
		noae_sim=calc_sim(target,source,noae_vocab,noae_feature)
		if noae_sim<0:
			continue;
		ae_sim=calc_sim(target,source,ae_vocab,ae_feature)

		sim_ae.append(ae_sim)
		sim_noae.append(noae_sim)
		human.append(line[2])

	csvfile.close()
	"""
	opt_file=open("pearson.output","w")

	for i in range(len(human)):
		opt_file.write(str(human[i])+" "+str(sim_ae[i])+" "+str(sim_noae[i])+"\n")
	opt_file.close()
	"""

	sim_noae=np.asarray(sim_noae,dtype=np.float32)
	sim_ae=np.asarray(sim_ae,dtype=np.float32)
	human=np.asarray(human,dtype=np.float32)

	res_noae=(sim_noae**2).sum()-sim_noae.sum()*sim_noae.sum()/len(human)
	res_noae=res_noae*((human**2).sum()-human.sum()*human.sum()/len(human))
	res_noae=(sim_noae.dot(human)-sim_noae.sum()*human.sum()/len(human))/math.sqrt(res_noae)

	res_ae=(sim_ae**2).sum()-sim_ae.sum()*sim_ae.sum()/len(human)
	res_ae=res_ae*((human**2).sum()-human.sum()*human.sum()/len(human))
	res_ae=(sim_ae.dot(human)-sim_ae.sum()*human.sum()/len(human))/math.sqrt(res_ae)

	res_of_noae=np.corrcoef(sim_noae,human)
	res_of_ae=np.corrcoef(sim_ae,human)

	print(res_of_noae)
	print(res_noae)
	print(res_of_ae)
	print(res_ae)