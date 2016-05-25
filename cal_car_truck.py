import numpy as np

with open("wiki_word2vec") as f:
	flag=0
	line=f.readline()
	while line:
		if(line.split()[0]=="car" or line.split()[0]=="truck"):
			if(line.split()[0]=="car"):
				car=line.split()[1:]
				for i in range(len(car)):
					car[i]=float(car[i])
				car=np.asarray(car,dtype=np.float32)
				line=f.readline()
				continue
			if(line.split()[0]=="truck"):
				truck=line.split()[1:]
				for i in range(len(truck)):
					truck[i]=float(truck[i])
				truck=np.asarray(truck,dtype=np.float32)
				print("car vector:")
				print(car)
				print("truck vector:")
				print(truck)
				print("cosine result of word2vec:")
				res=car.dot(truck)/(np.sqrt(car.dot(car))*np.sqrt(truck.dot(truck)))
				print(res)
				break;
		else:
			line=f.readline()


with open("wiki_word2vec_ae") as f:
	flag=0
	line=f.readline()
	while line:
		if(line.split()[0]=="car" or line.split()[0]=="truck"):
			if(line.split()[0]=="car"):
				car=line.split()[1:]
				for i in range(len(car)):
					car[i]=float(car[i])
				car=np.asarray(car,dtype=np.float32)
				line=f.readline()
				continue
			if(line.split()[0]=="truck"):
				truck=line.split()[1:]
				for i in range(len(truck)):
					truck[i]=float(truck[i])
				truck=np.asarray(truck,dtype=np.float32)
				print("car vector:")
				print(car)
				print("truck vector:")
				print(truck)
				print("cosine result of word2vec_ae:")
				res=car.dot(truck)/(np.sqrt(car.dot(car))*np.sqrt(truck.dot(truck)))
				print(res)
				break;
		else:
			line=f.readline()

