import sys

if __name__=='__main__':
	if len(sys.argv)<2:
		print("Usage:python cal_accuracy.py <filename>")
		sys.exit()

	filename=sys.argv[1]

	correct=0
	wrong=0

	try:
		with open(filename,'r',encoding='utf8') as f:
			for i in range(9582):	
				line=f.readline()
				backup=line.split()[0:3]
				target=line.split()[3]

				line=f.readline()
				line=line.replace('-','')
				line=line.split()

				if target in line:
					idx=line.index(target)
					if(idx==0):
						correct+=1
					else:
						idx-=1
						while(idx>=0):
							if line[idx] in backup:
								idx-=1
							else:
								wrong+=1
								break
						if idx<0:
							correct+=1
				else:
					wrong+=1					

				line=f.readline()

			print(correct)
			print(wrong)
		f.close()
	except IOError:
		print("file not found")
		sys.exit(-1)