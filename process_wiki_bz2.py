import logging
import os.path
import sys
import codecs

from gensim.corpora import WikiCorpus

if __name__ == '__main__':
	program=os.path.basename(sys.argv[0])
	logger=logging.getLogger(program)

	logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s')
	logging.root.setLevel(level=logging.INFO)
	logger.info("running %s"%' '.join(sys.argv))

	if len(sys.argv)<3:
		#print(globals()['__doc__'] % locals())
		sys.exit(1)

	inp,outp=sys.argv[1:3]
	space=""
	i=0

	output=codecs.open(outp,'w','utf-8')
	wiki=WikiCorpus(inp,lemmatize=False,dictionary={})

	for text in wiki.get_texts():
		#encoding is a trouble...
		text_str=""
		for j in range(len(text)):
			text_str+=str(text[j],encoding='utf-8')
			if j!=len(text)-1:
				text_str+=" "
		#write in	
		output.write(space.join(text_str)+"\n")
		i=i+1
		if(i%10000==0):
			logger.info("Saved "+str(i)+" articles")

	output.close()

	logger.info("Finished Saved "+str(i)+" articles")
