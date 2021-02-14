def handleError(msg):
	log.info(msg)
	f = open('../output/error.txt', 'a')
	f.write(msg + '\n')
	f.close()
