from config import logger
log = logger.getLogger(__file__)


def handleError(msg):
	log.info(msg)
	f = open('./output/error.log', 'a')
	f.write(msg + '\n')
	f.close()