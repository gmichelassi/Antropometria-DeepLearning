import os

ROOT_DIR = os.path.abspath(os.getcwd())
OUTPUT_DIR = ROOT_DIR + '/output'
IMG_DIR = ROOT_DIR + '/img'
ROTATED = OUTPUT_DIR + '/rotated'
CROPPED = OUTPUT_DIR + '/cropped'
EYES = OUTPUT_DIR + '/eyes'
GRADIENT = OUTPUT_DIR + '/gradient'

IMAGE_MASK = '/*image*.jpg'
DSCN_MASK = '/DSCN*.JPG'

CASOS = IMG_DIR + '/casos/DSCN*.JPG'
CONTROLES = IMG_DIR + '/controles/DSCN*JPG'
A22Q11 = IMG_DIR + '/22q11/*image*.jpg'
ANGELMAN = IMG_DIR + '/Angelman/*image*.jpg'
APERT = IMG_DIR + '/Apert/*image*.jpg'
CDL = IMG_DIR + '/CDL/*image*.jpg'
DOWN = IMG_DIR + '/Down/*image*.jpg'
FRAGILEX = IMG_DIR + '/FragileX/*image*.jpg'
MARFAN = IMG_DIR + '/Marfan/*image*.jpg'
PROGERIA = IMG_DIR + '/Progeria/*image*.jpg'
SOTOS = IMG_DIR + '/Sotos/*image*.jpg'
TREACHER = IMG_DIR + '/Treacher/*image*.jpg'
TURNER = IMG_DIR + '/Turner/*image*.jpg'
WILLIAMS = IMG_DIR + '/Williams/*image*.jpg'

DEFAULT_SHAPE = (584, 584, 3)
N_SPLITS = 10

LABELS = {
	'controles': 0,
	'casos': 1,
	'22q11': 2,
	'Angelman': 3,
	'Apert': 4,
	'CDL': 5,
	'Down': 6,
	'FragileX': 7,
	'Marfan': 8,
	'Progeria': 9,
	'Sotos': 10,
	'TreacherCollings': 11,
	'Turner': 12,
	'Williams': 13
}
