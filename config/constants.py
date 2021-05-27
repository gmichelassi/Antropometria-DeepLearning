import os

# Diretórios comuns
ROOT_DIR = os.path.abspath(os.getcwd())
OUTPUT_DIR = ROOT_DIR + '/output'
IMG_DIR = ROOT_DIR + '/img'
ROTATED = OUTPUT_DIR + '/rotated'
CROPPED = OUTPUT_DIR + '/cropped'
EYES = OUTPUT_DIR + '/eyes'
GRADIENT = OUTPUT_DIR + '/gradient'

# Máscaras para o glob
CASOS = '/casos'
CONTROLES = '/controles'
a22q11 = '/22q11'
ANGELMAN = '/Angelman'
APERT = '/Apert'
CDL = '/CDL'
DOWN = '/Down'
FRAGILEX = '/FragileX'
MARFAN = '/Marfan'
PROGERIA = '/Progeria'
SOTOS = '/Sotos'
TREACHER = '/TreacherCollins'
TURNER = '/Turner'
WILLIAMS = '/Williams'

IMAGE_MASK = '/*image*.jpg'
DSCN_MASK = '/DSCN*.JPG'

# Labels para as imagens
LABELS = {
	'controles': 	0,
	'casos': 		1,
	'22q11': 		2,
	'Angelman': 	3,
	'Apert': 		4,
	'CDL': 			5,
	'Down': 		6,
	'FragileX': 	7,
	'Marfan': 		8,
	'Progeria': 	9,
	'Sotos': 		10,
	'TreacherCollings': 11,
	'Turner': 		12,
	'Williams': 	13
}
