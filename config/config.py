import os

# Diretórios comuns
ROOT_DIR = os.path.abspath(os.getcwd())
OUTPUT_DIR = ROOT_DIR + '/output'
IMG_DIR = ROOT_DIR + '/img'
ROTATED = OUTPUT_DIR + '/rotated/'
CROPPED = OUTPUT_DIR + '/cropped/'

# Máscaras para o glob
CASOS 		= IMG_DIR + '/casos'
CONTROLES 	= IMG_DIR + '/controles'
a22q11 		= IMG_DIR + '/22q11'
ANGELMAN 	= IMG_DIR + '/Angelman'
APERT 		= IMG_DIR + '/Apert'
CDL 		= IMG_DIR + '/CDL'
DOWN 		= IMG_DIR + '/Down'
FRAGILEX 	= IMG_DIR + '/FragileX'
MARFAN 		= IMG_DIR + '/Marfan'
PROGERIA 	= IMG_DIR + '/Progeria'
SOTOS 		= IMG_DIR + '/Sotos'
TREACHER 	= IMG_DIR + '/TreacherCollins'
TURNER 		= IMG_DIR + '/Turner'
WILLIAMS 	= IMG_DIR + '/Williams'

IMAGE_MASK 	= '/*image*.jpg'
DSCN_MASK 	= '/DSCN*.JPG'
