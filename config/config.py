import os

# Diretórios comuns
ROOT_DIR 	= os.path.abspath(os.getcwd())
OUTPUT_DIR 	= ROOT_DIR + '/output'
IMG_DIR 	= ROOT_DIR + '/img'
ROTATED 	= OUTPUT_DIR + '/rotated'
CROPPED 	= OUTPUT_DIR + '/cropped'
EYES 		= OUTPUT_DIR + '/eyes'
GRADIENT 	= OUTPUT_DIR + '/gradient'

# Máscaras para o glob
CASOS 		= '/casos'
CONTROLES 	= '/controles'
a22q11 		= '/22q11'
ANGELMAN 	= '/Angelman'
APERT 		= '/Apert'
CDL 		= '/CDL'
DOWN 		= '/Down'
FRAGILEX 	= '/FragileX'
MARFAN 		= '/Marfan'
PROGERIA 	= '/Progeria'
SOTOS 		= '/Sotos'
TREACHER 	= '/TreacherCollins'
TURNER 		= '/Turner'
WILLIAMS 	= '/Williams'

IMAGE_MASK 	= '/*image*.jpg'
DSCN_MASK 	= '/DSCN*.JPG'
