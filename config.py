import os

# Diretórios comuns
ROOT_DIR = os.path.abspath(os.getcwd())
OUTPUT_DIR = ROOT_DIR + '/output'
IMG_DIR = ROOT_DIR + '/img'

# Máscaras para o glob
CASOS = IMG_DIR + '/casos'
CONTROLES = IMG_DIR + '/controles'
DSCN_MASK = '/DSCN*.JPG'
