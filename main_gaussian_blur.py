import os
import skimage as ski
from skimage import data, io
import shutil
from save_multitif_file import *
from general_functions import *
def processar_imagens(diretorio_origem, diretorio_destino):
    archives_names = [f for f in os.listdir(diretorio_origem)
                       if os.path.isfile(os.path.join(diretorio_origem, f))
                       and f.lower().endswith('.tif')
                       and 'shg' in f]                             #Arquivos ".tif" no diretório
    os.makedirs(diretorio_destino, exist_ok=True)
    os.makedirs(diretorio_destino+'/SHG', exist_ok=True)
    count = 1
    archives_names_PL = [f for f in os.listdir(diretorio_origem)
                          if os.path.isfile(os.path.join(diretorio_origem, f))
                          and f.lower().endswith('.tif')
                          and 'pl' in f]
    for figure in archives_names_PL:
            shutil.copy(os.path.join(diretorio_origem, figure), os.path.join(os.path.join(diretorio_destino, 'SHG'), figure))
    for figure in archives_names:
      print('{}/{}'.format(count,len(archives_names)))
      fator_multiplicativo = int([a for a in ((figure.strip()).replace('.tif','')).split('-') if 'acc' in a.lower()][0].lower().replace('acc',''))
      image = ski.io.imread(os.path.join(diretorio_origem, figure))
      results = list()
      for c in range(image.shape[2]):
        SHG_class = interest_class_noise_gaussian_blur(image[:, :, c], fator_multiplicativo)
        results.append(SHG_class)
        if c==1:
          base_name = os.path.splitext(figure)[0]  # Nome do arquivo sem extensão
          original_png = f"{base_name}_layer{c}_original.png"
          processed_png = f"{base_name}_layer{c}_processed.png"
          # Normaliza a camada original para 0–255 e converte para uint8
          original_layer_scaled = ((image[:, :, c] / 4095) * 255).astype(np.uint8)
          processed_layer_scaled = ((SHG_class / 4095) * 255).astype(np.uint8)
          #Salva as imagens PNG
          ski.io.imsave(os.path.join(diretorio_destino, original_png), original_layer_scaled)
          ski.io.imsave(os.path.join(diretorio_destino, processed_png), processed_layer_scaled)
      multi_tif_file_save (results, os.path.join(diretorio_destino,'SHG'), figure.replace('.tif',''))
      count += 1

file_path = '/home/ronald/Schistossoma/original_data/tif/60dias'

path_destiny = '/home/ronald/Schistossoma/new_gaussian'

processar_imagens(file_path, path_destiny)
