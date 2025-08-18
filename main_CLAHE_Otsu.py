import os
import re
import skimage as ski
import shutil
from skimage import data, io
from save_multitif_file import *
from general_functions import *

def processar_imagens(diretorio_origem, diretorio_destino):
    os.makedirs(diretorio_destino, exist_ok=True)

    sub_directories_names = [f for f in os.listdir(diretorio_origem)
                             if os.path.isfile(os.path.join(diretorio_origem, f)) == False]

    for sub_directorie_name in sub_directories_names:
        sub_directorie_path = os.path.join(diretorio_origem, sub_directorie_name)
        sub_directorie_destiny = os.path.join(diretorio_destino, sub_directorie_name)
        sub_directorie_destiny_png = os.path.join(sub_directorie_destiny, 'PNG_figures')
        os.makedirs(sub_directorie_destiny, exist_ok=True)
        os.makedirs(sub_directorie_destiny_png, exist_ok=True)
        archives_names_PL = [f for f in os.listdir(sub_directorie_path)
                          if os.path.isfile(os.path.join(sub_directorie_path, f))
                          and f.lower().endswith('.tif')
                          and 'pl' in f]
        archives_names_SHG = [f for f in os.listdir(sub_directorie_path)
                          if os.path.isfile(os.path.join(sub_directorie_path, f))
                          and f.lower().endswith('.tif')
                          and 'shg' in f]
        for figure in archives_names_PL:
            shutil.copy(os.path.join(sub_directorie_path, figure), os.path.join(sub_directorie_destiny, figure))
        count = 1
        for figure in archives_names_SHG:
            print('{}/{}'.format(count,len(archives_names_SHG)))
            match = re.search(r'acc(\d+)', figure, flags=re.IGNORECASE)
            acumulation = int(match.group(1)) if match else None
            image = ski.io.imread(os.path.join(sub_directorie_path, figure))
            results = list()
            for c in range(image.shape[2]):
                SHG_class = interest_class_noise_min_threshold(image[:, :, c], acumulation)
                results.append(SHG_class)
                if c==1:
                    base_name = os.path.splitext(figure)[0]  # Nome do arquivo sem extensão
                    original_png = f"{base_name}_layer{c}_original.png"
                    processed_png = f"{base_name}_layer{c}_processed.png"
                    # Normaliza a camada original para 0–255 e converte para uint8
                    original_layer_scaled = ((image[:, :, c] / 4095) * 255).astype(np.uint8)
                    processed_layer_scaled = ((SHG_class / 4095) * 255).astype(np.uint8)
                    #Salva as imagens PNG
                    ski.io.imsave(os.path.join(sub_directorie_destiny_png, original_png), original_layer_scaled)
                    ski.io.imsave(os.path.join(sub_directorie_destiny_png, processed_png), processed_layer_scaled)
            multi_tif_file_save (results, sub_directorie_destiny, figure.replace('.tif',''))
            count += 1
processar_imagens('/home/ronald/Schistossoma/original_data/tif', '/home/ronald/Schistossoma/destino_nova_tentativa')
