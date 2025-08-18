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

    for sub_directory_name in sub_directories_names:
        sub_directory_path = os.path.join(diretorio_origem, sub_directory_name)
        sub_directory_destiny = os.path.join(diretorio_destino, sub_directory_name)
        sub_directory_destiny_png = os.path.join(sub_directory_destiny, 'PNG_figures')
        os.makedirs(sub_directory_destiny, exist_ok=True)
        os.makedirs(sub_directory_destiny_png, exist_ok=True)
        archives_PL_list, correspondent_archives_SHG, correspondent_accumulation_PL, \
            correspondent_accumulation_SHG = generate_list_PL_and_SHG_archives_names_and_accumulations(sub_directory_path)
        for i in range(len(archives_PL_list)):
            shutil.copy(os.path.join(sub_directory_path, archives_PL_list[i]), os.path.join(sub_directory_destiny, archives_PL_list[i]))
            print(f'{i+1}/{len(correspondent_archives_SHG)}')
            SHG_acc = correspondent_accumulation_SHG[i]
            image_SHG = ski.io.imread(os.path.join(sub_directory_path, correspondent_archives_SHG[i]))
            results = list()
            for c in range(image_SHG.shape[2]):
                SHG_class = interest_class_noise(image_SHG[:, :, c], correspondent_accumulation_SHG[i])
                results.append(SHG_class)
                if c==1:
                    base_name = os.path.splitext(correspondent_archives_SHG[i])[0]  # Nome do arquivo sem extensão
                    original_png = f"{base_name}_layer{c}_original.png"
                    processed_png = f"{base_name}_layer{c}_processed.png"
                    # Normaliza a camada original para 0–255 e converte para uint8
                    original_layer_scaled = np.clip((image_SHG[:, :, c] / 4095) * 255, 0, 255).astype(np.uint8)
                    processed_layer_scaled = np.clip((SHG_class / 4095) * 255, 0, 255).astype(np.uint8)
                    #Salva as imagens PNG
                    ski.io.imsave(os.path.join(sub_directory_destiny_png, original_png), original_layer_scaled)
                    ski.io.imsave(os.path.join(sub_directory_destiny_png, processed_png), processed_layer_scaled)

            multi_tif_file_save (results, sub_directory_destiny, correspondent_archives_SHG[i].replace('.tif',''))
processar_imagens('/home/ronald/Schistossoma/original_data/tif', '/home/ronald/Schistossoma/new_CLAHE_Otsu_with_random_noise')
