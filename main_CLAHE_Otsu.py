import os
import re
import skimage as ski
import shutil
from skimage import data, io
from save_multitif_file import *
from general_functions import *

def list_and_process_figures(diretorio_origem, diretorio_destino):
    archives_PL_list, correspondent_archives_SHG, correspondent_accumulation_PL, \
    correspondent_accumulation_SHG = generate_list_PL_and_SHG_archives_names_and_accumulations(diretorio_origem)
    sub_directory_destiny_png = os.path.join(diretorio_destino, 'PNG_figures')
    os.makedirs(sub_directory_destiny_png, exist_ok=True)
    for i in range(len(archives_PL_list)):
        shutil.copy(os.path.join(diretorio_origem, archives_PL_list[i]), os.path.join(diretorio_destino, archives_PL_list[i]))
        print(f'{i+1}/{len(correspondent_archives_SHG)}')
        SHG_acc = correspondent_accumulation_SHG[i]
        image_SHG = ski.io.imread(os.path.join(diretorio_origem, correspondent_archives_SHG[i]))
        results = list()
        for c in range(image_SHG.shape[2]):
            SHG_class = interest_class_regular_noise(image_SHG[:, :, c], correspondent_accumulation_SHG[i])
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

        multi_tif_file_save (results, diretorio_destino, correspondent_archives_SHG[i])


def processar_imagens(diretorio_origem, diretorio_destino):
    try:
        # Normaliza (expande ~, resolve relativo, remove barras extras e symlinks)
        origem  = os.path.realpath(os.path.abspath(os.path.expanduser(diretorio_origem)))
        destino = os.path.realpath(os.path.abspath(os.path.expanduser(diretorio_destino)))

        if not os.path.isdir(origem):
            raise FileNotFoundError(f"Diretório de origem não existe: {origem}")

        # Erro se forem o mesmo diretório
        if origem == destino:
            raise ValueError("Erro: diretório de origem e destino não podem ser iguais.")

        # (opcional, mas recomendado) Erro se destino estiver dentro da origem
        if os.path.commonpath([origem, destino]) == origem:
            raise ValueError("Erro: o diretório de destino não pode estar dentro do diretório de origem.")

        # Só cria depois de validar
        os.makedirs(destino, exist_ok=True)

        sub_directories_names = [
            f for f in os.listdir(origem)
            if not os.path.isfile(os.path.join(origem, f))
        ]

        if len(sub_directories_names) == 0:
            list_and_process_figures(origem, destino)
            return
        else:
            for sub_directory_name in sub_directories_names:
                print('Working in {}'.format(sub_directory_name))
                sub_directory_path = os.path.join(origem, sub_directory_name)
                sub_directory_destiny = os.path.join(destino, sub_directory_name)
                os.makedirs(sub_directory_destiny, exist_ok=True)
                list_and_process_figures(sub_directory_path, sub_directory_destiny)
                print('Done!')
    except (ValueError, FileNotFoundError) as e:
        print(e)
        return  # finaliza a função
        # Se quiser abortar o programa inteiro:
        # sys.exit(str(e))

processar_imagens('/home/ronald/Schistossoma/original_data/tif', '/home/ronald/Schistossoma/CLAHE_Otsu/Final')
