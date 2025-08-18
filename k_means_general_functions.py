'''Importa funções essenciais'''

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage as ski
import shutil
from skimage.util import view_as_windows
from sklearn.cluster import KMeans
from save_multitif_file import *

'''Recebe como entrada o caminho dos arquivos originais
   Gera listas com nomes dos arquivos PL, correspondentes
   nomes dos arquivos SHG e as respectivas acumulações'''
def generate_list_PL_and_SHG_archives_names_and_accumulations(path):
    archives_PL_list = [f for f in os.listdir(path)
                        if os.path.isfile(os.path.join(path, f)) and '.tif' in f.lower()
                        and 'pl' in f.lower()]
    correspondent_archives_SHG = []
    correspondent_accumulation_PL = []
    correspondent_accumulation_SHG = []
    for f in archives_PL_list:
        match_PL = re.search(r'acc(\d+)', f, flags=re.IGNORECASE)
        correspondent_accumulation_PL.append(int(match_PL.group(1)) if match_PL else None)
        base_name = f.split('-pl')[0]
        correspondent_name_SHG = [g for g in os.listdir(path)
                                  if os.path.isfile(os.path.join(path, g)) and '.tif' in f.lower()
                                  and base_name in g and 'shg' in g.lower()][0]
        match_SHG = re.search(r'acc(\d+)', correspondent_name_SHG, flags=re.IGNORECASE)
        correspondent_accumulation_SHG.append(int(match_SHG.group(1)) if match_SHG else None)
        correspondent_archives_SHG.append(correspondent_name_SHG)
    return (archives_PL_list, correspondent_archives_SHG, correspondent_accumulation_PL,
            correspondent_accumulation_SHG)

'''Funcões para exibições gráficas das imagens e histogramas'''

def color_map(figure, color, max_value):
    if color == 'pure_green':
        norm_green = figure / max_value
        puro = np.stack((np.zeros_like(norm_green), norm_green, np.zeros_like(norm_green)), axis=-1)
    if color == 'pure_red':
        norm_red = figure / max_value
        puro = np.stack((norm_red, np.zeros_like(norm_red), np.zeros_like(norm_red)), axis=-1)
    return puro

def calculate_pure_colors (fig_1, fig_2,
                           color_map_1, color_map_2,
                           max_intensity_1, max_intensity_2):
    puro_1 = [color_map(fig_1, 'pure_green', max_intensity_1) if color_map_1 == 'pure_green'
                  else color_map(fig_1, 'pure_red', max_intensity_1)][0]

    puro_2 = [color_map(fig_2, 'pure_green', max_intensity_2) if color_map_2 == 'pure_green'
                  else color_map(fig_2, 'pure_red', max_intensity_2)][0]
    return puro_1, puro_2

def exib_figures(fig_1, fig_2,
                 color_map_1, color_map_2,
                 max_intensity_1, max_intensity_2,
                 title_fig_1, title_fig_2,
                 path_and_name_figure = None):

    puro_1, puro_2 = calculate_pure_colors(fig_1, fig_2,
                                           color_map_1, color_map_2,
                                           max_intensity_1, max_intensity_2)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    ax[0].imshow(puro_1)
    ax[0].set_title(title_fig_1)
    ax[0].axis('off')

    ax[1].imshow(puro_2)
    ax[1].set_title(title_fig_2)
    ax[1].axis('off')

    plt.tight_layout()
    if path_and_name_figure:
        plt.savefig(path_and_name_figure, transparent=True)
    plt.show()

def plot_histograms_side_by_side(img1, img2,
                                 title1, title2,
                                 overall_title = '',
                                 path_save = None):

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    ax[0].hist(img1.ravel(), bins=512, color='red')
    ax[0].set_title(title1)
    ax[0].set_xlabel('Intensidade')
    ax[0].set_ylabel('Contagem (log)')
    #ax[0].set_yscale('log')

    ax[1].hist(img2.ravel(), bins=512, color='green')
    ax[1].set_title(title2)
    ax[1].set_xlabel('Intensidade')
    #ax[1].set_yscale('log')

    if overall_title:
        fig.suptitle(overall_title, fontsize=14)

    plt.tight_layout()
    if path_save:
        plt.savefig(path_save, transparent = True)
    plt.show()


'''Encontra a janela com maior similaridade entre as imagens de PL e SHG.'''
def find_most_similar_patch(PL, SHG, window_size=(64, 64), stride=16):
    """Encontra a janela mais semelhante entre PL e SHG"""
    h, w = PL.shape
    win_h, win_w = window_size

    min_diff = float('inf')
    best_coords = (0, 0)

    for i in range(0, h - win_h + 1, stride):
        for j in range(0, w - win_w + 1, stride):
            patch_PL = PL[i:i+win_h, j:j+win_w]
            patch_SHG = SHG[i:i+win_h, j:j+win_w]

            # Métrica de similaridade: média da diferença absoluta
            diff = np.mean(np.abs(patch_PL - patch_SHG))

            if diff < min_diff:
                min_diff = diff
                best_coords = (i, j)

    return best_coords, window_size


def estimate_PL_SHG_ratio(PL, SHG, window_size=(64, 64), stride=16,
                          show_selection = False, save_selection = False,
                          path_and_name_to_save = None):
    coords, win_size = find_most_similar_patch(PL, SHG, window_size, stride)
    i, j = coords
    win_h, win_w = win_size

    # Verificação de coerência entre save_selection e path
    try:
        if save_selection and path_and_name_to_save is None:
            raise ValueError("Para salvar a seleção, o parâmetro 'path_and_name_to_save' deve ser fornecido.")
    except ValueError as e:
        print(f"[Aviso] {e}")
        save_selection = False  # ignora o salvamento

    if show_selection:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        pure_1, pure_2 = calculate_pure_colors(PL, SHG,
                                           'pure_red', 'pure_green',
                                           np.max(PL), np.max(SHG))
        # Exibe a imagem PL com o retângulo
        axes[0].imshow(pure_1)
        rect_pl = patches.Rectangle((j, i), win_w, win_h, linewidth=1, edgecolor='b', facecolor='none')
        axes[0].add_patch(rect_pl)
        axes[0].set_title('Imagem PL com Seleção')
        axes[0].axis('off')

        # Exibe a imagem SHG com o retângulo
        axes[1].imshow(pure_2)
        rect_shg = patches.Rectangle((j, i), win_w, win_h, linewidth=1, edgecolor='b', facecolor='none')
        axes[1].add_patch(rect_shg)
        axes[1].set_title('Imagem SHG com Seleção')
        axes[1].axis('off')

        plt.tight_layout()
        if save_selection:
            plt.savefig(path_and_name_to_save)
        plt.show(block=True)

    patch_PL = PL[i:i+win_h, j:j+win_w]
    patch_SHG = SHG[i:i+win_h, j:j+win_w]

    mean_PL = np.mean(patch_PL)
    mean_SHG = np.mean(patch_SHG)

    if mean_SHG == 0:
        return np.nan  # Evita divisão por zero
    else:
        return mean_PL / mean_SHG

def remove_PL_using_K_means (path_original, path_destiny, save_png_figs = False, path_png_figs = None):
    names_PL, names_SHG, acc_PL, acc_SHG = generate_list_PL_and_SHG_archives_names_and_accumulations(path_original)
    os.makedirs(path_destiny, exist_ok = True)
    # Verificação de coerência entre save_png_figs e path_png_figs
    try:
        if save_png_figs and path_png_figs is None:
            raise ValueError("Para salvar a imagem PNG, o parâmetro 'path_png_figs' deve ser fornecido.")
    except ValueError as e:
        print(f"[Aviso] {e}")
        save_png_figs = False  # ignora o salvamento

    for indice_field in range(len(names_PL)):
        results = []
        for j in range(3):
            #Abre as imagens PL e SHG e as normaliza, segundo as respectivas acumulações
            PL_image = ski.io.imread(os.path.join(path_original, names_PL[indice_field]))[0, :, :, j]
            SHG_image = ski.io.imread(os.path.join(path_original, names_SHG[indice_field]))[:, :, j]
            normalized_PL_image = PL_image/acc_PL[indice_field]
            normalized_SHG_image =SHG_image /acc_SHG[indice_field]

            #Calcula a razão entre as intensidades médias por acumulação para a área mais similar entre as imagens de PL e SHG
            ratio = estimate_PL_SHG_ratio(normalized_PL_image, normalized_SHG_image, window_size=(32, 32), stride=1, show_selection = False)

            #Subtrai da imagem de SHG uma fração da imagem de PL, segundo a razão entre as intensidades médias por acumulação
            SHG_with_PL_subtracted = acc_SHG[indice_field]*(normalized_SHG_image-normalized_PL_image/ratio)
            SHG_with_PL_subtracted = np.clip(np.round(SHG_with_PL_subtracted), 0, None).astype(int)

            #Calcula o KMeans e os respectivos rótulos
            kmeans = KMeans(n_clusters=3).fit(SHG_with_PL_subtracted.reshape(-1, 1))
            labels = kmeans.labels_.reshape(SHG_with_PL_subtracted.shape)

            # Identifica qual cluster é o mais brilhante
            cluster_means = kmeans.cluster_centers_.flatten()
            bright_cluster = np.argmax(cluster_means)

            # Cria imagem binária: True para pixels do cluster mais brilhante
            binary_kmeans = (labels == bright_cluster)

            # Cria a nova imagem com a mesma forma da original
            SHG_figure_PL_removed = np.where(binary_kmeans, SHG_image, 11 * acc_SHG[indice_field])

            #Salva uma das camadas da imagem em PNG
            if j == 1:
                if save_png_figs:
                    os.makedirs(path_png_figs, exist_ok = True)
                    colored_original, colored_filtered = calculate_pure_colors(SHG_image, SHG_figure_PL_removed,
                                                                               'pure_green', 'pure_green',
                                                                               np.max(SHG_image), np.max(SHG_figure_PL_removed))

                    # Salva como imagens RGB coloridas
                    ski.io.imsave(os.path.join(path_png_figs, names_SHG[indice_field].replace('.tif', '_Original.png')),
                                  (colored_original * 255).astype(np.uint8))

                    ski.io.imsave(os.path.join(path_png_figs, names_SHG[indice_field].replace('.tif', '_Filtered.png')),
                                  (colored_filtered * 255).astype(np.uint8))

            #Adiciona resultado ao results
            results.append(SHG_figure_PL_removed)
        multi_tif_file_save (results, path_destiny, names_SHG[indice_field])
        shutil.copy(os.path.join(path_original, names_PL[indice_field]), os.path.join(path_destiny, names_PL[indice_field]))
    return None

def remove_PL_using_K_means_original_figures (path_original, path_destiny, save_png_figs = False, path_png_figs = None):
    names_PL, names_SHG, acc_PL, acc_SHG = generate_list_PL_and_SHG_archives_names_and_accumulations(path_original)
    os.makedirs(path_destiny, exist_ok = True)
    # Verificação de coerência entre save_png_figs e path_png_figs
    try:
        if save_png_figs and path_png_figs is None:
            raise ValueError("Para salvar a imagem PNG, o parâmetro 'path_png_figs' deve ser fornecido.")
    except ValueError as e:
        print(f"[Aviso] {e}")
        save_png_figs = False  # ignora o salvamento

    for indice_field in range(len(names_PL)):
        results = []
        for j in range(3):
            #Abre as imagens PL e SHG e as normaliza, segundo as respectivas acumulações
            PL_image = ski.io.imread(os.path.join(path_original, names_PL[indice_field]))[0, :, :, j]
            SHG_image = ski.io.imread(os.path.join(path_original, names_SHG[indice_field]))[:, :, j]
            #normalized_PL_image = PL_image/acc_PL[indice_field]
            #normalized_SHG_image =SHG_image /acc_SHG[indice_field]

            #Calcula a razão entre as intensidades médias por acumulação para a área mais similar entre as imagens de PL e SHG
            #ratio = estimate_PL_SHG_ratio(normalized_PL_image, normalized_SHG_image, window_size=(32, 32), stride=1, show_selection = False)

            #Subtrai da imagem de SHG uma fração da imagem de PL, segundo a razão entre as intensidades médias por acumulação
            #SHG_with_PL_subtracted = acc_SHG[indice_field]*(normalized_SHG_image-normalized_PL_image/ratio)
            #SHG_with_PL_subtracted = np.clip(np.round(SHG_with_PL_subtracted), 0, None).astype(int)

            #Calcula o KMeans e os respectivos rótulos
            kmeans = KMeans(n_clusters=2).fit(SHG_image.reshape(-1, 1))
            labels = kmeans.labels_.reshape(SHG_image.shape)

            # Identifica qual cluster é o mais brilhante
            cluster_means = kmeans.cluster_centers_.flatten()
            bright_cluster = np.argmax(cluster_means)

            # Cria imagem binária: True para pixels do cluster mais brilhante
            binary_kmeans = (labels == bright_cluster)

            # Cria a nova imagem com a mesma forma da original
            SHG_figure_PL_removed = np.where(binary_kmeans, SHG_image, 11 * acc_SHG[indice_field])

            #Salva uma das camadas da imagem em PNG
            if j == 1:
                if save_png_figs:
                    os.makedirs(path_png_figs, exist_ok = True)
                    colored_original, colored_filtered = calculate_pure_colors(SHG_image, SHG_figure_PL_removed,
                                                                               'pure_green', 'pure_green',
                                                                               np.max(SHG_image), np.max(SHG_figure_PL_removed))

                    # Salva como imagens RGB coloridas
                    ski.io.imsave(os.path.join(path_png_figs, names_SHG[indice_field].replace('.tif', '_Original.png')),
                                  (colored_original * 255).astype(np.uint8))

                    ski.io.imsave(os.path.join(path_png_figs, names_SHG[indice_field].replace('.tif', '_Filtered.png')),
                                  (colored_filtered * 255).astype(np.uint8))

            #Adiciona resultado ao results
            results.append(SHG_figure_PL_removed)
        multi_tif_file_save (results, path_destiny, names_SHG[indice_field])
        shutil.copy(os.path.join(path_original, names_PL[indice_field]), os.path.join(path_destiny, names_PL[indice_field]))
    return None
