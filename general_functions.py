import numpy as np
from skimage import exposure
import cv2
import re
import os

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

def rescale(data, min_value, max_value):
    try:
      data_min, data_max = data.min(), data.max()
      data_scaled = np.round((data - data_min) * (max_value - min_value) / (data_max - data_min) + min_value)
      data_scaled[data_scaled > max_value] = max_value          #Garante que após o arredondamento nenhum valor seja superior ao valor máximo definido
      return data_scaled.astype(np.uint16)
    except Exception as e:
      print('Ocorreu um erro inesperado: {}'.format(e))
      return None

def calculate_treshold_and_variance_between_classes(image):
    '''Calcula o threshold que maximiza a variância entre classes (método de Otsu),
       assim como a variância máxima, para duas classes somente'''
    intensity, count = np.unique(image, return_counts=True)
    n = len(intensity)
    probability = count/count.sum()
    cumulative_probability = probability.cumsum()
    E_i = intensity * probability
    partial_weighted_sum = E_i.cumsum()
    global_mean = E_i.sum()
    mean_C1 = np.zeros_like(intensity, dtype=float)
    cumulative_probability_C1 = np.zeros_like(intensity, dtype=float)
    mean_C0 = partial_weighted_sum/cumulative_probability
    mean_C1[:-1] = (partial_weighted_sum[-1] - partial_weighted_sum[:-1]) / (1 - cumulative_probability[:-1])
    cumulative_probability_C1[:-1] = 1 - cumulative_probability[:-1]
    variance_between_classes = cumulative_probability*(global_mean - mean_C0)**2+cumulative_probability_C1*(global_mean-mean_C1)**2
    indice_maior_valor = np.argmax(variance_between_classes)
    return intensity[indice_maior_valor], variance_between_classes[indice_maior_valor]

def find_best_parameters(image):
    clip_limits_to_try = np.arange(0.01,0.21,0.01)
    kernel_sizes_to_try = [(8,8),(12,12),(16,16),(20,20),(24,24),(28,28),(32,32)]
    clip_limits_index, kernel_sizes_index = np.meshgrid(np.arange(len(clip_limits_to_try)), np.arange(len(kernel_sizes_to_try)))
    clip_limits_index, kernel_sizes_index = clip_limits_index.flatten(), kernel_sizes_index.flatten()
    clip_limits = clip_limits_to_try[clip_limits_index]
    kernel_sizes = [kernel_sizes_to_try[i] for i in kernel_sizes_index]
    results = [
      calculate_treshold_and_variance_between_classes(
          rescale(exposure.equalize_adapthist(image, clip_limit=clip_limits[i], kernel_size=kernel_sizes[i]), min(image.ravel()), max(image.ravel()))
      )
      for i in range(len(clip_limits))
      ]
    thresholds, variances = zip(*results)
    return int(thresholds[np.argmax(variances)])

def normalized_probability_density_function(x):
    return 0.0709307107483321/(0.237703668950957*(0.278947261501954*x - 1)**2 + 1)

def generate_random_noise(N):
    # domínio discreto
    xs = np.arange(0, 596)   # 0..595 inclusive

    vals = normalized_probability_density_function(xs)

    # assegurar não-negatividade (por garantia)
    vals = np.clip(vals, 0, None).astype(np.float64)
    prob = vals/sum(vals)         #Outra vez, para garantir que a soma das probabilidades seja 1

    # gerar amostras pseudo-aleatórias
    rng = np.random.default_rng()  # muda/retira seed se não deseja determinismo

    noise = rng.choice(xs, size=N, p=prob)
    return noise.astype(np.uint16)


def interest_class_noise(figure, acc):
    threshold = find_best_parameters(figure)
    SHG_class = figure.copy()
    num_pixels = np.sum(SHG_class < threshold)
    coords = np.where(SHG_class < threshold)
    noise = acc*generate_random_noise(num_pixels)
    noise =  np.clip(noise, 0, 4095)
    SHG_class[coords] = noise
    return SHG_class


def interest_class_noise_gaussian_blur(figure, fator_multiplicativo):
    blur = cv2.GaussianBlur(figure, (65,65), 0)
    threshold=find_best_parameters(blur)
    SHG_class = figure.copy()
    for row in SHG_class:
      for i in range(len(row)):
        if row[i] < threshold:
          row[i] = 11*fator_multiplicativo
    return SHG_class
