from k_means_general_functions import *

'''directories_names = ['60dias', '120dias', '30dias', 'control']

for i in directories_names:
    path_original = os.path.join('/home/ronald/Schistossoma/original_data/tif', i)
    path_destiny = os.path.join('/home/ronald/Schistossoma/k_means_original_images_2_clusters', i)
    remove_PL_using_K_means_original_figures (path_original, path_destiny, save_png_figs = True, path_png_figs = os.path.join(path_destiny, 'PNG_figures'))'''
path_original = '/home/ronald/Schistossoma/estudo_subtrair_PL'
path_destiny = os.path.join(path_original,'resultado_util')
os.makedirs(path_destiny, exist_ok = True)
remove_PL_using_K_means (path_original, path_destiny, save_png_figs = False, path_png_figs = os.path.join(path_destiny, 'PNG_figures'))

