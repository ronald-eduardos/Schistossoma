import numpy as np
import tifffile

def multi_tif_file_save (list_with_figures, destiny_path, name_of_new_figure):
    list_with_figures_array = np.stack(list_with_figures)
    # Converter para shape (T=1, Z=3, C=1, Y=512, X=512)
    stack_4d = list_with_figures_array[:, np.newaxis, :, :]             # (Z, 1, Y, X)    #Adiciona eixo do canal ao stack
    stack_5d = stack_4d[np.newaxis, ...]              # (1, Z, 1, Y, X)         #Adiciona eixo do tempo ao stack
    # Salvar como multi-TIFF
    repository_and_image_path = destiny_path+'/'+ name_of_new_figure

    tifffile.imwrite(
    repository_and_image_path,
    stack_5d,
    photometric='minisblack',  # escala de cinza (0=preto, 255=branco)
    metadata={'axes': 'TZCYX'},  # convenção usada por tifffile
    resolution=(72, 72),       # DPI (igual ao do arquivo que você enviou)
    imagej=True                # Adiciona cabeçalhos compatíveis com ImageJ/Zeiss
    )
    return None
