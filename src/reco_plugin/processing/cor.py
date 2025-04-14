import cupy as cp
from tqdm import tqdm

def process_try_center_of_rotation(self, widget):
    """
    Process the center of rotation based on the widget values.
    """
    print(f"Processing Center of Rotation")
    try:
        self.center_of_rotation = int(widget.center_of_rotation.value())
        print(f"Center of Rotation: {self.center_of_rotation}")
    except Exception as e:
        print(f"Error processing center of rotation: {e}")

def process_precise_local(self, widget):
    """
    Process the precise local based on the widget values.
    """
    print(f"Processing Precise Local")
    try:
        self.precise_local = int(widget.precise_local.value())
        print(f"Precise Local: {self.precise_local}")
    except Exception as e:
        print(f"Error processing precise local: {e}")

def calc_cor(projs):
    """
    projs_cp: projections [angles, hauteur, largeur] (CuPy array)
    """
    if projs.ndim != 3:
        theta, nx = projs.shape
        ny = 1
    else :
        theta, ny, nx = projs.shape

    start = 0
    stop = ny
    step = 10
    cor = cp.zeros((stop - start + step - 1) // step, dtype=cp.float32)  # Stocke les centres pour chaque ligne
    plot_data = []

    i = 0
    for y in tqdm(range(start, stop, step), desc="Recherche du COR par ligne"):
        # sinogramme d'une ligne horizontale
        sino1 = cp.asarray(projs[:theta // 2, y, ::-1])  # Première moitié inversée
        sino2 = cp.asarray(projs[theta // 2:, y, :])     # Deuxième moitié

        errors = cp.zeros(nx - 1, dtype=cp.float16)  # Stocke les erreurs pour chaque décalage
        for shift in range(1, nx):
            t1 = sino1[:, -shift:]
            t2 = sino2[:, :shift]
            if t1.shape != t2.shape:
                continue
            mse = cp.mean((t1 - t2) ** 2)
            errors[shift - 1] = mse

        best_shift = cp.argmin(errors)
        plot_data.append(errors.get())  # Convertit en NumPy pour le traçage
        cor[i] = (best_shift) / 2  # Position estimée du COR
        i += 1

    return cor.get(), plot_data  # Convertit `cor` en NumPy pour l'utilisation ultérieure