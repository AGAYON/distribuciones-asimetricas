import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# ---------- utilidades robustas ----------
def mad(x):
    """
    Calcula la desviación absoluta mediana (MAD).
    """
    med = np.median(x)
    return np.median(np.abs(x - med))

def madn(x):
    """
    Escala normalizada del MAD para aproximar desviación estándar bajo normalidad (MADN).
    """
    return 1.4826 * mad(x)

def bowley_skew(x):
    """
    Mide la asimetría robusta de Bowley usando cuartiles.
    """
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    return (q3 + q1 - 2*q2) / iqr

def excess_kurtosis(x):
    """
    Calcula el exceso de curtosis (Fisher) con corrección de sesgo.
    """
    x = np.asarray(x)
    n = len(x)
    if n < 4:
        return 0.0
    m = x.mean()
    s2 = np.mean((x - m)**2)
    if s2 == 0:
        return 0.0
    m4 = np.mean((x - m)**4)
    g2 = m4 / (s2**2) - 3.0
    return g2

# ---------- estimador de moda ----------
def moda_kde(x, bw_method='scott'):
    """
    Estima la moda suavizada de una distribución continua usando KDE.
    """
    kde = gaussian_kde(x, bw_method=bw_method)
    grid = np.linspace(np.min(x), np.max(x), 1000)
    dens = kde(grid)
    return float(grid[np.argmax(dens)])

def moda_kde_robusta(x, bw_method='scott', grid_size=2000, expansion_factor=1.2, 
                     min_peak_height=0.1, min_peak_width=0.05):
    """
    Estima la moda robusta con validación de estabilidad del pico.
    
    Parameters:
    -----------
    x : array-like
        Datos de entrada
    bw_method : str or scalar
        Método de ancho de banda para KDE
    grid_size : int
        Tamaño de la grilla para evaluación
    expansion_factor : float
        Factor de expansión del rango de datos para el grid
    min_peak_height : float
        Altura mínima relativa del pico (respecto al máximo)
    min_peak_width : float
        Ancho mínimo relativo del pico (respecto al rango)
        
    Returns:
    --------
    dict : Información completa de la moda robusta
    """
    x = np.asarray(x)
    if len(x) < 3:
        return {"moda": np.median(x), "altura_relativa": 0.0, "ancho_pico": 0.0, "es_robusta": False}
    
    # KDE con grid expandido
    kde = gaussian_kde(x, bw_method=bw_method)
    x_min, x_max = np.min(x), np.max(x)
    rango = x_max - x_min
    grid_min = x_min - (expansion_factor - 1) * rango / 2
    grid_max = x_max + (expansion_factor - 1) * rango / 2
    grid = np.linspace(grid_min, grid_max, grid_size)
    dens = kde(grid)
    
    # Encontrar el pico principal
    max_idx = np.argmax(dens)
    max_density = dens[max_idx]
    moda_value = grid[max_idx]
    
    # Calcular altura relativa del pico
    altura_relativa = max_density / np.mean(dens) if np.mean(dens) > 0 else 0.0
    
    # Estimar ancho del pico (ancho a media altura)
    half_max = max_density / 2
    left_idx = max_idx
    right_idx = max_idx
    
    # Buscar hacia la izquierda
    while left_idx > 0 and dens[left_idx] > half_max:
        left_idx -= 1
    
    # Buscar hacia la derecha
    while right_idx < len(dens) - 1 and dens[right_idx] > half_max:
        right_idx += 1
    
    ancho_pico = (grid[right_idx] - grid[left_idx]) / rango if rango > 0 else 0.0
    
    # Validar robustez del pico
    es_robusta = (altura_relativa >= min_peak_height and 
                  ancho_pico >= min_peak_width and
                  grid_min <= moda_value <= grid_max)
    
    return {
        "moda": float(moda_value),
        "altura_relativa": float(altura_relativa),
        "ancho_pico": float(ancho_pico),
        "es_robusta": bool(es_robusta),
        "densidad_maxima": float(max_density)
    }

# ---------- mapeos de sensibilidad ----------
def weight_exponential(s, alpha=0.693):
    """
    Mapeo exponencial de sesgo a peso de la media (alpha ~ ln(2)).
    """
    w_mean = np.exp(-alpha * s)
    return float(np.clip(w_mean, 0.0, 1.0))

def weight_logistic(s, s0=1.0, p=2.0):
    """
    Mapeo logístico del sesgo a peso de la media.
    s0 define el punto medio (peso=0.5), p controla la pendiente.
    """
    w_mean = 1.0 / (1.0 + (s / max(s0, 1e-12))**p)
    return float(np.clip(w_mean, 0.0, 1.0))

def weight_linear(s, s_max=2.0):
    """
    Mapeo lineal del sesgo a peso de la media.
    s_max define el punto donde el peso se vuelve 0.
    """
    w_mean = max(0.0, 1.0 - s / max(s_max, 1e-12))
    return float(np.clip(w_mean, 0.0, 1.0))

def adjust_by_kurtosis(w_mean, g2, beta=0.25):
    """
    Penaliza el peso de la media según el exceso de curtosis.
    """
    factor = np.exp(-beta * max(0.0, g2))
    return float(np.clip(w_mean * factor, 0.0, 1.0))

def shrink_s_by_n(s, n, c=100.0):
    """
    Ajusta el valor de sesgo s según el tamaño muestral para estabilización.
    """
    return float(s * np.sqrt(n / (n + c)))

# ---------- softmax para pesos ----------
def softmax(vals, temperature=1.0):
    """
    Aplica softmax con temperatura a una lista de valores.
    """
    vals = np.array(vals) / temperature
    exp_vals = np.exp(vals - np.max(vals))  # estabilidad numérica
    return exp_vals / np.sum(exp_vals)

def convex_weights(distances, method='inverse_distance', alpha=2.0):
    """
    Calcula pesos convexos basados en distancias.
    
    Parameters:
    -----------
    distances : array-like
        Distancias entre medidas de tendencia central
    method : str
        Método para calcular pesos ('inverse_distance', 'exponential', 'polynomial')
    alpha : float
        Parámetro de control de la función de peso
        
    Returns:
    --------
    np.ndarray : Pesos normalizados que suman 1
    """
    distances = np.array(distances)
    distances = distances + 1e-12  # Evitar división por cero
    
    if method == 'inverse_distance':
        weights = 1 / (distances ** alpha)
    elif method == 'exponential':
        weights = np.exp(-alpha * distances)
    elif method == 'polynomial':
        weights = 1 / (1 + distances ** alpha)
    else:
        raise ValueError("method debe ser 'inverse_distance', 'exponential' o 'polynomial'")
    
    # Normalizar para que sumen 1
    return weights / np.sum(weights)

# ---------- métrica principal ----------
def metrica_ponderada(
    x,
    method="logistic",
    usar_medida_robusta=True,
    usar_transformacion_no_lineal=True,
    ajustar_por_n=True,
    use_kurtosis=False,
    use_bowley=False,
    incluir_moda=False,
    moda_robusta=False,
    weight_method='softmax',  # 'softmax' o 'convex'
    convex_method='inverse_distance',
    temperature=0.5,
    alpha=0.693, s0=1.0, p=2.0, s_max=2.0,
    shrink_c=100.0,
    clip=(0.05, 0.95),
    # Parámetros para moda robusta
    bw_method='scott', grid_size=2000, expansion_factor=1.2,
    min_peak_height=0.1, min_peak_width=0.05
):
    """
    Calcula una tendencia central ponderada entre media, mediana y (opcionalmente) moda,
    con pesos adaptativos según asimetría, curtosis y asimetría robusta de Bowley.
    Parámetros:
    - Moda robusta con validación de estabilidad del pico
    - Métodos de ponderación: softmax y convex weights
    - Control de robustez de la moda basado en altura y ancho del pico
    """
    x = pd.Series(x).dropna().values
    n = len(x)
    if n == 0:
        return pd.Series({
            "n": 0, "media": np.nan, "mediana": np.nan, "moda": np.nan,
            "MADN": np.nan, "bowley": np.nan, "exceso_kurtosis": np.nan,
            "peso_media": np.nan, "peso_mediana": np.nan, "peso_moda": np.nan,
            "tendencia_ponderada": np.nan, "moda_robusta": False, 
            "altura_pico": np.nan, "ancho_pico": np.nan
        })

    media = float(np.mean(x))
    mediana = float(np.median(x))
    
    # Calcular moda (simple o robusta)
    if incluir_moda:
        if moda_robusta:
            moda_info = moda_kde_robusta(x, bw_method=bw_method, grid_size=grid_size,
                                       expansion_factor=expansion_factor,
                                       min_peak_height=min_peak_height,
                                       min_peak_width=min_peak_width)
            moda = moda_info["moda"]
            es_moda_robusta = moda_info["es_robusta"]
            altura_pico = moda_info["altura_relativa"]
            ancho_pico = moda_info["ancho_pico"]
        else:
            moda = float(moda_kde(x, bw_method=bw_method))
            es_moda_robusta = True  # Asumimos que es robusta si no se valida
            altura_pico = 1.0
            ancho_pico = 1.0
    else:
        moda = np.nan
        es_moda_robusta = False
        altura_pico = np.nan
        ancho_pico = np.nan

    escala = madn(x) if usar_medida_robusta else np.std(x)
    s1 = 0.0 if escala == 0 else abs(media - mediana) / escala
    s2 = 0.0 if escala == 0 or not incluir_moda else abs(mediana - moda) / escala
    s3 = 0.0 if escala == 0 or not incluir_moda else abs(media - moda) / escala

    # Calcular asimetría de Bowley para ajustes adicionales
    bowley_asimetria = bowley_skew(x)
    
    if ajustar_por_n:
        s1 = shrink_s_by_n(s1, n, c=shrink_c)
        s2 = shrink_s_by_n(s2, n, c=shrink_c)
        s3 = shrink_s_by_n(s3, n, c=shrink_c)

    g2 = excess_kurtosis(x) if use_kurtosis else 0.0

    if incluir_moda:
        # Si la moda no es robusta, darle menos peso
        if moda_robusta and not es_moda_robusta:
            # Reducir significativamente las distancias que involucran la moda
            s2 = s2 * 2.0  # Penalizar distancia mediana-moda
            s3 = s3 * 2.0  # Penalizar distancia media-moda
        
        if weight_method == 'softmax':
            scores = -np.array([s1, s2, s3])
            pesos = softmax(scores, temperature=temperature)
        elif weight_method == 'convex':
            distances = np.array([s1, s2, s3])
            pesos = convex_weights(distances, method=convex_method, alpha=temperature)
        else:
            raise ValueError("weight_method debe ser 'softmax' o 'convex'")
            
        tendencia = pesos[0] * media + pesos[1] * mediana + pesos[2] * moda
        return pd.Series({
            "n": n,
            "media": media,
            "mediana": mediana,
            "moda": moda,
            "MADN": escala,
            "bowley": bowley_asimetria,
            "exceso_kurtosis": g2,
            "s1_median_mean": s1,
            "s2_median_mode": s2,
            "s3_mean_mode": s3,
            "peso_media": pesos[0],
            "peso_mediana": pesos[1],
            "peso_moda": pesos[2],
            "tendencia_ponderada": tendencia,
            "moda_robusta": es_moda_robusta,
            "altura_pico": altura_pico,
            "ancho_pico": ancho_pico
        })
    else:
        if usar_transformacion_no_lineal:
            if method == "exponential":
                w_media = weight_exponential(s1, alpha=alpha)
            elif method == "logistic":
                w_media = weight_logistic(s1, s0=s0, p=p)
            elif method == "linear":
                w_media = weight_linear(s1, s_max=s_max)
            else:
                raise ValueError("method debe ser 'exponential', 'logistic' o 'linear'")
        else:
            w_media = max(0.0, 1.0 - s1)

        if use_kurtosis:
            w_media = adjust_by_kurtosis(w_media, g2, beta=0.25)
            
        # Ajuste adicional usando la asimetría de Bowley
        if use_bowley:
            bowley_factor = np.exp(-0.2 * abs(bowley_asimetria))  # Penalización por asimetría de Bowley
            w_media = w_media * bowley_factor

        lo, hi = clip
        w_media = float(np.clip(w_media, lo, hi))
        w_mediana = 1.0 - w_media
        tendencia = w_media * media + w_mediana * mediana

        return pd.Series({
            "n": n,
            "media": media,
            "mediana": mediana,
            "moda": np.nan,
            "MADN": escala,
            "bowley": bowley_asimetria,
            "exceso_kurtosis": g2,
            "s_robusto": s1,
            "peso_media": w_media,
            "peso_mediana": w_mediana,
            "peso_moda": 0.0,
            "tendencia_ponderada": tendencia,
            "moda_robusta": False,
            "altura_pico": np.nan,
            "ancho_pico": np.nan
        })
