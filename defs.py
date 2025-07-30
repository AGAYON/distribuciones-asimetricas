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

# ---------- métrica principal ----------
def metrica_ponderada(
    x,
    method="logistic",
    usar_medida_robusta=True,
    usar_transformacion_no_lineal=True,
    ajustar_por_n=True,
    use_kurtosis=False,
    incluir_moda=False,
    temperature=0.5,
    alpha=0.693, s0=1.0, p=2.0,
    shrink_c=100.0,
    clip=(0.05, 0.95)
):
    """
    Calcula una tendencia central ponderada entre media, mediana y (opcionalmente) moda,
    con pesos adaptativos según asimetría y curtosis.
    """
    x = pd.Series(x).dropna().values
    n = len(x)
    if n == 0:
        return pd.Series({
            "n": 0, "media": np.nan, "mediana": np.nan, "moda": np.nan,
            "MADN": np.nan, "bowley": np.nan, "exceso_kurtosis": np.nan,
            "peso_media": np.nan, "peso_mediana": np.nan, "peso_moda": np.nan,
            "tendencia_ponderada": np.nan
        })

    media = float(np.mean(x))
    mediana = float(np.median(x))
    moda = float(moda_kde(x)) if incluir_moda else np.nan

    escala = madn(x) if usar_medida_robusta else np.std(x)
    s1 = 0.0 if escala == 0 else abs(media - mediana) / escala
    s2 = 0.0 if escala == 0 or not incluir_moda else abs(mediana - moda) / escala
    s3 = 0.0 if escala == 0 or not incluir_moda else abs(media - moda) / escala

    if ajustar_por_n:
        s1 = shrink_s_by_n(s1, n, c=shrink_c)
        s2 = shrink_s_by_n(s2, n, c=shrink_c)
        s3 = shrink_s_by_n(s3, n, c=shrink_c)

    g2 = excess_kurtosis(x) if use_kurtosis else 0.0

    if incluir_moda:
        scores = -np.array([s1, s2, s3])
        pesos = softmax(scores, temperature=temperature)
        tendencia = pesos[0] * media + pesos[1] * mediana + pesos[2] * moda
        return pd.Series({
            "n": n,
            "media": media,
            "mediana": mediana,
            "moda": moda,
            "MADN": escala,
            "bowley": bowley_skew(x),
            "exceso_kurtosis": g2,
            "s1_median_mean": s1,
            "s2_median_mode": s2,
            "s3_mean_mode": s3,
            "peso_media": pesos[0],
            "peso_mediana": pesos[1],
            "peso_moda": pesos[2],
            "tendencia_ponderada": tendencia
        })
    else:
        if usar_transformacion_no_lineal:
            if method == "exponential":
                w_media = weight_exponential(s1, alpha=alpha)
            elif method == "logistic":
                w_media = weight_logistic(s1, s0=s0, p=p)
            else:
                raise ValueError("method debe ser 'exponential' o 'logistic'")
        else:
            w_media = max(0.0, 1.0 - s1)

        if use_kurtosis:
            w_media = adjust_by_kurtosis(w_media, g2, beta=0.25)

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
            "bowley": bowley_skew(x),
            "exceso_kurtosis": g2,
            "s_robusto": s1,
            "peso_media": w_media,
            "peso_mediana": w_mediana,
            "peso_moda": 0.0,
            "tendencia_ponderada": tendencia
        })
