import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# =========================================================
# utilidades robustas
# =========================================================
def mad(x):
    """Desviación absoluta a la mediana (MAD)."""
    med = np.median(x)
    return np.median(np.abs(x - med))

def madn(x):
    """MAD normalizado para aproximar la DE bajo normalidad."""
    return 1.4826 * mad(x)

def bowley_skew(x):
    """Asimetría robusta de Bowley (cuartiles)."""
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    return (q3 + q1 - 2*q2) / iqr

def excess_kurtosis(x):
    """Exceso de curtosis de Fisher (g2)."""
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

def trimmed_mean(x, prop=0.1):
    """Media recortada simétrica (p. ej. 10%)."""
    x = np.sort(np.asarray(x))
    n = len(x)
    k = int(np.floor(prop * n))
    if n - 2*k <= 0:
        return float(np.mean(x))
    return float(np.mean(x[k:n-k]))

def symmetry_index(x, ps=(0.1, 0.2, 0.3, 0.4)):
    """
    Índice de simetría: promedio de |Q(p)+Q(1-p)-2*Q(0.5)| normalizado por MADN.
    Valor pequeño -> alta simetría.
    """
    x = np.asarray(x)
    if len(x) < 5:
        return 0.0
    qs = np.percentile(x, [50] + [100*p for p in ps] + [100*(1-p) for p in ps])
    q50 = qs[0]
    vals = []
    for i, p in enumerate(ps, start=1):
        ql = qs[i]
        qr = qs[i+len(ps)]
        vals.append(abs(ql + qr - 2*q50))
    denom = madn(x)
    if denom == 0:
        return 0.0
    return float(np.mean(vals) / denom)

# =========================================================
# estimadores de moda
# =========================================================
def moda_kde(x, bw_method='scott'):
    """Moda suavizada con KDE."""
    x = np.asarray(x)
    kde = gaussian_kde(x, bw_method=bw_method)
    grid = np.linspace(np.min(x), np.max(x), 1000)
    dens = kde(grid)
    return float(grid[np.argmax(dens)])

def moda_kde_robusta(x, bw_method='scott', grid_size=2000, usar_expansion=False,
                     columna_expansion=None, expansion_data=None, expansion_factor=1.2,
                     min_peak_height=0.1, min_peak_width=0.05):
    """
    Moda robusta con validación del pico (altura relativa y ancho a media altura).
    """
    x = np.asarray(x)
    if len(x) < 3:
        return {"moda": np.median(x), "altura_relativa": 0.0, "ancho_pico": 0.0, "es_robusta": False}

    # Pesos opcionales (factores de expansión)
    if usar_expansion and columna_expansion is not None and expansion_data is not None:
        if hasattr(expansion_data, columna_expansion):  # DataFrame
            factores = expansion_data[columna_expansion].values
        elif isinstance(expansion_data, dict) and columna_expansion in expansion_data:
            factores = np.array(expansion_data[columna_expansion])
        else:
            factores = np.ones(len(x))
        if len(factores) != len(x):
            factores = np.ones(len(x))
        kde = gaussian_kde(x, bw_method=bw_method, weights=factores/np.sum(factores))
        expansion_range = np.mean(factores)
    else:
        kde = gaussian_kde(x, bw_method=bw_method)
        expansion_range = expansion_factor

    # Grid expandido
    x_min, x_max = np.min(x), np.max(x)
    rango = x_max - x_min
    grid_min = x_min - (expansion_range - 1) * rango / 2
    grid_max = x_max + (expansion_range - 1) * rango / 2
    grid = np.linspace(grid_min, grid_max, grid_size)
    dens = kde(grid)

    # Pico principal
    max_idx = np.argmax(dens)
    max_density = dens[max_idx]
    moda_value = grid[max_idx]

    # Altura relativa
    altura_relativa = max_density / np.mean(dens) if np.mean(dens) > 0 else 0.0

    # Ancho a media altura
    half_max = max_density / 2
    left_idx = max_idx
    right_idx = max_idx
    while left_idx > 0 and dens[left_idx] > half_max:
        left_idx -= 1
    while right_idx < len(dens) - 1 and dens[right_idx] > half_max:
        right_idx += 1
    ancho_pico = (grid[right_idx] - grid[left_idx]) / rango if rango > 0 else 0.0

    # Robustez del pico
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

# =========================================================
# mapeos y ajustes
# =========================================================
def weight_exponential(s, alpha=0.693):
    """Mapeo exponencial de sesgo a peso de la media (alpha ~ ln(2))."""
    w_mean = np.exp(-alpha * s)
    return float(np.clip(w_mean, 0.0, 1.0))

def weight_logistic(s, s0=1.0, p=2.0):
    """Mapeo logístico: s0 punto medio (0.5), p pendiente."""
    w_mean = 1.0 / (1.0 + (s / max(s0, 1e-12))**p)
    return float(np.clip(w_mean, 0.0, 1.0))

def weight_linear(s, s_max=2.0):
    """Mapeo lineal: s_max donde el peso se vuelve 0."""
    w_mean = max(0.0, 1.0 - s / max(s_max, 1e-12))
    return float(np.clip(w_mean, 0.0, 1.0))

def adjust_by_kurtosis(w_mean, g2, beta=0.25):
    """Penaliza la media por exceso de curtosis (solo castigo)."""
    factor = np.exp(-beta * max(0.0, g2))
    return float(np.clip(w_mean * factor, 0.0, 1.0))

def adjust_by_kurtosis_bell(w_mean, g2, reward_window=0.5, reward_gain=0.08,
                            beta_neg=0.15, beta_pos=0.25):
    """
    Ajuste 'en campana' centrado en g2=0:
      - Recompensa leve cerca de 0 (mesocúrtico): factor ~ 1 + reward_gain.
      - Penaliza cuando |g2| > reward_window (colas raras o platic/lepto fuertes).
    """
    g2_abs = abs(g2)
    # Recompensa cerca de 0
    if g2_abs <= reward_window:
        factor = 1.0 + reward_gain * (1.0 - g2_abs / max(reward_window, 1e-12))
    else:
        # Penalización fuera de la ventana, diferenciando signo si se desea
        # (por simplicidad aplicamos misma beta a ambos lados, pero permitimos params distintos)
        over = g2_abs - reward_window
        beta = beta_pos if g2 >= 0 else beta_neg
        factor = np.exp(-beta * over)
    return float(np.clip(w_mean * factor, 0.0, 1.0))

def shrink_s_by_n(s, n, c=100.0):
    """Ajusta s por tamaño muestral para estabilización (versión 'conservadora')."""
    return float(s * np.sqrt(n / (n + c)))

def shrink_s_good_by_n(s, n, n_ref=300):
    """Ajuste alternativo (más agresivo) para escenarios 'buenos' (simetría/mesocúrticos)."""
    return float(s * np.sqrt(n_ref / (n + n_ref)))

# =========================================================
# pesos
# =========================================================
def softmax(vals, temperature=1.0):
    """Softmax con temperatura."""
    vals = np.array(vals) / max(temperature, 1e-12)
    exp_vals = np.exp(vals - np.max(vals))
    return exp_vals / np.sum(exp_vals)

def convex_weights(distances, method='inverse_distance', alpha=2.0):
    """
    Pesos convexos a partir de distancias (más cerca => más peso).
    method: 'inverse_distance', 'exponential', 'polynomial'
    """
    distances = np.array(distances) + 1e-12
    if method == 'inverse_distance':
        weights = 1 / (distances ** alpha)
    elif method == 'exponential':
        weights = np.exp(-alpha * distances)
    elif method == 'polynomial':
        weights = 1 / (1 + distances ** alpha)
    else:
        raise ValueError("method debe ser 'inverse_distance', 'exponential' o 'polynomial'")
    return weights / np.sum(weights)

# =========================================================
# métrica principal (v2)
# =========================================================
def metrica_ponderada_v2(
    x,
    # mapeo base
    method="logistic",                    # 'exponential' | 'logistic' | 'linear'
    usar_medida_robusta=True,
    usar_transformacion_no_lineal=True,
    ajustar_por_n=True,
    # ajustes
    use_kurtosis=False,
    use_kurtosis_bell=False,              # NUEVO: recompensa cerca de g2≈0
    use_bowley=False,
    # centros
    incluir_moda=False,
    moda_robusta=False,
    incluir_trimmed=False,                # NUEVO: incluir media recortada
    trimmed_prop=0.1,                     # NUEVO: % de recorte simétrico
    # pesos
    weight_method='softmax',              # 'softmax' | 'convex'
    convex_method='inverse_distance',
    temperature=0.5,
    # hiperparámetros mapeo
    alpha=0.693,
    s0=1.0,
    p=2.0,
    s_max=2.0,
    shrink_c=100.0,
    clip=(0.05, 0.95),
    # moda robusta
    bw_method='scott',
    grid_size=2000,
    usar_expansion=False,
    columna_expansion=None,
    expansion_data=None,
    expansion_factor=1.2,
    min_peak_height=0.1,
    min_peak_width=0.05,
    # políticas pro-media
    w_media_floor=None                    # NUEVO: piso mínimo opcional para w_media
):
    """
    Tendencia central ponderada entre:
      - media,
      - mediana,
      - (opcional) moda,
      - (opcional) media recortada.
    Con reglas mejoradas para premiar escenarios simétricos/mesocúrticos y n grande.
    """
    x = pd.Series(x).dropna().values
    n = len(x)
    if n == 0:
        return pd.Series({
            "n": 0, "media": np.nan, "mediana": np.nan, "moda": np.nan, "trimmed": np.nan,
            "MADN": np.nan, "bowley": np.nan, "exceso_kurtosis": np.nan,
            "peso_media": np.nan, "peso_mediana": np.nan, "peso_moda": np.nan, "peso_trimmed": np.nan,
            "tendencia_ponderada": np.nan, "moda_robusta": False, "altura_pico": np.nan, "ancho_pico": np.nan
        })

    media = float(np.mean(x))
    mediana = float(np.median(x))
    trimmed = float(trimmed_mean(x, trimmed_prop)) if incluir_trimmed else np.nan

    # Calcular moda (simple o robusta)
    if incluir_moda:
        if moda_robusta:
            moda_info = moda_kde_robusta(
                x, bw_method=bw_method, grid_size=grid_size,
                usar_expansion=usar_expansion, columna_expansion=columna_expansion,
                expansion_data=expansion_data, expansion_factor=expansion_factor,
                min_peak_height=min_peak_height, min_peak_width=min_peak_width
            )
            moda = moda_info["moda"]
            es_moda_robusta = moda_info["es_robusta"]
            altura_pico = moda_info["altura_relativa"]
            ancho_pico = moda_info["ancho_pico"]
        else:
            moda = float(moda_kde(x, bw_method=bw_method))
            es_moda_robusta = True
            altura_pico = 1.0
            ancho_pico = 1.0
    else:
        moda = np.nan
        es_moda_robusta = False
        altura_pico = np.nan
        ancho_pico = np.nan

    # Escala
    escala = madn(x) if usar_medida_robusta else np.std(x)
    if escala == 0:
        escala = 1e-12

    # Distancias normalizadas (usamos mediana como ancla)
    s_mean_med = abs(media - mediana) / escala
    s_mode_med = 0.0 if not incluir_moda else abs(mediana - moda) / escala
    s_mean_mode = 0.0 if not incluir_moda else abs(media - moda) / escala
    s_trim_med = 0.0 if not incluir_trimmed else abs(trimmed - mediana) / escala
    s_mean_trim = 0.0 if not incluir_trimmed else abs(media - trimmed) / escala

    bowley_asimetria = bowley_skew(x)
    g2 = excess_kurtosis(x) if use_kurtosis or use_kurtosis_bell else 0.0

    if ajustar_por_n:
        s_mean_med = shrink_s_by_n(s_mean_med, n, c=shrink_c)
        if incluir_moda:
            s_mode_med = shrink_s_by_n(s_mode_med, n, c=shrink_c)
            s_mean_mode = shrink_s_by_n(s_mean_mode, n, c=shrink_c)
        if incluir_trimmed:
            s_trim_med = shrink_s_by_n(s_trim_med, n, c=shrink_c)
            s_mean_trim = shrink_s_by_n(s_mean_trim, n, c=shrink_c)

    # Si hay moda no robusta, penalizar distancias ligadas a moda
    if incluir_moda and moda_robusta and not es_moda_robusta:
        s_mode_med *= 2.0
        s_mean_mode *= 2.0

    # ---- Asignación de pesos ----
    pesos_centros = {}
    centros = []
    distancias = []

    # Definir 'scores' o 'distances' según el método elegido
    if incluir_moda and incluir_trimmed:
        centros = ['media', 'mediana', 'moda', 'trimmed']
        distancias = np.array([s_mean_med, s_mode_med, s_mean_mode + 1e-12, s_trim_med])  # 4 valores
        # Para el cuarto (trimmed) usamos distancia a mediana; para 'mean-mode' añadimos pequeño término
        scores = -distancias
    elif incluir_moda:
        centros = ['media', 'mediana', 'moda']
        distancias = np.array([s_mean_med, s_mode_med, s_mean_mode + 1e-12])
        scores = -distancias
    elif incluir_trimmed:
        centros = ['media', 'mediana', 'trimmed']
        distancias = np.array([s_mean_med, s_trim_med, s_mean_trim + 1e-12])
        scores = -distancias
    else:
        centros = ['media', 'mediana']
        distancias = np.array([s_mean_med, 1e-12])  # dos centros: usar s_mean_med y un epsilon
        scores = -distancias

    # Ponderación
    if usar_transformacion_no_lineal:
        if method == "exponential":
            w_media = weight_exponential(s_mean_med, alpha=alpha)
        elif method == "logistic":
            w_media = weight_logistic(s_mean_med, s0=s0, p=p)
        elif method == "linear":
            w_media = weight_linear(s_mean_med, s_max=s_max)
        else:
            raise ValueError("method debe ser 'exponential', 'logistic' o 'linear'")
    else:
        w_media = max(0.0, 1.0 - s_mean_med)

    # Ajustes por curtosis
    if use_kurtosis:
        w_media = adjust_by_kurtosis(w_media, g2, beta=0.25)
    if use_kurtosis_bell:
        w_media = adjust_by_kurtosis_bell(w_media, g2, reward_window=0.5, reward_gain=0.08,
                                          beta_neg=0.15, beta_pos=0.25)

    # Ajuste adicional por Bowley (penaliza asimetría)
    if use_bowley:
        bowley_factor = np.exp(-0.2 * abs(bowley_asimetria))
        w_media *= bowley_factor

    # Reglas de piso de peso de la media (premio explícito)
    if w_media_floor is not None:
        w_media = max(w_media, float(w_media_floor))

    # Clip y peso complementar para caso de 2 centros
    lo, hi = clip
    w_media = float(np.clip(w_media, lo, hi))

    if len(centros) == 2:
        # pesos directos para media/mediana
        peso_media = w_media
        peso_mediana = 1.0 - w_media
        peso_moda = 0.0
        peso_trim = 0.0
        tendencia = peso_media * media + peso_mediana * mediana
    else:
        # más de 2 centros: usar softmax/convex sobre distancias
        if weight_method == 'softmax':
            pesos = softmax(scores, temperature=temperature)
        elif weight_method == 'convex':
            pesos = convex_weights(distancias, method=convex_method, alpha=temperature)
        else:
            raise ValueError("weight_method debe ser 'softmax' o 'convex'")

        # mapear pesos a centros
        mapa = dict(zip(centros, pesos))
        peso_media = float(mapa.get('media', 0.0))
        peso_mediana = float(mapa.get('mediana', 0.0))
        peso_moda = float(mapa.get('moda', 0.0))
        peso_trim = float(mapa.get('trimmed', 0.0))

        # normalizar de nuevo por seguridad
        total = peso_media + peso_mediana + peso_moda + peso_trim
        if total <= 0:
            peso_media, peso_mediana = 0.5, 0.5
            peso_moda = peso_trim = 0.0
            total = 1.0
        peso_media /= total
        peso_mediana /= total
        peso_moda /= total
        peso_trim /= total

        tendencia = (peso_media * media +
                     peso_mediana * mediana +
                     peso_moda * (0.0 if np.isnan(moda) else moda) +
                     peso_trim * (0.0 if np.isnan(trimmed) else trimmed))

    return pd.Series({
        "n": n,
        "media": media,
        "mediana": mediana,
        "moda": moda,
        "trimmed": trimmed,
        "MADN": madn(x),
        "bowley": bowley_asimetria,
        "exceso_kurtosis": g2,
        "s_mean_med": s_mean_med,
        "s_mode_med": s_mode_med if incluir_moda else np.nan,
        "s_mean_mode": s_mean_mode if incluir_moda else np.nan,
        "s_trim_med": s_trim_med if incluir_trimmed else np.nan,
        "s_mean_trim": s_mean_trim if incluir_trimmed else np.nan,
        "peso_media": peso_media,
        "peso_mediana": peso_mediana,
        "peso_moda": peso_moda,
        "peso_trimmed": peso_trim,
        "tendencia_ponderada": tendencia,
        "moda_robusta": es_moda_robusta,
        "altura_pico": altura_pico,
        "ancho_pico": ancho_pico
    })

# =========================================================
# función automática (v2) con reglas mejoradas
# =========================================================
def metrica_ajustada_v2(x, usar_expansion=False, columna_expansion=None, expansion_data=None):
    """
    Determina parámetros y centros óptimos con reglas mejoradas:
      - Bono explícito a la media en escenarios simétricos/mesocúrticos y n grande.
      - Ajuste de curtosis 'en campana' para premiar g2≈0.
      - Inclusión de media recortada en colas pesadas simétricas.
      - Perfil mesocúrtico con método 'linear' y temperatura/clip más amplios.
    """
    x_clean = pd.Series(x).dropna().values
    n = len(x_clean)

    if n < 3:
        res = metrica_ponderada_v2(x, incluir_moda=False, incluir_trimmed=False, usar_medida_robusta=True)
        return {"resultado": res, "diagnostico": {"razon": "muestra_muy_pequeña", "n": n}}

    media = np.mean(x_clean)
    mediana = np.median(x_clean)
    madn_val = madn(x_clean)
    bowley_abs = abs(bowley_skew(x_clean))
    sesgo_norm = abs(media - mediana) / madn_val if madn_val > 0 else 0.0
    g2 = excess_kurtosis(x_clean)
    s_sym = symmetry_index(x_clean)

    # Nivel de asimetría (similar a v1)
    if bowley_abs < 0.1 and sesgo_norm < 0.5:
        nivel_asimetria = "baja"
    elif bowley_abs < 0.3 and sesgo_norm < 1.0:
        nivel_asimetria = "moderada"
    else:
        nivel_asimetria = "alta"

    curtosis_signif = abs(g2) > 1.0

    # --- Reglas de inclusión de moda y trimmed ---
    usar_moda = False
    moda_es_robusta = False
    incluir_trimmed = False

    # Evaluación de moda solo si n≥50 y no estamos en mesocúrtico "limpio"
    if n >= 50:
        try:
            moda_info = moda_kde_robusta(
                x_clean,
                usar_expansion=usar_expansion,
                columna_expansion=columna_expansion,
                expansion_data=expansion_data,
                min_peak_height=0.15,  # estricto
                min_peak_width=0.05
            )
            moda_es_robusta = bool(moda_info["es_robusta"])
            altura = moda_info["altura_relativa"]
            ancho = moda_info["ancho_pico"]
            # si hay asimetría alta o multimodalidad clara, usar moda
            if moda_es_robusta and (altura > 2.0 or (nivel_asimetria == "alta" and altura > 1.5 and ancho > 0.03)):
                usar_moda = True
        except Exception:
            usar_moda = False

    # Media recortada: colas pesadas simétricas o g2 alto pero s_sym pequeño
    if (abs(g2) > 1.0 and s_sym < 0.08) or (nivel_asimetria == "baja" and abs(g2) > 0.8):
        incluir_trimmed = True

    # --- Selección de método base ---
    if nivel_asimetria == "baja" and abs(g2) < 0.5 and s_sym < 0.06:
        # Perfil mesocúrtico "limpio"
        method = "linear"
        usar_robusto = False
        temperature = 1.0
        clip = (0.10, 0.90)
        use_bowley = False
        use_kurtosis = False
        use_kurtosis_bell = True  # premiar g2 cerca de 0
    elif nivel_asimetria == "moderada":
        method = "logistic"
        usar_robusto = (sesgo_norm > 0.3) or curtosis_signif
        temperature = 0.5
        clip = (0.05, 0.95)
        use_bowley = bowley_abs > 0.2
        use_kurtosis = curtosis_signif
        use_kurtosis_bell = not curtosis_signif  # si no hay curtosis grande, permitir un pequeño premio
    else:
        method = "exponential"
        usar_robusto = True
        temperature = 0.3
        clip = (0.02, 0.98)
        use_bowley = True
        use_kurtosis = True
        use_kurtosis_bell = False

    # --- Parámetros específicos ---
    if method == "exponential":
        alpha = 0.693 if nivel_asimetria == "moderada" else 1.2
        s0, p, s_max = 1.0, 2.0, 2.0
    elif method == "logistic":
        alpha = 0.693
        s0 = 0.8 if (nivel_asimetria == "alta") else 1.0
        p = 3.0 if curtosis_signif else 2.0
        s_max = 2.0
    else:
        alpha, s0, p = 0.693, 1.0, 2.0
        s_max = 1.5 if nivel_asimetria == "baja" else 2.5

    # --- Tamaño muestral ---
    shrink_c = 50.0 if n < 100 else 100.0 if n < 500 else 200.0

    # --- Método de pesos cuando hay >2 centros ---
    if usar_moda or incluir_trimmed:
        if n > 1000 and nivel_asimetria == "alta":
            weight_method = "convex"
            convex_method = "exponential"
        else:
            weight_method = "softmax"
            convex_method = "inverse_distance"
    else:
        weight_method = "softmax"
        convex_method = "inverse_distance"

    # --- Bono explícito a la media en 'buen régimen' ---
    w_media_floor = None
    if (n >= 800) and (s_sym < 0.05) and (abs(g2) < 0.5) and (nivel_asimetria == "baja"):
        # subir temperatura (suaviza) y aumentar piso de la media
        temperature = max(temperature, 0.9)
        clip = (max(clip[0], 0.20), clip[1])
        w_media_floor = 0.60  # piso mínimo
        # alternativa: si quisieras ajustar s por n en régimen bueno, podrías hacerlo
        # a través de shrink_s_good_by_n dentro de metrica_ponderada_v2 (requiere pasarlo)

    # --- Ejecutar métrica v2 ---
    resultado = metrica_ponderada_v2(
        x=x_clean,
        method=method,
        usar_medida_robusta=usar_robusto,
        usar_transformacion_no_lineal=True,
        ajustar_por_n=True,
        use_kurtosis=use_kurtosis,
        use_kurtosis_bell=use_kurtosis_bell,
        use_bowley=use_bowley,
        incluir_moda=usar_moda,
        moda_robusta=usar_moda,   # si usamos moda, validar robustez
        incluir_trimmed=incluir_trimmed,
        trimmed_prop=0.10,
        weight_method=weight_method,
        convex_method=convex_method,
        temperature=temperature,
        alpha=alpha,
        s0=s0,
        p=p,
        s_max=s_max,
        shrink_c=shrink_c,
        clip=clip,
        # moda robusta
        usar_expansion=usar_expansion,
        columna_expansion=columna_expansion,
        expansion_data=expansion_data,
        min_peak_height=0.1,
        min_peak_width=0.05,
        # bono explícito
        w_media_floor=w_media_floor
    )

    diagnostico = {
        "n": n,
        "nivel_asimetria": nivel_asimetria,
        "sesgo_normalizado": sesgo_norm,
        "bowley_abs": bowley_abs,
        "exceso_kurtosis": g2,
        "curtosis_significativa": curtosis_signif,
        "symmetry_index": s_sym,
        "usar_moda": usar_moda,
        "moda_robusta": bool(resultado.get("moda_robusta", False)),
        "incluir_trimmed": incluir_trimmed,
        "parametros_elegidos": {
            "method": method,
            "usar_medida_robusta": usar_robusto,
            "use_kurtosis": use_kurtosis,
            "use_kurtosis_bell": use_kurtosis_bell,
            "use_bowley": use_bowley,
            "weight_method": weight_method,
            "convex_method": convex_method,
            "temperature": temperature,
            "alpha": alpha,
            "s0": s0,
            "p": p,
            "s_max": s_max,
            "shrink_c": shrink_c,
            "clip": clip,
            "w_media_floor": w_media_floor
        }
    }

    return {"resultado": resultado, "diagnostico": diagnostico}
