import numpy as np
import defs2
import defs
from defs import metrica_ajustada, bowley_skew, excess_kurtosis, madn
from defs2 import metrica_ajustada_v2
import pandas as pd
from typing import List, Tuple, Dict, Optional
from scipy.stats import wilcoxon, norm
from math import comb


##################################################################################
##################################################################################
##################################################################################
##################################################################################

# FUNCIONES PARA PRUEBAS DE ESTABILIDAD


# Función util para recortar 5% por cola
def trim_xy(x, prop=0.05):
    x = np.sort(np.asarray(x))
    n = len(x)
    k = int(np.floor(prop * n))
    if n - k <= 0:
        return x.copy()
    # Recorte unilateral: si side='left' quita los k menores, si side='right' quita los k mayores
    # Por defecto, recorte simétrico
    return x[k:n-k]

def trim_unilateral(x, prop=0.05, side='left'):
    x = np.sort(np.asarray(x))
    n = len(x)
    k = int(np.floor(prop * n))
    if n - k <= 0:
        return x.copy()
    if side == 'left':
        return x[k:]
    elif side == 'right':
        return x[:n-k]
    else:
        return x.copy()

# Bloque 1: generadores de datos (familias) + inyección de outliers

def sample_family_and_params(rng):
    familias = ["normal","t_student","uniforme","lognormal","gamma","beta","mixtura_bimodal"]
    fam = rng.choice(familias)

    if fam == "normal":
        mu = rng.uniform(-2, 2)
        sigma = rng.uniform(0.5, 3.0)
        params = dict(mu=mu, sigma=sigma)

    elif fam == "t_student":
        df = int(rng.integers(2, 16))
        scale = rng.uniform(0.5, 3.0)
        params = dict(df=df, scale=scale)

    elif fam == "uniforme":
        a = rng.uniform(-5, 0)
        b = rng.uniform(0, 5)
        if b <= a: b = a + abs(rng.normal(1.0, 0.5))
        params = dict(a=a, b=b)

    elif fam == "lognormal":
        mu = rng.uniform(-1, 1)
        sigma = rng.uniform(0.3, 1.2)
        params = dict(mu=mu, sigma=sigma)

    elif fam == "gamma":
        shape = rng.uniform(0.5, 8.0)
        scale = rng.uniform(0.5, 3.0)
        params = dict(shape=shape, scale=scale)

    elif fam == "beta":
        a = rng.uniform(0.3, 5.0)
        b = rng.uniform(0.3, 5.0)
        params = dict(a=a, b=b)

    elif fam == "mixtura_bimodal":
        pi = rng.uniform(0.2, 0.8)
        m1 = rng.normal(0, 1)
        gap = rng.uniform(1.0, 6.0)
        m2 = m1 + np.sign(rng.normal()) * gap
        s1 = rng.uniform(0.4, 2.0)
        s2 = rng.uniform(0.4, 2.0)
        params = dict(pi=pi, m1=m1, s1=s1, m2=m2, s2=s2)

    return fam, params

def sample_n(rng):
    # 1/3 pequeño 20-40, 1/3 mediano 80-200, 1/3 grande 500-5000
    bucket = rng.integers(0, 3)
    if bucket == 0:
        n = int(rng.integers(20, 41))
        rango = "pequeño"
    elif bucket == 1:
        n = int(rng.integers(80, 201))
        rango = "mediano"
    else:
        n = int(rng.integers(500, 5001))
        rango = "grande"
    return n, rango

def draw_sample(fam, params, n, rng):
    if fam == "normal":
        x = rng.normal(params["mu"], params["sigma"], size=n)

    elif fam == "t_student":
        # t * scale
        x = rng.standard_t(df=params["df"], size=n) * params["scale"]

    elif fam == "uniforme":
        x = rng.uniform(params["a"], params["b"], size=n)

    elif fam == "lognormal":
        x = rng.lognormal(mean=params["mu"], sigma=params["sigma"], size=n)

    elif fam == "gamma":
        x = rng.gamma(shape=params["shape"], scale=params["scale"], size=n)

    elif fam == "beta":
        x = rng.beta(a=params["a"], b=params["b"], size=n)
        x = 10.0 * (x - 0.5)  # re-escala a un rango real centrado

    elif fam == "mixtura_bimodal":
        z = rng.random(size=n)
        comp1 = rng.normal(params["m1"], params["s1"], size=n)
        comp2 = rng.normal(params["m2"], params["s2"], size=n)
        x = np.where(z < params["pi"], comp1, comp2)

    return x

def maybe_inject_outliers(x, rng, p_scenario=0.30, p_out=0.05):
    # 30% de las simulaciones reciben outliers; 5% de las obs. son outliers si aplica
    inject = rng.random() < p_scenario
    n_out = 0
    if inject:
        n = len(x)
        k = int(rng.binomial(n, p_out))
        if k > 0:
            sd = np.std(x) if np.std(x) > 0 else (madn(x) / 1.4826) or 1.0
            idx = rng.choice(np.arange(n), size=k, replace=False)
            jumps = rng.uniform(5*sd, 10*sd, size=k)
            signs = rng.choice([-1, 1], size=k)
            x = x.copy()
            x[idx] = x[idx] + signs * jumps
            n_out = k
    return x, inject, n_out



# Bloque 2: función que evalúa una simulación (una distribución)

def eval_one_sim(sim_id, rng, trim_prop=0.05):
    fam, params = sample_family_and_params(rng)
    n, rango_n = sample_n(rng)
    x = draw_sample(fam, params, n, rng)
    x, tuvo_outliers, n_out = maybe_inject_outliers(x, rng)

    # Recortes unilaterales
    from funciones_aux import trim_unilateral
    x_trim_left = trim_unilateral(x, prop=trim_prop, side='left')
    x_trim_right = trim_unilateral(x, prop=trim_prop, side='right')

    # Baselines
    mean_full = float(np.mean(x))
    median_full = float(np.median(x))
    mp1_full = float(metrica_ajustada(x)["resultado"]["tendencia_ponderada"])
    mp2_full = float(metrica_ajustada_v2(x)["resultado"]["tendencia_ponderada"])

    # Recorte izquierda
    mean_left = float(np.mean(x_trim_left))
    median_left = float(np.median(x_trim_left))
    mp1_left = float(metrica_ajustada(x_trim_left)["resultado"]["tendencia_ponderada"])
    mp2_left = float(metrica_ajustada_v2(x_trim_left)["resultado"]["tendencia_ponderada"])

    # Recorte derecha
    mean_right = float(np.mean(x_trim_right))
    median_right = float(np.median(x_trim_right))
    mp1_right = float(metrica_ajustada(x_trim_right)["resultado"]["tendencia_ponderada"])
    mp2_right = float(metrica_ajustada_v2(x_trim_right)["resultado"]["tendencia_ponderada"])

    # Deltas de estabilidad
    delta_mean_left = abs(mean_full - mean_left)
    delta_median_left = abs(median_full - median_left)
    delta_mp1_left = abs(mp1_full - mp1_left)
    delta_mp2_left = abs(mp2_full - mp2_left)

    delta_mean_right = abs(mean_full - mean_right)
    delta_median_right = abs(median_full - median_right)
    delta_mp1_right = abs(mp1_full - mp1_right)
    delta_mp2_right = abs(mp2_full - mp2_right)

    # Diagnósticos básicos
    skew_emp = bowley_skew(x)
    kurt_emp = excess_kurtosis(x)

    row = dict(
        sim_id=sim_id,
        familia=fam,
        n=n,
        rango_n=rango_n,
        tuvo_outliers=tuvo_outliers,
        n_outliers=n_out,
        skewness=skew_emp,
        kurtosis=kurt_emp,
        # valores full
        mean_full=mean_full, median_full=median_full, mp1_full=mp1_full, mp2_full=mp2_full,
        # valores recorte izquierda
        mean_left=mean_left, median_left=median_left, mp1_left=mp1_left, mp2_left=mp2_left,
        # valores recorte derecha
        mean_right=mean_right, median_right=median_right, mp1_right=mp1_right, mp2_right=mp2_right,
        # deltas de estabilidad izquierda
        delta_mean_left=delta_mean_left, delta_median_left=delta_median_left, delta_mp1_left=delta_mp1_left, delta_mp2_left=delta_mp2_left,
        # deltas de estabilidad derecha
        delta_mean_right=delta_mean_right, delta_median_right=delta_median_right, delta_mp1_right=delta_mp1_right, delta_mp2_right=delta_mp2_right,
        # mejoras de cada métrica vs baselines (izquierda)
        d_mp1_vs_mean_left=(delta_mean_left - delta_mp1_left),
        d_mp2_vs_mean_left=(delta_mean_left - delta_mp2_left),
        d_mp1_vs_median_left=(delta_median_left - delta_mp1_left),
        d_mp2_vs_median_left=(delta_median_left - delta_mp2_left),
        d_v2_vs_v1_left=(delta_mp1_left - delta_mp2_left),
        # mejoras de cada métrica vs baselines (derecha)
        d_mp1_vs_mean_right=(delta_mean_right - delta_mp1_right),
        d_mp2_vs_mean_right=(delta_mean_right - delta_mp2_right),
        d_mp1_vs_median_right=(delta_median_right - delta_mp1_right),
        d_mp2_vs_median_right=(delta_median_right - delta_mp2_right),
        d_v2_vs_v1_right=(delta_mp1_right - delta_mp2_right),
        params=params
    )
    return row


def resumen_victorias(
    df: pd.DataFrame,
    col_diff: str,
    nombre: str = None,
    threshold: float = 0.0
) -> Dict:
    """
    Calcula proporción y conteo de casos donde (df[col_diff] > threshold).
    Devuelve además media y mediana de ese diff para contexto.
    """
    vals = df[col_diff].dropna()
    mask = vals > threshold
    nombre = nombre or col_diff
    return {
        "comparacion": nombre,
        "proporcion": float(mask.mean()),
        "casos_gana": int(mask.sum()),
        "total": int(vals.shape[0]),
        "media_diff": float(vals.mean()),
        "mediana_diff": float(vals.median())
    }

def victorias_grouped(
    df: pd.DataFrame,
    group_col: str,
    diff_cols: List[str],
    threshold: float = 0.0
) -> pd.DataFrame:
    """
    Por cada grupo en group_col devuelve la proporción de casos con diff > threshold
    para cada columna en diff_cols.
    """
    def prop_pos(x):
        x = x.dropna()
        return float((x > threshold).mean())

    agg_dict = {c: prop_pos for c in diff_cols}
    out = (
        df.groupby(group_col)
          .agg(agg_dict)
          .reset_index()
    )
    return out

def test_wilcoxon_unilateral(
    df: pd.DataFrame,
    col_diff: str,
    nombre: str = None,
    alternative: str = "greater"
) -> Dict:
    """
    Wilcoxon signed-rank test unilateral sobre col_diff vs 0.
    alternative='greater' prueba si median(col_diff) > 0.
    """
    vals = df[col_diff].dropna()
    # zero_method="wilcox" ignora ceros exactos en las diferencias
    stat, p = wilcoxon(vals, alternative=alternative, zero_method="wilcox")
    return {
        "comparacion": nombre or col_diff,
        "n": int(len(vals)),
        "statistic": float(stat),
        "p_value": float(p),
        "alternative": alternative
    }

def run_resumenes(
    df: pd.DataFrame,
    comparisons: List[Tuple[str, str]],
    threshold: float = 0.0
) -> pd.DataFrame:
    """
    Construye tabla de resúmenes para una lista de (col_diff, etiqueta_amigable).
    """
    rows = [
        resumen_victorias(df, col, name, threshold)
        for col, name in comparisons
    ]
    return pd.DataFrame(rows)

def run_tests(
    df: pd.DataFrame,
    comparisons: List[Tuple[str, str]],
    alternative: str = "greater"
) -> pd.DataFrame:
    """
    Ejecuta Wilcoxon unilateral para cada (col_diff, etiqueta_amigable).
    """
    rows = [
        test_wilcoxon_unilateral(df, col, name, alternative=alternative)
        for col, name in comparisons
    ]
    return pd.DataFrame(rows)

def columnas_comparacion_default() -> List[Tuple[str, str]]:
    """
    Atajo con las 5 comparaciones estándar del A/B:
      - V1/V2 vs Media y Mediana, y V2 vs V1.
    """
    return [
        ("d_mp1_vs_mean",   "V1 vs Media"),
        ("d_mp2_vs_mean",   "V2 vs Media"),
        ("d_mp1_vs_median", "V1 vs Mediana"),
        ("d_mp2_vs_median", "V2 vs Mediana"),
        ("d_v2_vs_v1",      "V2 vs V1"),
    ]



##################################################################################
##################################################################################
##################################################################################
##################################################################################


# FUNCIONES PARA USO_GENERAL.IPYNB


