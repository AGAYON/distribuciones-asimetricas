import numpy as np
import defs2
import defs
from defs import metrica_ajustada, bowley_skew, excess_kurtosis, madn
from defs2 import metrica_ajustada_v2
import pandas as pd
from typing import List, Tuple, Dict, Optional
from scipy.stats import wilcoxon, norm
from math import comb
import matplotlib.pyplot as plt
from scipy import stats
import json


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

def cargar_datos_desde_archivo(ruta_archivo, columna=None, separador=',', encoding='latin1'):
    """
    Carga datos desde un archivo CSV, Excel o JSON y muestra información relevante.
    Retorna: datos (Series), df (DataFrame)
    """
    try:
        if ruta_archivo.endswith('.xlsx') or ruta_archivo.endswith('.xls'):
            df = pd.read_excel(ruta_archivo)
            print(f"✅ Archivo Excel cargado: {ruta_archivo}")
        elif ruta_archivo.endswith('.json'):
            df = pd.read_json(ruta_archivo)
            print(f"✅ Archivo JSON cargado: {ruta_archivo}")
        else:
            df = pd.read_csv(ruta_archivo, sep=separador, encoding=encoding)
            print(f"✅ Archivo CSV cargado: {ruta_archivo}")

        print(f"📊 Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas")
        print(f"📋 Columnas disponibles: {list(df.columns)}")

        if columna is None:
            columnas_numericas = df.select_dtypes(include=[np.number]).columns
            if len(columnas_numericas) == 0:
                raise ValueError("No se encontraron columnas numéricas")
            columna = columnas_numericas[0]
            print(f"🎯 Usando columna automática: '{columna}'")
        else:
            if columna not in df.columns:
                raise ValueError(f"Columna '{columna}' no encontrada")
            print(f"🎯 Usando columna especificada: '{columna}'")

        datos = df[columna].dropna()
        datos = datos[pd.to_numeric(datos, errors='coerce').notna()]
        datos = pd.to_numeric(datos)

        print(f"✅ Datos extraídos: {len(datos)} valores válidos")
        print(f"📈 Rango: {datos.min():.2f} - {datos.max():.2f}")

        if datos is not None and len(datos) > 0:
            print(f"\n✅ ¡PERFECTO! Tus datos han sido cargados exitosamente")
            print(f"📊 Observaciones: {len(datos):,}")
            print(f"📈 Estos datos se usarán en todo el análisis")
            print(f"\n🚀 PUEDES CONTINUAR CON PASO 1.5 (Limpieza) o saltar al PASO 2 (Análisis)")
        else:
            print(f"\n❌ No se cargaron datos desde archivo")
            print(f"💡 Si no tienes archivo, continúa con las opciones 2 o 3")
        return datos, df

    except Exception as e:
        print(f"❌ Error al cargar archivo: {e}")
        print("\n💡 Consejos:")
        print("   • Verifica que la ruta del archivo sea correcta")
        print("   • Para archivos con caracteres especiales, prueba encoding='utf-8'")
        print("   • Asegúrate de que el archivo tenga columnas numéricas")
        print(f"\n❌ No se cargaron datos desde archivo")
        print(f"💡 Si no tienes archivo, continúa con las opciones 2 o 3")
        return None, None
    



def generar_datos_ejemplo(tipo="ingresos", n=1000, semilla=42, verbose=True):
    """
    Genera datos de ejemplo para probar la herramienta.
    Retorna: datos (Series), nombre_datos (str), unidad (str)
    """
    import numpy as np
    import pandas as pd
    np.random.seed(semilla)
    if tipo == "ingresos":
        datos = np.random.lognormal(mean=10, sigma=0.8, size=n)
        nombre = "Distribución de Ingresos (log-normal)"
        unidad = "pesos"
    elif tipo == "tiempos":
        datos = np.random.exponential(scale=5, size=n)
        nombre = "Tiempos de Respuesta (exponencial)"
        unidad = "segundos"
    elif tipo == "precios":
        datos = np.random.gamma(shape=2, scale=100, size=n)
        nombre = "Precios de Productos (gamma)"
        unidad = "pesos"
    elif tipo == "asimetrico_alto":
        base = np.random.exponential(scale=2, size=int(n*0.9))
        outliers = np.random.exponential(scale=20, size=int(n*0.1)) + 50
        datos = np.concatenate([base, outliers])
        np.random.shuffle(datos)
        nombre = "Distribución Altamente Asimétrica"
        unidad = "unidades"
    elif tipo == "normal":
        datos = np.random.normal(loc=50, scale=15, size=n)
        datos = datos[datos > 0]
        nombre = "Distribución Normal"
        unidad = "unidades"
    else:
        raise ValueError(f"Tipo '{tipo}' no reconocido. Usa: ingresos, tiempos, precios, asimetrico_alto, normal")
    datos = pd.Series(datos)
    if verbose:
        print(f"✅ Datos generados: {nombre}")
        print(f"📊 Observaciones: {len(datos)}")
        print(f"📈 Rango: {datos.min():.2f} - {datos.max():.2f} {unidad}")
    return datos, nombre, unidad




def limpiar_datos(
    datos,
    limpiar_nulos=True,
    limpiar_ceros=False,
    limpiar_negativos=False,
    limpiar_outliers=False,
    percentil_inferior=1,
    percentil_superior=99,
    limpiar_personalizado=False,
    valor_min=None,
    valor_max=None,
    verbose=True
):
    """
    Aplica limpieza de datos según los parámetros activados.
    Retorna: datos_limpios (Series)
    """
    import numpy as np
    datos_originales = datos.copy()
    if verbose:
        print(f"📊 DATOS ORIGINALES:")
        print(f"   • Observaciones: {len(datos):,}")
        print(f"   • Rango: {datos.min():.2f} - {datos.max():.2f}")
        print(f"   • Valores nulos: {datos.isnull().sum()}")
        print(f"   • Valores cero: {(datos == 0).sum()}")
        print(f"   • Valores negativos: {(datos < 0).sum()}")
        print(f"\n🛠️  OPCIONES DE LIMPIEZA DISPONIBLES:")
        print("=" * 50)

    datos_limpios = datos.copy()
    removidos_total = 0

    if limpiar_nulos:
        antes = len(datos_limpios)
        datos_limpios = datos_limpios.dropna()
        removidos = antes - len(datos_limpios)
        removidos_total += removidos
        if verbose: print(f"✅ Valores nulos removidos: {removidos}")

    if limpiar_ceros:
        antes = len(datos_limpios)
        datos_limpios = datos_limpios[datos_limpios != 0]
        removidos = antes - len(datos_limpios)
        removidos_total += removidos
        if verbose: print(f"✅ Valores cero removidos: {removidos}")

    if limpiar_negativos:
        antes = len(datos_limpios)
        datos_limpios = datos_limpios[datos_limpios >= 0]
        removidos = antes - len(datos_limpios)
        removidos_total += removidos
        if verbose: print(f"✅ Valores negativos removidos: {removidos}")

    if limpiar_outliers:
        p_inf = np.percentile(datos_limpios, percentil_inferior)
        p_sup = np.percentile(datos_limpios, percentil_superior)
        antes = len(datos_limpios)
        datos_limpios = datos_limpios[(datos_limpios >= p_inf) & (datos_limpios <= p_sup)]
        removidos = antes - len(datos_limpios)
        removidos_total += removidos
        if verbose:
            print(f"✅ Outliers extremos removidos (P{percentil_inferior}-P{percentil_superior}): {removidos}")
            print(f"   • Límite inferior: {p_inf:.2f}")
            print(f"   • Límite superior: {p_sup:.2f}")

    if limpiar_personalizado:
        antes = len(datos_limpios)
        if valor_min is not None:
            datos_limpios = datos_limpios[datos_limpios >= valor_min]
        if valor_max is not None:
            datos_limpios = datos_limpios[datos_limpios <= valor_max]
        removidos = antes - len(datos_limpios)
        removidos_total += removidos
        if verbose: print(f"✅ Filtro personalizado aplicado. Total removidos: {removidos}")

    # Resumen
    if verbose:
        if len(datos_limpios) == len(datos_originales):
            print(f"\n📊 NO SE APLICARON FILTROS")
            print(f"   • Los datos originales se mantienen sin cambios")
        else:
            observaciones_removidas = len(datos_originales) - len(datos_limpios)
            porcentaje_removido = (observaciones_removidas / len(datos_originales)) * 100
            print(f"\n📊 RESUMEN DE LIMPIEZA APLICADA:")
            print("=" * 50)
            print(f"   • Observaciones originales: {len(datos_originales):,}")
            print(f"   • Observaciones finales: {len(datos_limpios):,}")
            print(f"   • Removidas: {observaciones_removidas:,} ({porcentaje_removido:.1f}%)")
            print(f"   • Nuevo rango: {datos_limpios.min():.2f} - {datos_limpios.max():.2f}")
            print(f"   • Nueva media: {datos_limpios.mean():.2f}")
            print(f"   • Nueva mediana: {datos_limpios.median():.2f}")
            if porcentaje_removido > 20:
                print(f"\n⚠️  ADVERTENCIA: Se removió más del 20% de los datos")
            elif porcentaje_removido > 5:
                print(f"\n🟡 MODERADO: Se removió {porcentaje_removido:.1f}% de los datos")
            else:
                print(f"\n🟢 LIMPIEZA MÍNIMA: Solo se removió {porcentaje_removido:.1f}% de los datos")
        print(f"\n" + "=" * 70)
        print("📊 ESTADÍSTICAS FINALES DE LOS DATOS (post-limpieza)")
        print("=" * 70)
        print(f"• Media aritmética: {datos_limpios.mean():,.3f}")
        print(f"• Mediana: {datos_limpios.median():,.3f}")
        print(f"• Desviación estándar: {datos_limpios.std():,.3f}")
        print(f"• Asimetría (momentos): {datos_limpios.skew():.3f}")
        print(f"• Curtosis (momentos): {datos_limpios.kurtosis():.3f}")
        print(f"• Coeficiente de variación: {(datos_limpios.std() / datos_limpios.mean()):.3f}")
        print(f"\n✅ DATOS FINALES LISTOS PARA ANÁLISIS:")
        print(f"   • Observaciones: {len(datos_limpios):,}")
        print(f"   • Rango: {datos_limpios.min():.2f} - {datos_limpios.max():.2f}")
        print(f"   • Estos datos se usarán en el análisis automático")
        print(f"\n🚀 Continúa con el PASO 2 (Análisis Automático)")
    return datos_limpios




def analizar_datos_completo(datos, nombre_datos="tus datos"):
    """
    Ejecuta el análisis automático completo sobre los datos y muestra diagnóstico y resultados.
    Retorna: resultado_auto (dict)
    """
    import pandas as pd

    print("🚀 INICIANDO ANÁLISIS AUTOMÁTICO")
    print("=" * 80)
    if datos is None or len(datos) == 0:
        print("❌ ERROR: No se encontraron datos válidos para analizar")
        print("💡 Regresa al PASO 1 y carga tus datos correctamente")
        print("📋 Opciones disponibles:")
        print("   • Opción 1: Cargar desde archivo CSV/Excel/JSON")
        print("   • Opción 2: Cargar desde lista de valores")
        print("   • Opción 3: Usar datos de ejemplo")
        return None
    print(f"📊 ANALIZANDO: {nombre_datos}")
    print(f"📈 Observaciones: {len(datos):,}")
    print(f"📈 Rango: {datos.min():.2f} - {datos.max():.2f}")
    print(f"\n🤖 EJECUTANDO ANÁLISIS AUTOMÁTICO...")
    print("-" * 60)
    try:
        from defs import metrica_ajustada
        resultado_auto = metrica_ajustada(datos)
        resultado = resultado_auto['resultado']
        diagnostico = resultado_auto['diagnostico']
        print(f"✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
        # ...diagnóstico y resultados (igual que tu código original)...
        # Puedes copiar aquí el resto de los prints si quieres todos los detalles.
        return resultado_auto
    except Exception as e:
        print(f"❌ ERROR en el análisis automático: {e}")
        print("💡 Verifica que tus datos sean numéricos y no contengan valores extremos problemáticos")
        return None





def crear_visualizaciones(datos, resultado_auto, nombre_datos="tus datos"):
    """
    Crea un conjunto completo de visualizaciones para entender los resultados
    """
    if resultado_auto is None:
        print("❌ No hay resultados para visualizar. Ejecuta primero el análisis automático.")
        return

    resultado = resultado_auto['resultado']
    diagnostico = resultado_auto['diagnostico']

    plt.style.use('default')
    fig = plt.figure(figsize=(20, 15))

    print(f"📊 CREANDO VISUALIZACIONES PARA: {nombre_datos}")
    print("=" * 70)

    # Gráfico 1: Distribución principal
    ax1 = plt.subplot(3, 3, 1)
    ax1.hist(datos, bins=min(50, len(datos)//10), alpha=0.7, color='lightblue',
             density=True, edgecolor='black', linewidth=0.5)
    ax1.axvline(resultado['media'], color='red', linestyle='--', linewidth=2.5,
               label=f'Media: {resultado["media"]:,.1f}')
    ax1.axvline(resultado['mediana'], color='green', linestyle='--', linewidth=2.5,
               label=f'Mediana: {resultado["mediana"]:,.1f}')
    ax1.axvline(resultado['tendencia_ponderada'], color='purple', linestyle='-', linewidth=3,
               label=f'Métrica Ponderada: {resultado["tendencia_ponderada"]:,.1f}')
    if not pd.isna(resultado['moda']):
        ax1.axvline(resultado['moda'], color='orange', linestyle=':', linewidth=2.5,
                   label=f'Moda: {resultado["moda"]:,.1f}')
    ax1.set_title(f'Distribución de {nombre_datos}', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Valores')
    ax1.set_ylabel('Densidad')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Gráfico 2: Comparación de métricos
    ax2 = plt.subplot(3, 3, 2)
    metricas = ['Media\nSimple', 'Mediana', 'Métrica\nPonderada']
    valores = [resultado['media'], resultado['mediana'], resultado['tendencia_ponderada']]
    colores = ['red', 'green', 'purple']
    bars = ax2.bar(metricas, valores, color=colores, alpha=0.7, edgecolor='black')
    ax2.set_title('Comparación de Métricos', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Valor')
    for bar, valor in zip(bars, valores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{valor:,.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Gráfico 3: Distribución de pesos
    ax3 = plt.subplot(3, 3, 3)
    if resultado['peso_moda'] > 0:
        labels = ['Media', 'Mediana', 'Moda']
        sizes = [resultado['peso_media'], resultado['peso_mediana'], resultado['peso_moda']]
        colors = ['red', 'green', 'orange']
    else:
        labels = ['Media', 'Mediana']
        sizes = [resultado['peso_media'], resultado['peso_mediana']]
        colors = ['red', 'green']
    ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 10})
    ax3.set_title('Distribución de Pesos', fontweight='bold', fontsize=12)

    # Gráfico 4: Métricas de diagnóstico
    ax4 = plt.subplot(3, 3, 4)
    metricas_diag = ['Asimetría\nBowley', 'Sesgo\nNormalizado', 'Exceso\nCurtosis']
    valores_diag = [abs(diagnostico['bowley_asimetria']),
                   diagnostico['sesgo_normalizado'],
                   abs(diagnostico['exceso_curtosis'])]
    colores_diag = ['blue', 'cyan', 'magenta']
    bars_diag = ax4.bar(metricas_diag, valores_diag, color=colores_diag, alpha=0.7, edgecolor='black')
    ax4.set_title('Métricas de Diagnóstico', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Valor')
    for bar, valor in zip(bars_diag, valores_diag):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                 f'{valor:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    # Gráfico 5: Box plot
    ax5 = plt.subplot(3, 3, 5)
    box = ax5.boxplot(datos, patch_artist=True, vert=True)
    box['boxes'][0].set_facecolor('lightcoral')
    box['boxes'][0].set_alpha(0.7)
    ax5.axhline(resultado['media'], color='red', linestyle='--', linewidth=2, label='Media')
    ax5.axhline(resultado['mediana'], color='green', linestyle='--', linewidth=2, label='Mediana')
    ax5.axhline(resultado['tendencia_ponderada'], color='purple', linestyle='-', linewidth=3, label='Métrica Ponderada')
    ax5.set_title('Box Plot con Métricos', fontweight='bold', fontsize=12)
    ax5.set_ylabel('Valores')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # Gráfico 6: Histograma con curva de densidad
    ax6 = plt.subplot(3, 3, 6)
    ax6.hist(datos, bins=min(30, len(datos)//20), alpha=0.5, color='lightgray', density=True,
             label='Histograma')
    if len(datos) > 50:
        try:
            datos.plot(kind='kde', ax=ax6, color='red', linewidth=2, label='Curva de densidad')
        except:
            pass
    ax6.set_title('Distribución con Curva de Densidad', fontweight='bold', fontsize=12)
    ax6.set_xlabel('Valores')
    ax6.set_ylabel('Densidad')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)

    # Gráfico 7: Q-Q plot contra normal
    ax7 = plt.subplot(3, 3, 7)
    stats.probplot(datos, dist="norm", plot=ax7)
    ax7.set_title('Q-Q Plot vs Distribución Normal', fontweight='bold', fontsize=12)
    ax7.grid(True, alpha=0.3)

    # Gráfico 8: Región central (sin outliers)
    ax8 = plt.subplot(3, 3, 8)
    p5, p95 = np.percentile(datos, [5, 95])
    datos_centrales = datos[(datos >= p5) & (datos <= p95)]
    ax8.hist(datos_centrales, bins=min(30, len(datos_centrales)//10), alpha=0.7,
             color='lightgreen', density=True, edgecolor='black', linewidth=0.5)
    ax8.axvline(resultado['media'], color='red', linestyle='--', linewidth=2, label='Media')
    ax8.axvline(resultado['mediana'], color='green', linestyle='--', linewidth=2, label='Mediana')
    ax8.axvline(resultado['tendencia_ponderada'], color='purple', linestyle='-', linewidth=3,
               label='Métrica Ponderada')
    ax8.set_title('Región Central (P5-P95)', fontweight='bold', fontsize=12)
    ax8.set_xlabel('Valores')
    ax8.set_ylabel('Densidad')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)

    # Gráfico 9: Resumen de configuración
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    config_text = f"""
CONFIGURACIÓN AUTOMÁTICA:

• Nivel de asimetría: {diagnostico['nivel_asimetria'].upper()}
• Método elegido: {diagnostico['parametros_elegidos']['method'].upper()}
• Ponderación: {diagnostico['parametros_elegidos']['weight_method'].upper()}

RESULTADOS CLAVE:

• Observaciones: {len(datos):,}
• Media simple: {resultado['media']:,.2f}
• Métrica ponderada: {resultado['tendencia_ponderada']:,.2f}
• Diferencia: {abs(resultado['tendencia_ponderada'] - resultado['media']):,.2f}

PESOS FINALES:

• Media: {resultado['peso_media']:.1%}
• Mediana: {resultado['peso_mediana']:.1%}"""
    if resultado['peso_moda'] > 0:
        config_text += f"\n• Moda: {resultado['peso_moda']:.1%}"
    ax9.text(0.1, 0.9, config_text, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))

    plt.tight_layout()
    plt.show()

    print("✅ Visualizaciones creadas exitosamente")
    print("📊 Interpretación:")
    if diagnostico['nivel_asimetria'] == "alta":
        print("   🔴 Tu distribución es altamente asimétrica - la métrica ponderada es muy recomendable")
    elif diagnostico['nivel_asimetria'] == "moderada":
        print("   🟡 Tu distribución tiene asimetría moderada - la métrica ponderada mejora la representatividad")
    else:
        print("   🟢 Tu distribución es relativamente simétrica - ambas métricas son válidas")




import matplotlib.pyplot as plt

def ejecutar_comparacion_avanzada(
    datos,
    resultado_final=None,
    # Parámetros para metrica_ponderada (puedes modificar desde tu flujo)
    method="linear",
    incluir_moda=False,
    moda_robusta=False,
    use_kurtosis=False,
    use_bowley=False,
    weight_method=None,
    usar_medida_robusta=False,
    temperature=1.0,
    alpha=None,
    s_max=None,
    clip=(0.1, 0.9)
):
    """
    Ejecuta la comparación avanzada de configuraciones de la métrica ponderada.
    Modifica los parámetros al llamar esta función para probar diferentes variantes.
    """
    from defs import metrica_ponderada
    resultado = metrica_ponderada(
        datos,
        method=method,
        incluir_moda=incluir_moda,
        moda_robusta=moda_robusta,
        use_kurtosis=use_kurtosis,
        use_bowley=use_bowley,
        weight_method=weight_method,
        usar_medida_robusta=usar_medida_robusta,
        temperature=temperature,
        alpha=alpha,
        s_max=s_max,
        clip=clip
    )

    print(f"⚙️  COMPARACIÓN AVANZADA DE MÉTRICAS")
    print("=" * 70)
    print(f"   • Resultado: {resultado['tendencia_ponderada']:.3f}")
    print(f"   • Peso media: {resultado['peso_media']:.3f}")
    print(f"   • Peso mediana: {resultado['peso_mediana']:.3f}")
    if 'peso_moda' in resultado:
        print(f"   • Peso moda: {resultado['peso_moda']:.3f}")

    # Comparación con otras configuraciones si resultado_final está disponible
    configuraciones = {
        "Automática": resultado_final['resultado']['tendencia_ponderada'] if resultado_final else None,
        "Personalizada": resultado['tendencia_ponderada'],
        "Media Simple": datos.mean(),
        "Mediana Simple": datos.median()
    }

    print(f"\n📊 COMPARACIÓN DE CONFIGURACIONES:")
    print("-" * 70)
    for nombre, valor in configuraciones.items():
        if valor is not None:
            print(f"   • {nombre:<15}: {valor:>10.3f}")

    # Gráfico comparativo
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    nombres = [k for k, v in configuraciones.items() if v is not None]
    valores = [v for v in configuraciones.values() if v is not None]
    colores = ['purple', 'blue', 'gray', 'black']
    bars = ax.bar(nombres, valores, color=colores[:len(nombres)], alpha=0.7, edgecolor='black')
    ax.set_title('Comparación de Configuraciones', fontweight='bold', fontsize=14)
    ax.set_ylabel('Valor de la Métrica', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    for bar, valor in zip(bars, valores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{valor:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

    return resultado





def exportar_resultados(
    datos,
    resultado_auto,
    nombre_archivo="resultados_metrica_ponderada",
    exportar_json=True,
    exportar_csv=True,
    exportar_txt=True,
    mostrar_resumen=True
):
    """
    Exporta los resultados del análisis a diferentes formatos según los parámetros.
    """
    if resultado_auto is None:
        print("❌ No hay resultados para exportar")
        return

    resultado = resultado_auto['resultado']
    diagnostico = resultado_auto['diagnostico']

    resumen = {
        'Análisis': {
            'Observaciones': len(datos),
            'Rango_min': float(datos.min()),
            'Rango_max': float(datos.max()),
            'Media_simple': float(datos.mean()),
            'Mediana': float(datos.median()),
            'Desviacion_std': float(datos.std()),
            'Asimetria_momentos': float(datos.skew()),
            'Curtosis_momentos': float(datos.kurtosis())
        },
        'Diagnostico_automatico': {
            'Nivel_asimetria': diagnostico['nivel_asimetria'],
            'Asimetria_Bowley': float(diagnostico['bowley_asimetria']),
            'Sesgo_normalizado': float(diagnostico['sesgo_normalizado']),
            'Exceso_curtosis': float(diagnostico['exceso_curtosis']),
            'Curtosis_significativa': diagnostico['curtosis_significativa'],
            'Usar_moda': diagnostico['usar_moda'],
            'Moda_robusta': diagnostico.get('moda_robusta', False)
        },
        'Configuracion_elegida': diagnostico['parametros_elegidos'],
        'Resultados_finales': {
            'Media_aritmetica': float(resultado['media']),
            'Mediana': float(resultado['mediana']),
            'Moda_KDE': float(resultado['moda']) if not pd.isna(resultado['moda']) else None,
            'Metrica_ponderada': float(resultado['tendencia_ponderada']),
            'Peso_media': float(resultado['peso_media']),
            'Peso_mediana': float(resultado['peso_mediana']),
            'Peso_moda': float(resultado['peso_moda']),
            'MADN': float(resultado['MADN']),
            'Diferencia_vs_media': float(abs(resultado['tendencia_ponderada'] - resultado['media'])),
            'Diferencia_relativa_pct': float(abs(resultado['tendencia_ponderada'] - resultado['media']) / resultado['media'] * 100)
        }
    }

    if mostrar_resumen:
        print("📊 RESUMEN FINAL DE RESULTADOS:")
        print("-" * 50)
        print(f"🎯 MÉTRICA PONDERADA RECOMENDADA: {resultado['tendencia_ponderada']:,.3f}")
        print(f"📈 vs Media simple: {resultado['media']:,.3f}")
        print(f"💰 Diferencia absoluta: {abs(resultado['tendencia_ponderada'] - resultado['media']):,.3f}")
        print(f"📊 Diferencia relativa: {abs(resultado['tendencia_ponderada'] - resultado['media']) / resultado['media'] * 100:.2f}%")
        print(f"\n⚙️  CONFIGURACIÓN AUTOMÁTICA ELEGIDA:")
        print(f"   • Nivel de asimetría: {diagnostico['nivel_asimetria'].upper()}")
        print(f"   • Método: {diagnostico['parametros_elegidos']['method'].upper()}")
        print(f"   • Ponderación: {diagnostico['parametros_elegidos']['weight_method'].upper()}")
        print(f"\n⚖️  DISTRIBUCIÓN DE PESOS:")
        print(f"   • Media: {resultado['peso_media']:.1%}")
        print(f"   • Mediana: {resultado['peso_mediana']:.1%}")
        if resultado['peso_moda'] > 0:
            print(f"   • Moda: {resultado['peso_moda']:.1%}")

    try:
        if exportar_json:
            with open(f"{nombre_archivo}.json", 'w', encoding='utf-8') as f:
                json.dump(resumen, f, indent=4, ensure_ascii=False)
            print(f"\n✅ Resultados guardados en: {nombre_archivo}.json")
        if exportar_csv:
            resultados_csv = pd.DataFrame({
                'Metrica': ['Media_simple', 'Mediana', 'Metrica_ponderada'],
                'Valor': [resultado['media'], resultado['mediana'], resultado['tendencia_ponderada']],
                'Peso': [resultado['peso_media'], resultado['peso_mediana'], 1.0]
            })
            resultados_csv.to_csv(f"{nombre_archivo}.csv", index=False, encoding='utf-8')
            print(f"✅ Tabla de resultados guardada en: {nombre_archivo}.csv")
        if exportar_txt:
            with open(f"{nombre_archivo}_reporte.txt", 'w', encoding='utf-8') as f:
                f.write("REPORTE DE ANÁLISIS - MÉTRICA PONDERADA PARA DISTRIBUCIONES ASIMÉTRICAS\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Fecha de análisis: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Observaciones analizadas: {len(datos):,}\n\n")
                f.write("RESULTADOS PRINCIPALES:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Media aritmética simple: {resultado['media']:,.3f}\n")
                f.write(f"Mediana: {resultado['mediana']:,.3f}\n")
                f.write(f"Métrica ponderada (recomendada): {resultado['tendencia_ponderada']:,.3f}\n\n")
                f.write("DIAGNÓSTICO AUTOMÁTICO:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Nivel de asimetría: {diagnostico['nivel_asimetria']}\n")
                f.write(f"Asimetría de Bowley: {diagnostico['bowley_asimetria']:.4f}\n")
                f.write(f"Método elegido: {diagnostico['parametros_elegidos']['method']}\n")
                f.write(f"Ponderación elegida: {diagnostico['parametros_elegidos']['weight_method']}\n\n")
                f.write("RECOMENDACIÓN:\n")
                f.write("-" * 30 + "\n")
                if diagnostico['nivel_asimetria'] == "alta":
                    f.write("La distribución presenta ALTA asimetría. Se recomienda fuertemente usar la métrica ponderada.\n")
                elif diagnostico['nivel_asimetria'] == "moderada":
                    f.write("La distribución presenta asimetría MODERADA. La métrica ponderada mejora la representatividad.\n")
                else:
                    f.write("La distribución es relativamente SIMÉTRICA. Ambas métricas son válidas.\n")
            print(f"✅ Reporte completo guardado en: {nombre_archivo}_reporte.txt")
    except Exception as e:
        print(f"⚠️  Error al guardar archivos: {e}")
        print("💡 Los resultados siguen disponibles en memoria")

    if mostrar_resumen:
        print(f"\n🎯 PRÓXIMOS PASOS RECOMENDADOS:")
        print("-" * 50)
        print("1. Usa la métrica ponderada en tus análisis futuros")
        print("2. Documenta la metodología en tus reportes")
        print("3. Aplica la misma herramienta a datasets similares")
        print("4. Comparte los archivos generados con tu equipo")
        if abs(resultado['tendencia_ponderada'] - resultado['media']) / resultado['media'] * 100 > 5:
            print("\n🔴 NOTA IMPORTANTE:")
            print("   La diferencia entre la métrica ponderada y la media simple es significativa.")
            print("   Considera usar la métrica ponderada para decisiones importantes.")

    return resumen