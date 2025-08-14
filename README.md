# Métrica Ponderada para Representatividad Distributiva
## 🎯 Encuentra el dato más representativo de tu distribución, incluso cuando la media simple te falla

### ¿Tu promedio no cuenta toda la historia? Aquí está la solución.

Bienvenido/a al repositorio de **análisis robusto de distribuciones asimétricas**, un proyecto nacido de una necesidad real: encontrar una métrica más representativa que la media aritmética para datos de salarios en México, pero que creció hasta convertirse en una herramienta general para **cualquier distribución que se salga de lo "normal"** (exceptuando distribuciones bimodales).

## 🚀 ¿Qué problema se resuelve?

Imagínate esto: tienes datos de ingresos, precios, tiempos de respuesta, o cualquier variable económica/social. Calculas el promedio y sientes que **no representa realmente** lo que vive la mayoría de tu población. ¿Te suena familiar?

**El problema:** La media aritmética se "infla" (o se "poncha") con valores extremos y en distribuciones asimétricas puede estar muy lejos de lo que experimenta el ciudadano/cliente/usuario típico.

**La propuesta:** Una **métrica ponderada inteligente** que combina automáticamente media, mediana y (cuando es apropiado) moda, adaptándose a las características específicas de tu distribución.

## 💡 ¿Por qué nació este proyecto?

Este proyecto nació de una pregunta concreta: **¿Cuál es el salario representativo de un mexicano?** 

Cuando analizamos los datos de la ENOE (Encuesta Nacional de Ocupación y Empleo), nos dimos cuenta de que:
- La **media** ($59.07/hora) se veía inflada por altos ejecutivos y profesionistas
- La **mediana** ($42.71/hora) ignoraba completamente la información de la distribución completa
- Necesitábamos algo **más inteligente** que capturara la complejidad real de los datos

El resultado: una métrica que **detecta automáticamente** el nivel de asimetría y **adapta su comportamiento** para darte el valor más representativo posible.

## 🛠️ ¿Qué hace exactamente esta herramienta?

### **Análisis Automático Inteligente**
1. **🔍 Detecta automáticamente** las características de tu distribución (asimetría, curtosis, valores extremos)
2. **🧠 Selecciona la configuración óptima** basándose en más de 15 criterios estadísticos
3. **⚖️ Combina inteligentemente** media, mediana y moda con pesos adaptativos
4. **📊 Te explica** por qué eligió cada configuración y qué significan los resultados

### **Fundamento Matemático Sólido**
- **Medidas robustas**: MADN (Median Absolute Deviation Normalized)
- **Asimetría robusta**: Bowley skewness basado en cuartiles
- **Estimación de moda**: KDE (Kernel Density Estimation) con validación de robustez
- **Ponderación adaptiva**: Softmax, convex weights, y mapeos logísticos/exponenciales

### **Casos de Uso Perfectos**
✅ **Datos de ingresos y salarios** (como el caso que nos motivó)  
✅ **Precios** y variables financieras  
✅ **Tiempos de espera** o respuesta  
✅ **Datos médicos** con distribuciones complejas  
✅ **Métricas de negocio** con outliers  
✅ **Investigación académica** y políticas públicas  

## 📁 ¿Qué encuentras en este repositorio?

### **`uso_general.ipynb`** - Tu punto de partida
Un notebook **completamente auto-contenido** que te guía paso a paso:
- 📂 **Cargar datos** (CSV, Excel, listas, o generar ejemplos)
- 🤖 **Análisis automático** completo con una sola función
- 📊 **Visualizaciones comprehensivas** (9 gráficos explicativos)
- ⚙️ **Configuración avanzada** si quieres control total
- 📋 **Exportación** a múltiples formatos (JSON, CSV, reportes)

### **`metodologia_ejemplo.ipynb`** - La historia completa
Un análisis **académico completo** usando datos reales de salarios mexicanos:
- 📚 **Fundamentación matemática** paso a paso
- 🧮 **Desarrollo teórico** con LaTeX y demostraciones
- 📈 **Validación empírica** con datos de la ENOE
- 🎯 **Caso de estudio real** que inspiró todo el proyecto

### **`defs.py`** - El motor matemático
Las funciones principales que hacen toda la magia:
- `metrica_ajustada()` - Análisis automático completo
- `metrica_ponderada()` - Control manual para usuarios avanzados
- Funciones auxiliares robustas para todos los cálculos 

### **`defs2.py`** - El motor matemático al cuadrado
Ajustes a la métrica, original. Es el archivo en el cual
implemento las mejoras que espero terminen integradas en defs.py

### **`prueba_estabilidad.ipynb`** - Robustez y comparación de métricas  
Un análisis **experimental** para validar la estabilidad de las métricas propuestas:
- 🧪 **Simulación masiva** de distribuciones y recortes de datos
- ⚔️ **Comparación directa** entre métricas ponderadas, media y mediana
- 📊 **Resumenes y visualizaciones** de victorias, diferencias y significancia estadística
- 🏆 **Tests Wilcoxon** para demostrar superioridad estadística
- 🔍 **Conclusiones claras** sobre los resultados obtenidos


## 🚀 ¿Por dónde empezar?

### **Para usuarios que quieren resultados rápidos:**
1. Abre `uso_general.ipynb`
2. Carga tus datos
3. Ejecuta
4. ¡Listo! Ya tienes tu métrica representativa con explicación completa

### **Para usuarios que quieren entender la metodología:**
1. Revisa `metodologia_ejemplo.ipynb` - aquí está la historia completa
2. Ve el análisis paso a paso con datos reales de salarios mexicanos
3. Entiende **por qué** funciona y **cuándo** usar cada configuración

### **Para desarrolladores y académicos:**
1. Explora `defs.py` para ver la implementación completa
2. Modifica y experimenta con los parámetros
3. Cita el trabajo si lo usas en investigación académica

## 💰 Un ejemplo real que lo dice todo

**Datos de salarios mexicanos (ENOE 2023):**
- Media aritmética: $59.07/hora ← *¿Realmente representa al trabajador típico?*
- Mediana: $42.71/hora ← *Ignora información valiosa de la distribución*
- **Métrica ponderada: $43.04/hora** ← *¡El dato más representativo!*

**¿La diferencia?** unos cuantos pesos/hora puede parecer poco, pero multiplicado por 58 millones de trabajadores representa **miles de millones** de pesos en diferencias de estimaciones poblacionales.

## 🌟 ¿Por qué este proyecto es especial?

1. **🎯 Nació de una necesidad real**, (y de un capricho personal)
2. **🤖 Es completamente automático** - no necesitas ser estadístico para usarlo
3. **📚 Tiene fundamento teórico sólido** - (se acepta ayuda para citar correctamente)
4. **🔧 Es flexible** - desde uso automático hasta control total avanzado
5. **🌍 Tiene impacto real** - puede cambiar cómo interpretamos datos importantes
6. **🎓 Es educativo** - aprendes estadística robusta mientras lo usas

---

## 🤝 ¿Cómo puedes ayudar?

- **Usa la herramienta** con tus propios datos y comparte los resultados
- **Reporta problemas** o sugiere mejoras
- **Comparte el proyecto** con quien pueda beneficiarse
- **Contribuye** con nuevas funcionalidades o casos de uso

---

**¿Tu promedio no cuenta toda la historia? Ahora ya sabes qué hacer.** 📊✨

*Este proyecto busca democratizar el análisis estadístico robusto, brindando herramientas de nivel académico pero con la simplicidad que necesita cualquier analista, investigador, o tomador de decisiones.*
