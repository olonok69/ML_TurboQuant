# Caso de estudio: cuando comprimir vectores no ayuda

**Una comprobación real de TurboQuant en dos funciones de similitud de una plataforma de documentos legales. Conclusión para ambas: no ayuda.**

Este capítulo complementa la guía técnica. Las secciones 5 y 6 de la guía muestran dónde brilla TurboQuant:
la caché KV y los índices vectoriales grandes. Aquí se muestra el caso contrario, que un taller necesita
igual: cómo saber, **antes** de comprimir nada, si la compresión de vectores puede mover el problema que
tienes. El código está en `es_bench/` en este repositorio y funciona en un portátil.

> **En pocas palabras.** Teníamos dos funciones de "buscar elementos similares" lentas o costosas y una técnica
> nueva de compresión. Antes de medir la compresión hicimos tres preguntas: dónde viven los vectores, qué
> puntuación muestra realmente el producto y qué es lo que de verdad es lento. En una función las respuestas
> descartaron la compresión por completo. En la otra mostraron que no es lo primero que hay que arreglar.

---

## 1. Las dos funciones

| | A. Provisiones similares | B. Respuestas sugeridas en memos de comentarios |
|---|---|---|
| Qué ve el usuario | "Provisiones similares a esta, por encima del X %" en toda la base de provisiones de una firma | Respuestas pasadas a preguntas parecidas mientras se responde un comentario nuevo |
| Puntuación oficial | **Similitud por distancia de edición** (`rapidfuzz.fuzz.ratio`, 0–100) sobre el texto limpio | **Coseno** entre embeddings de OpenAI (3.072 dimensiones) |
| Dónde vive | Una matriz de pares precalculada en PostgreSQL; se guarda si ≥ 30 y la interfaz usa 70 por defecto | Una colección en una base de datos vectorial; búsqueda **exacta** (fuerza bruta) filtrada por firma |
| Problema | Crecimiento del almacenamiento y escrituras lentas: cada provisión nueva se compara con todas las de la firma | Latencia por petición (llamada de embedding, búsqueda vectorial, re-ranker, comprobación de relevancia con LLM) |
| ¿Hay vectores? | Existen embeddings en el índice de búsqueda, pero **ninguna función los lee** | Sí, son la búsqueda |

---

## 2. Pregunta 1: ¿dónde viven los vectores?

El primer plan de benchmark suponía una colección de embeddings de provisiones en la base de datos
vectorial, con un umbral de coseno para la fusión automática. Al leer el código se vio que no existe
ninguna de las dos. Los embeddings de provisiones se escriben en el índice de **Elasticsearch**
(`text-embedding-3-small`, 1.536 dimensiones), y la única función que los leía, un modo de búsqueda
semántica, se había retirado meses antes. Tanto "provisiones similares" como la fusión automática puntúan
pares con distancia de edición.

**Lección.** Un benchmark vale lo que vale su premisa. Cinco minutos leyendo dónde se escriben y se leen los
vectores evitaron medir una colección que no existe. El benchmark se reescribió para Elasticsearch
(`es_bench/`).

---

## 3. El experimento (`es_bench/`)

**Datos.** 2.501 textos de provisiones reales distintos, limpiados exactamente como los limpia la
plataforma, con embeddings del mismo modelo e indexados en un Elasticsearch 8.18 local con el mismo mapping.

**Verdad de referencia.** La puntuación que usa el producto: `fuzz.ratio` exacto de 500 provisiones contra
las 2.501.

**Pregunta.** Si en lugar de puntuar todos los pares usáramos vecinos por coseno (float32 o comprimidos) o
candidatos por trigramas para elegir *qué* pares puntuar, ¿cuántos de los pares reales encontraríamos?

**Primero, validar.** Antes de confiar en ningún número se ejecutan 12 comprobaciones con respuesta conocida
(`canary.py`). Por ejemplo: los trigramas coinciden con `pg_trgm` de PostgreSQL a 5·10⁻⁹, un índice
exacto de Elasticsearch reproduce el top-k exacto y la carga de la muestra devuelve exactamente las filas
esperadas. Detectaron dos errores del propio benchmark antes de que produjera un solo resultado.

### 3.1 ¿Cuántos pares son "similares"? El número que lo decide todo

| Se guarda si fuzz.ratio ≥ | 30 | 40 | 50 | 60 | 70 |
|---|---|---|---|---|---|
| Porcentaje de pares guardados | **70 %** | 45 % | **1,6 %** | 0,8 % | 0,7 % |

Dos provisiones legales largas *sin relación* ya puntúan alrededor de 38, porque la distancia de edición
sobre texto largo en inglés tiene una línea base alta. Con un umbral de 30, la matriz de similitud
"dispersa" es en realidad casi **densa**: para una firma con 500.000 provisiones son unos 87.000 millones de
pares. Con 50 son unos 2.000 millones.

### 3.2 ¿Puede un filtro de candidatos encontrar los pares reales?

Porcentaje de los pares con fuzz.ratio ≥ 70 (el valor por defecto de la interfaz) que aparecen entre los
candidatos de cada provisión:

| Candidatos por provisión | 10 | 50 | 100 | 200 |
|---|---|---|---|---|
| Vecinos por coseno, float32 | 20 % | 54 % | 82 % | 99,8 % |
| Vecinos por coseno, TurboQuant 2 bits | 20 % | 54 % | 82 % | 99,8 % |
| Vecinos por trigramas (`pg_trgm`) | 20 % | 55 % | 82 % | **100 %** |

Con un umbral de 30 todos los filtros fallan, porque los "vecinos" son el 70 % de la firma.

### 3.3 ¿Cambia algo la compresión?

| Método | Memoria frente a float32 | Coincidencia del top-10 con float32 (sin re-scoring → re-scoring 2×) | Recall de pares fuzz ≥ 70 @200 |
|---|---|---|---|
| float32 | 1× | 1,000 | 99,8 % |
| int8 escalar | 4× menos | 0,949 → 0,999 | 99,8 % |
| binario 1 bit | 32× menos | 0,824 → 0,962 | 99,7 % |
| TurboQuant 4 bits | 8× menos | **0,966 → 1,000** | 99,8 % |
| TurboQuant 2 bits | 16× menos | **0,902 → 0,992** | 99,8 % |

TurboQuant cumple lo que promete el artículo: con cada presupuesto de bits conserva los vecinos por coseno
mejor que int8 o binario. Pero el recall de los pares que el producto puntúa no se mueve, porque los pares
que importan son casi duplicados que todos los métodos encuentran dentro de 200 candidatos.

Las opciones propias de Elasticsearch dan la misma imagen (índice local, coincidencia del top-10 con la
búsqueda exacta): `hnsw` 0,993, `int8_hnsw` 0,986, `int4_hnsw` 0,950 (0,996 con re-scoring), `bbq_hnsw` 0,878
(0,995 con re-scoring). Ojo: Elasticsearch 8.18 ya aplica `int8_hnsw` por defecto cuando el mapping no elige
nada, así que muchos índices están comprimidos sin que nadie lo haya decidido.

---

## 4. Conclusiones

**A. Provisiones similares: TurboQuant no puede ayudar.**
* La puntuación oficial es distancia de edición, no coseno. Los vectores comprimidos solo podrían elegir
  candidatos, y la compresión cambia su recall menos de un 0,3 %, aunque sí reordena el top-10 por coseno. El límite
  es la diferencia entre coseno y distancia de edición, no la precisión de los vectores.
* El coste es **cuántos pares se guardan** (70 % con umbral 30). Ninguna técnica vectorial cambia ese número.
  Las palancas reales son el umbral (una decisión de producto) y un filtro de candidatos. Los trigramas, ya
  disponibles en PostgreSQL con índice, funcionan tan bien como los embeddings.

**B. Sugerencias en memos: también medido (`memo_bench/`). TurboQuant tampoco ayuda aquí.**

Se reconstruyó en local el camino de los memos: 1.238 comentarios reales (685 preguntas, 553 respuestas), el
modelo de la plataforma (`text-embedding-3-large`, 3.072 dimensiones), la misma versión de Qdrant, búsqueda
exacta con filtro por firma, excluyendo el propio hilo, 50 resultados y umbrales 0,5 y luego 0,42. Primero, una
comprobación con respuesta conocida: los resultados de Qdrant coinciden con una búsqueda exacta en numpy en 200
de 200 consultas.

*Dónde se va el tiempo en una petición de sugerencias:*

| Paso | Tiempo (p50) |
|---|---|
| Embedding del comentario nuevo (llamada a la API) | ~240 ms |
| Búsqueda vectorial, 1.238 memos, exacta | **~14 ms** |
| Comprobación de relevancia con LLM sobre los 12 primeros (llamada de tamaño similar) | **~3.800 ms** |

La búsqueda vectorial es mucho menos del 1 % de la petición, y las sugerencias se calculan en segundo plano y se
guardan en caché, no mientras el usuario espera.

*¿Cuándo importaría la búsqueda vectorial?* Todos los memos en una sola firma (el peor caso para una búsqueda exacta):

| Memos en la firma | Búsqueda exacta p50 | RAM de vectores, float32 | Con el int8 propio de Qdrant |
|---|---|---|---|
| 10.000 | 21 ms | 117 MB | 29 MB |
| 50.000 | 53 ms | 586 MB | 146 MB |
| 200.000 | 339 ms | 2,3 GB | 0,6 GB (255 ms) |

Solo hacia 200.000 comentarios en una única firma la búsqueda llega a un tercio de segundo, todavía muy por debajo
del paso del LLM. Incluso entonces, las opciones propias de Qdrant (un índice HNSW en lugar de búsqueda exacta,
cuantización int8) ya lo resuelven con la versión que se usa hoy.

*¿Cambia la compresión las sugerencias?* Umbral 0,5, pares del top 50 de cada pregunta:

| Método | Memoria por vector | Coincidencia del top-10 | Sugerencias perdidas en 0,5 |
|---|---|---|---|
| float32 | 12 KB | 1,000 | 0 |
| int8 escalar (offline) | 3 KB | 0,965 | 1.438 de 22.193 (6,5 %) |
| **TurboQuant 4 bits** | 1,5 KB | **0,983** | **244 (1,1 %)** |
| TurboQuant 2 bits | 0,75 KB | 0,945 | 3.574 (16 %) |
| int8 / binario de Qdrant **con re-scoring** | — | — | **0** |

TurboQuant vuelve a superar al int8 simple en la misma tarea, como promete el artículo. Pero la cuantización
propia de Qdrant con re-scoring no pierde nada y no necesita código nuevo, así que no queda hueco que TurboQuant
pueda llenar.

*Lo que sí encontró el benchmark:* dos memos al azar ya puntúan 0,38 de media (percentil 95: 0,56), así que el
umbral 0,5 deja pasar unos 152 candidatos por pregunta. La calidad de las sugerencias depende del re-ranker y
de la comprobación con LLM, no de la precisión de los vectores. Ahí es donde rendiría el trabajo de mejora.

---

## 5. Lista de comprobación: antes de comprimir vectores

1. **¿Dónde se escriben y se leen los vectores?** Si nadie los lee, la pregunta es si seguir pagándolos, no
   cómo comprimirlos.
2. **¿La puntuación del producto es vectorial?** Si es distancia de edición, BM25 o una regla de negocio, los
   vectores solo pueden prefiltrar, y un prefiltro léxico puede hacerlo igual de bien.
3. **¿Qué domina el coste?** Cuenta elementos guardados, llamadas a modelos y viajes de red. La compresión
   reduce bytes por vector, no el número de nada.
4. **¿La búsqueda es exacta o aproximada?** La búsqueda exacta sobre conjuntos pequeños filtrados rara vez
   está limitada por memoria.
5. **Valida los instrumentos.** Comprueba cada medición con una respuesta conocida antes de confiar en ella.
   Las nuestras detectaron dos errores que, si no, habrían producido tablas plausibles pero falsas.

## 6. Cómo ejecutarlo

Ver `es_bench/README.md`: `docker compose up -d`, `python canary.py` (debe imprimir `ALL PASS`) y luego
`python run_es_bench.py --es <origen> --local-es http://127.0.0.1:9201 --out <carpeta>`. Del clúster de
origen solo se lee; los índices solo se crean en el clúster local.
