"""Generate the SVG figures of the TurboQuant technical guide, in English and Spanish.

Run from the repository root:  python docs/guide/make_figures.py
Writes docs/guide/img/en/*.svg and docs/guide/img/es/*.svg.
The rotation, bias and bounds figures are computed live with turboquant_core.py,
so they show what the demo code actually does.
"""
from __future__ import annotations

import math
import os
import sys
from html import escape

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
import torch  # noqa: E402
from turboquant_core import TurboQuant, lloyd_max_codebook, random_rotation  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "img")

# Palette shared with the slide deck
NAVY, INK, MUTED = "#0F1B2D", "#1B2433", "#5F6979"
ORANGE, PEACH, PEACHBG = "#B4521A", "#F2A35E", "#FBE9DC"
CREAM, CARD, BORDER = "#F6F5F0", "#FDFCF9", "#E2DFD6"
BLUE, BLUEBG, GREY = "#2F6DB5", "#DCE7F5", "#C9CED6"
FONT = "IBM Plex Sans, Segoe UI, Helvetica, Arial, sans-serif"
MONO = "JetBrains Mono, Consolas, Menlo, monospace"


# ---------------------------------------------------------------------------------------
# tiny SVG helpers
# ---------------------------------------------------------------------------------------
class Svg:
    def __init__(self, w, h, title):
        self.w, self.h, self.parts = w, h, []
        self.title = title
        self.rect(0, 0, w, h, CREAM, rx=0)

    def add(self, s):
        self.parts.append(s)

    def rect(self, x, y, w, h, fill, stroke=None, rx=10, sw=1.5, dash=None, opacity=None):
        st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        if dash:
            st += f' stroke-dasharray="{dash}"'
        op = f' opacity="{opacity}"' if opacity is not None else ""
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"{st}{op}/>')

    def text(self, x, y, s, size=15, color=INK, weight=400, anchor="start", mono=False, italic=False):
        fam = MONO if mono else FONT
        it = ' font-style="italic"' if italic else ""
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{fam}" font-size="{size}" font-weight="{weight}" '
                 f'fill="{color}" text-anchor="{anchor}"{it}>{escape(s)}</text>')

    def lines(self, x, y, s, size=15, color=INK, weight=400, anchor="start", width=30, lh=1.3, mono=False):
        """Wrap s to about `width` characters per line; returns the y after the block."""
        out, cur = [], ""
        for para in s.split("\n"):
            cur = ""
            for word in para.split(" "):
                if cur and len(cur) + 1 + len(word) > width:
                    out.append(cur)
                    cur = word
                else:
                    cur = (cur + " " + word) if cur else word
            out.append(cur)
        for i, ln in enumerate(out):
            self.text(x, y + i * size * lh, ln, size, color, weight, anchor, mono)
        return y + len(out) * size * lh

    def line(self, x1, y1, x2, y2, color=MUTED, sw=1.5, dash=None, arrow=False):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        a = ' marker-end="url(#arr)"' if arrow else ""
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{sw}"{d}{a}/>')

    def arrow(self, x1, y1, x2, y2, color=ORANGE, sw=2.5):
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{sw}" marker-end="url(#arr)"/>')

    def path(self, d, stroke=INK, fill="none", sw=2, dash=None, opacity=None):
        ds = f' stroke-dasharray="{dash}"' if dash else ""
        op = f' opacity="{opacity}"' if opacity is not None else ""
        self.add(f'<path d="{d}" stroke="{stroke}" fill="{fill}" stroke-width="{sw}"{ds}{op}/>')

    def circle(self, x, y, r, fill, stroke=None):
        st = f' stroke="{stroke}" stroke-width="1.5"' if stroke else ""
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}"{st}/>')

    def box(self, x, y, w, h, title, body=None, dark=False, accent=False, tsize=15, bsize=13, mono=False):
        """Card with a wrapped title and an optional wrapped body. Returns the y below the text."""
        fill = NAVY if dark else (PEACHBG if accent else CARD)
        self.rect(x, y, w, h, fill, None if dark else BORDER)
        tc = PEACH if dark else NAVY
        bc = "#D9DEE6" if dark else "#3A4556"
        yy = self.lines(x + 14, y + 24, title, tsize, tc, 600, width=int((w - 28) / (tsize * 0.56)))
        if body:
            yy = self.lines(x + 14, yy + 2, body, bsize, bc, width=int((w - 28) / (bsize * (0.62 if mono else 0.52))), mono=mono)
        return yy

    def save(self, path):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}" '
                f'role="img" aria-label="{escape(self.title)}">\n<title>{escape(self.title)}</title>\n'
                f'<defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                f'<path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker></defs>\n')
        body = "\n".join(self.parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(head + body + "\n</svg>\n")


def header(s, title, subtitle=None):
    s.text(32, 44, title, 22, NAVY, 600)
    if subtitle:
        s.text(32, 70, subtitle, 14.5, MUTED)


# ---------------------------------------------------------------------------------------
# strings
# ---------------------------------------------------------------------------------------
T = {
    "en": dict(
        f00_t="TurboQuant in one picture",
        f00_s="Every vector goes through the same three steps. Nothing is learned from the data.",
        f00_1="1. Any vector", f00_1b="An embedding, a key or a value: hundreds of numbers, some of them very large (outliers).",
        f00_2="2. Random rotation", f00_2b="Spin it with a fixed random rotation. Now every number follows the same, known bell curve.",
        f00_3="3. Round with one ruler", f00_3b="Round each number to the nearest of 2ᵇ fixed values, computed once for that bell curve.",
        f00_4="4. Store a few bits", f00_4b="b bits per number plus one 16-bit length. 4 bits instead of 16 is about 4x smaller.",
        f00_foot="Optional stage 2 (TurboQuant_prod): spend 1 more bit on the leftover error to make similarity scores unbiased.",
        f01_t="The memory problem",
        f01_s="Memory, not compute, limits long context and large indexes.",
        f01_l="Llama-3.1-8B in FP16: the KV cache grows with every token",
        f01_w="Model weights", f01_k8="KV cache, 8k tokens", f01_k32="KV cache, 32k tokens", f01_k128="KV cache, 128k tokens",
        f01_tq="128k tokens, TurboQuant 3.5-bit",
        f01_r="Vector index: 1 million OpenAI embeddings (1536-d)",
        f01_f32="float32", f01_tq4="TurboQuant 4-bit",
        f02_t="Quantization is rounding",
        f02_s="Store which allowed value is nearest, instead of the full number.",
        f02_orig="Original number (16 bits)", f02_levels="2 bits = 4 allowed values", f02_code="Stored code: 2 bits",
        f02_err="rounding error", f02_note="More bits mean more allowed values and a smaller error. Each extra bit cuts the squared error by about 4x.",
        f03_t="Three papers, one idea",
        f03_s="Apply a random transform first, so the data's distribution is known in advance.",
        f03_q="QJL", f03_qd="Jun 2024 · arXiv 2406.03482",
        f03_qb="Random projection, keep only the sign bit. Unbiased inner products with zero overhead. 3-bit KV cache, over 5x less memory.",
        f03_p="PolarQuant", f03_pd="Feb 2025 · arXiv 2502.02617",
        f03_pb="Random preconditioning, then a recursive polar transform. Angles have a known, tight distribution, so no scales are needed. Over 4.2x KV compression.",
        f03_t3="TurboQuant", f03_td="Apr 2025 · arXiv 2504.19874",
        f03_tb="Random rotation plus an optimal scalar quantizer per coordinate, then QJL on the residual. Provably within about 2.7x of the best possible.",
        f03_reuse="reuses QJL as stage 2", f03_idea="same trick: randomize first",
        f04_t="The hidden tax of classic quantization",
        f04_s="Bits needed to store one 128-number vector (one attention head).",
        f04_codes="codes", f04_scales="scales and zero points (fp16)", f04_norm="one norm (fp16)",
        f04_int4="INT4, groups of 32", f04_tq4="TurboQuant 4-bit", f04_int2="INT2, groups of 32", f04_tq2="TurboQuant 2-bit",
        f04_bpc="bits per number",
        f05_t="Step 1: the random rotation",
        f05_s="The same 64-number vector before and after one random rotation (scaled by √d).",
        f05_b="Before: a few huge outliers", f05_a="After: every coordinate looks alike",
        f05_note="A rotation keeps lengths and angles, so no information is lost. It only spreads the energy evenly over all coordinates.",
        f06_t="Step 2: one fixed ruler (Lloyd-Max codebook)",
        f06_s="2-bit example. After rotation every coordinate follows this bell curve, so the best 4 values are known in advance.",
        f06_c="centroid (stored value)", f06_bd="boundary",
        f06_ex="Example: coordinate 0.70 falls in region 10, decoded as 0.45",
        f06_axis="coordinate × √d",
        f07_t="TurboQuant_mse, step by step (Algorithm 1)",
        f07_enc="Encode (on write)", f07_dec="Decode (on read)",
        f07_e1="Measure length", f07_e1b="‖x‖ stored in fp16",
        f07_e2="Normalize", f07_e2b="u = x / ‖x‖",
        f07_e3="Rotate", f07_e3b="y = Π · u",
        f07_e4="Round", f07_e4b="nearest of 2ᵇ centroids, per coordinate",
        f07_e5="Pack bits", f07_e5b="b·d bits + 16",
        f07_d1="Unpack", f07_d1b="codes → centroids",
        f07_d2="Rotate back", f07_d2b="Πᵀ · ỹ",
        f07_d3="Rescale", f07_d3b="× ‖x‖",
        f07_d4="Approximate vector", f07_d4b="x̃ ≈ x",
        f07_foot="Π is drawn once from a seed and shared by every vector. The centroids depend only on d and b.",
        f08_t="Stage 2: unbiased inner products (TurboQuant_prod, Algorithm 2)",
        f08_s="Use b−1 bits for the MSE stage, then one QJL sign bit per coordinate on what is left over.",
        f08_1="MSE stage, b−1 bits", f08_1b="x̃_mse = Q_mse(x)",
        f08_2="Residual", f08_2b="r = x − x̃_mse  (small)",
        f08_3="QJL, 1 bit", f08_3b="signs = sign(S · r), keep ‖r‖",
        f08_4="Unbiased estimate", f08_4b="⟨y, x̃_mse⟩ + √(π/2)/d · ‖r‖ · ⟨S·y, signs⟩",
        f08_h="Inner-product error at 2 bits, 20,000 vectors (measured with turboquant_core.py)",
        f08_mse="TurboQuant_mse: shifted (biased)", f08_prod="TurboQuant_prod: centred on 0 (unbiased)",
        f09_t="How close to optimal?",
        f09_s="MSE for unit vectors (d = 128). Measured with turboquant_core.py against the paper's bounds.",
        f09_lo="lower bound 4⁻ᵇ (no quantizer can do better)", f09_up="upper bound √3π/2 · 4⁻ᵇ (proved for TurboQuant)",
        f09_me="TurboQuant_mse, measured", f09_x="bits per coordinate (b)", f09_y="MSE (log scale)",
        f10_t="Use case 1: the KV cache in LLM inference",
        f10_s="Each new token looks back at every earlier token. Their keys (K) and values (V) are kept in the KV cache.",
        f10_tokens=["The", "capital", "of", "France", "is"], f10_next="Paris",
        f10_cache="KV cache (one K and one V per token, per layer, per head)",
        f10_q="new query q", f10_att="attention: compare q with every K, mix the Vs",
        f10_w="write: TurboQuant encodes K and V (4 bits instead of 16)",
        f10_r="read: decode, or score directly from the codes in a fused kernel",
        f10_foot="The cache grows with context length and with the number of users; every generated token re-reads all of it.",
        f11_t="Outlier split: fractional bit-widths",
        f11_s="Some channels of keys carry much more energy. Give them one more bit (Section 4.3 of the paper).",
        f11_o="32 outlier channels × 3 bits", f11_r="96 regular channels × 2 bits",
        f11_eq="(32·3 + 96·2) / 128 = 2.25 bits of codes",
        f11_eq2="+ two fp16 norms (32 bits / 128) = 2.5 bits per channel",
        f11_35="3.5-bit setting in the demo: 64 channels × 4 bits + 64 × 3 bits (+ norms = 3.75 measured)",
        f12_t="Use case 2: vector search and RAG",
        f12_s="Find the stored vectors with the highest inner product (similarity) to a query.",
        f12_docs="Documents", f12_emb="Embedding model", f12_idx="Compressed index", f12_idxb="TurboQuant codes, 8x to 16x smaller",
        f12_q="Question", f12_scan="Scan codes", f12_scanb="score every vector, keep top 100",
        f12_rr="Exact re-rank (optional)", f12_rrb="re-score 100 candidates with full vectors",
        f12_top="Top 10", f12_llm="to the LLM (RAG)",
        f12_pq="Product quantization: k-means training first (minutes), re-train when data drifts",
        f12_tq="TurboQuant: rotate and round (milliseconds), add vectors the moment they arrive",
        f12_ing="Indexing",
        f13_t="What compression buys: context length",
        f13_s="Llama-3.1-8B, 8 GB of GPU memory reserved for the KV cache (a 24 GB GPU after 16 GB of weights).",
        f13_lbl=["FP16", "INT4 + scales (g=32)", "TurboQuant 4-bit", "TurboQuant 3.5-bit", "TurboQuant 2.5-bit"],
        f13_unit="tokens",
        f14_t="Demo 2 sandbox results: recall at equal bits",
        f14_s="Recall@1@1: how often the true nearest neighbour is ranked first. 100k vectors, 1k queries, 384-d.",
        f14_b4="4 bits per dimension", f14_b2="2 bits per dimension",
        f14_m=["turbovec", "TurboQuant (PyTorch)", "FAISS RaBitQ", "FAISS PQ LUT256", "FAISS SQ 4-bit", "FAISS PQ-FastScan"],
        f14_leg1="TurboQuant (no training)", f14_leg2="trained or calibrated baselines",
        f14_build="build time",
    ),
    "es": dict(
        f00_t="TurboQuant en una imagen",
        f00_s="Cada vector pasa por los mismos tres pasos. No se aprende nada de los datos.",
        f00_1="1. Cualquier vector", f00_1b="Un embedding, una clave o un valor: cientos de números, algunos muy grandes (outliers).",
        f00_2="2. Rotación aleatoria", f00_2b="Se gira con una rotación aleatoria fija. Ahora cada número sigue la misma curva de campana conocida.",
        f00_3="3. Redondear con una regla", f00_3b="Cada número se redondea al más cercano de 2ᵇ valores fijos, calculados una sola vez.",
        f00_4="4. Guardar pocos bits", f00_4b="b bits por número más una longitud de 16 bits. 4 bits en lugar de 16: unas 4 veces menos.",
        f00_foot="Etapa 2 opcional (TurboQuant_prod): 1 bit más sobre el error restante para que las similitudes sean insesgadas.",
        f01_t="El problema de la memoria",
        f01_s="La memoria, no el cómputo, limita el contexto largo y los índices grandes.",
        f01_l="Llama-3.1-8B en FP16: la caché KV crece con cada token",
        f01_w="Pesos del modelo", f01_k8="Caché KV, 8k tokens", f01_k32="Caché KV, 32k tokens", f01_k128="Caché KV, 128k tokens",
        f01_tq="128k tokens, TurboQuant 3,5 bits",
        f01_r="Índice vectorial: 1 millón de embeddings de OpenAI (1536-d)",
        f01_f32="float32", f01_tq4="TurboQuant 4 bits",
        f02_t="Cuantizar es redondear",
        f02_s="Se guarda cuál de los valores permitidos es el más cercano, no el número completo.",
        f02_orig="Número original (16 bits)", f02_levels="2 bits = 4 valores permitidos", f02_code="Código guardado: 2 bits",
        f02_err="error de redondeo", f02_note="Más bits dan más valores permitidos y menos error. Cada bit extra divide el error cuadrático por unas 4.",
        f03_t="Tres artículos, una idea",
        f03_s="Aplicar primero una transformación aleatoria, para conocer de antemano la distribución de los datos.",
        f03_q="QJL", f03_qd="Jun 2024 · arXiv 2406.03482",
        f03_qb="Proyección aleatoria y solo el bit de signo. Productos internos insesgados sin sobrecoste. Caché KV a 3 bits, más de 5x menos memoria.",
        f03_p="PolarQuant", f03_pd="Feb 2025 · arXiv 2502.02617",
        f03_pb="Precondicionamiento aleatorio y transformación polar recursiva. Los ángulos tienen una distribución conocida, sin escalas. Más de 4,2x de compresión.",
        f03_t3="TurboQuant", f03_td="Abr 2025 · arXiv 2504.19874",
        f03_tb="Rotación aleatoria más un cuantizador escalar óptimo por coordenada, y QJL sobre el residuo. Demostrado a menos de 2,7x del óptimo.",
        f03_reuse="reutiliza QJL como etapa 2", f03_idea="mismo truco: aleatorizar primero",
        f04_t="El impuesto oculto de la cuantización clásica",
        f04_s="Bits necesarios para guardar un vector de 128 números (una cabeza de atención).",
        f04_codes="códigos", f04_scales="escalas y puntos cero (fp16)", f04_norm="una norma (fp16)",
        f04_int4="INT4, grupos de 32", f04_tq4="TurboQuant 4 bits", f04_int2="INT2, grupos de 32", f04_tq2="TurboQuant 2 bits",
        f04_bpc="bits por número",
        f05_t="Paso 1: la rotación aleatoria",
        f05_s="El mismo vector de 64 números antes y después de una rotación aleatoria (escalado por √d).",
        f05_b="Antes: unos pocos outliers enormes", f05_a="Después: todas las coordenadas se parecen",
        f05_note="Una rotación conserva longitudes y ángulos, así que no se pierde información. Solo reparte la energía por igual entre las coordenadas.",
        f06_t="Paso 2: una regla fija (codebook de Lloyd-Max)",
        f06_s="Ejemplo de 2 bits. Tras la rotación cada coordenada sigue esta campana, así que los 4 mejores valores se conocen de antemano.",
        f06_c="centroide (valor guardado)", f06_bd="frontera",
        f06_ex="Ejemplo: la coordenada 0,70 cae en la región 10 y se decodifica como 0,45",
        f06_axis="coordenada × √d",
        f07_t="TurboQuant_mse, paso a paso (Algoritmo 1)",
        f07_enc="Codificar (al escribir)", f07_dec="Decodificar (al leer)",
        f07_e1="Medir longitud", f07_e1b="‖x‖ guardada en fp16",
        f07_e2="Normalizar", f07_e2b="u = x / ‖x‖",
        f07_e3="Rotar", f07_e3b="y = Π · u",
        f07_e4="Redondear", f07_e4b="centroide más cercano de 2ᵇ, por coordenada",
        f07_e5="Empaquetar", f07_e5b="b·d bits + 16",
        f07_d1="Desempaquetar", f07_d1b="códigos → centroides",
        f07_d2="Rotar de vuelta", f07_d2b="Πᵀ · ỹ",
        f07_d3="Reescalar", f07_d3b="× ‖x‖",
        f07_d4="Vector aproximado", f07_d4b="x̃ ≈ x",
        f07_foot="Π se genera una vez a partir de una semilla y la comparten todos los vectores. Los centroides solo dependen de d y b.",
        f08_t="Etapa 2: productos internos insesgados (TurboQuant_prod, Algoritmo 2)",
        f08_s="b−1 bits para la etapa MSE y luego un bit de signo QJL por coordenada sobre lo que queda.",
        f08_1="Etapa MSE, b−1 bits", f08_1b="x̃_mse = Q_mse(x)",
        f08_2="Residuo", f08_2b="r = x − x̃_mse  (pequeño)",
        f08_3="QJL, 1 bit", f08_3b="signos = sign(S · r), guardar ‖r‖",
        f08_4="Estimación insesgada", f08_4b="⟨y, x̃_mse⟩ + √(π/2)/d · ‖r‖ · ⟨S·y, signos⟩",
        f08_h="Error del producto interno a 2 bits, 20.000 vectores (medido con turboquant_core.py)",
        f08_mse="TurboQuant_mse: desplazado (sesgado)", f08_prod="TurboQuant_prod: centrado en 0 (insesgado)",
        f09_t="¿Qué tan cerca del óptimo?",
        f09_s="MSE para vectores unitarios (d = 128). Medido con turboquant_core.py frente a las cotas del artículo.",
        f09_lo="cota inferior 4⁻ᵇ (ningún cuantizador lo hace mejor)", f09_up="cota superior √3π/2 · 4⁻ᵇ (demostrada para TurboQuant)",
        f09_me="TurboQuant_mse, medido", f09_x="bits por coordenada (b)", f09_y="MSE (escala log)",
        f10_t="Caso de uso 1: la caché KV en la inferencia de LLM",
        f10_s="Cada token nuevo mira a todos los anteriores. Sus claves (K) y valores (V) se guardan en la caché KV.",
        f10_tokens=["La", "capital", "de", "Francia", "es"], f10_next="París",
        f10_cache="Caché KV (una K y una V por token, por capa y por cabeza)",
        f10_q="nueva consulta q", f10_att="atención: comparar q con cada K, mezclar las V",
        f10_w="escritura: TurboQuant codifica K y V (4 bits en lugar de 16)",
        f10_r="lectura: decodificar, o puntuar directamente desde los códigos con un kernel fusionado",
        f10_foot="La caché crece con la longitud del contexto y con el número de usuarios; cada token generado la vuelve a leer entera.",
        f11_t="División de outliers: anchos de bit fraccionarios",
        f11_s="Algunos canales de las claves tienen mucha más energía. Reciben un bit más (Sección 4.3 del artículo).",
        f11_o="32 canales outlier × 3 bits", f11_r="96 canales normales × 2 bits",
        f11_eq="(32·3 + 96·2) / 128 = 2,25 bits de códigos",
        f11_eq2="+ dos normas fp16 (32 bits / 128) = 2,5 bits por canal",
        f11_35="Configuración de 3,5 bits en la demo: 64 canales × 4 bits + 64 × 3 bits (+ normas = 3,75 medido)",
        f12_t="Caso de uso 2: búsqueda vectorial y RAG",
        f12_s="Encontrar los vectores guardados con mayor producto interno (similitud) con una consulta.",
        f12_docs="Documentos", f12_emb="Modelo de embeddings", f12_idx="Índice comprimido", f12_idxb="códigos TurboQuant, de 8x a 16x más pequeño",
        f12_q="Pregunta", f12_scan="Recorrer códigos", f12_scanb="puntuar cada vector, quedarse con 100",
        f12_rr="Re-ranking exacto (opcional)", f12_rrb="re-puntuar 100 candidatos con vectores completos",
        f12_top="Top 10", f12_llm="al LLM (RAG)",
        f12_pq="Cuantización por producto: primero entrenar k-means (minutos), re-entrenar si los datos cambian",
        f12_tq="TurboQuant: rotar y redondear (milisegundos), añadir vectores en cuanto llegan",
        f12_ing="Indexación",
        f13_t="Lo que da la compresión: longitud de contexto",
        f13_s="Llama-3.1-8B, 8 GB de memoria de GPU para la caché KV (una GPU de 24 GB tras 16 GB de pesos).",
        f13_lbl=["FP16", "INT4 + escalas (g=32)", "TurboQuant 4 bits", "TurboQuant 3,5 bits", "TurboQuant 2,5 bits"],
        f13_unit="tokens",
        f14_t="Resultados de la demo 2 (sandbox): recall con los mismos bits",
        f14_s="Recall@1@1: con qué frecuencia el vecino más cercano real queda primero. 100k vectores, 1k consultas, 384-d.",
        f14_b4="4 bits por dimensión", f14_b2="2 bits por dimensión",
        f14_m=["turbovec", "TurboQuant (PyTorch)", "FAISS RaBitQ", "FAISS PQ LUT256", "FAISS SQ 4 bits", "FAISS PQ-FastScan"],
        f14_leg1="TurboQuant (sin entrenamiento)", f14_leg2="baselines entrenados o calibrados",
        f14_build="tiempo de construcción",
    ),
}


def num(x, lang, nd=0):
    s = f"{x:,.{nd}f}"
    if lang == "es":
        s = s.replace(",", "_").replace(".", ",").replace("_", ".")
    return s


# ---------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------
def fig00(t, lang):
    s = Svg(1000, 350, t["f00_t"])
    header(s, t["f00_t"], t["f00_s"])
    xs, w = [32, 278, 524, 770], 198
    titles = [t["f00_1"], t["f00_2"], t["f00_3"], t["f00_4"]]
    bodies = [t["f00_1b"], t["f00_2b"], t["f00_3b"], t["f00_4b"]]
    spiky = [0.2, 0.1, 2.6, -0.3, 0.15, -2.2, 0.25, 0.1, -0.2, 0.3, 0.05, 1.9]
    even = [0.7, -0.9, 0.4, 1.1, -0.6, 0.9, -1.0, 0.5, -0.4, 0.8, -0.7, 0.6]
    for i, x in enumerate(xs):
        last = i == 3
        s.rect(x, 92, w, 222, NAVY if last else CARD, None if last else BORDER)
        s.lines(x + 14, 116, titles[i], 15, PEACH if last else NAVY, 600, width=22)
        s.lines(x + 14, 236, bodies[i], 12.5, "#D9DEE6" if last else "#3A4556", width=31)
        base = 186  # mini illustration band: y 148..222
        if i in (0, 1):
            vals = spiky if i == 0 else even
            for j, v in enumerate(vals):
                bx = x + 20 + j * 13.5
                h = abs(v) * 14
                s.rect(bx, base - h if v > 0 else base, 9, h, ORANGE if abs(v) > 1.5 else BLUE, rx=1)
            s.line(x + 14, base, x + w - 14, base, MUTED, 1)
        elif i == 2:
            for j, c in enumerate([-1.5, -0.45, 0.45, 1.5]):
                cx = x + w / 2 + c * 52
                s.line(cx, 156, cx, 202, ORANGE, 2)
                s.text(cx, 218, ["00", "01", "10", "11"][j], 12, MUTED, 400, "middle", mono=True)
            s.line(x + 16, 180, x + w - 16, 180, MUTED, 1)
            s.circle(x + w / 2 + 0.7 * 52, 180, 5, BLUE)
        else:
            s.text(x + 14, 176, "10 01 11 00 10 …", 15, PEACH, 600, mono=True)
            s.text(x + 14, 202, "+ ‖x‖ (16 bit)", 13, "#D9DEE6", mono=True)
        if i < 3:
            s.arrow(x + w + 6, 186, x + 246 - 6, 186)
    s.text(32, 338, t["f00_foot"], 13.5, MUTED, italic=True)
    return s


def fig01(t, lang):
    s = Svg(1000, 420, t["f01_t"])
    header(s, t["f01_t"], t["f01_s"])
    s.text(32, 108, t["f01_l"], 15, NAVY, 600)
    rows = [(t["f01_w"], 16, GREY), (t["f01_k8"], 1, BLUE), (t["f01_k32"], 4, BLUE), (t["f01_k128"], 16, ORANGE), (t["f01_tq"], 3.75, "#3C8C5A")]
    scale = 22
    for i, (lbl, gb, col) in enumerate(rows):
        y = 128 + i * 44
        s.text(32, y + 20, lbl, 13.5, INK)
        s.rect(270, y + 4, gb * scale, 24, col, rx=4)
        s.text(270 + gb * scale + 8, y + 22, f"{num(gb, lang, 2 if gb % 1 else 0)} GB", 13.5, INK, 600)
    s.text(32, 372, "128 KB/token = 32 layers × 8 KV heads × 128 dims × (K+V) × 2 bytes", 12.5, MUTED, mono=True)
    # right panel
    x0 = 700
    s.rect(x0, 96, 268, 296, CARD, BORDER)
    s.lines(x0 + 16, 124, t["f01_r"], 14.5, NAVY, 600, width=30)
    for i, (lbl, gb, col) in enumerate([(t["f01_f32"], 6.1, GREY), (t["f01_tq4"], 0.77, ORANGE)]):
        y = 186 + i * 92
        s.text(x0 + 16, y, lbl, 13.5, INK)
        s.rect(x0 + 16, y + 10, gb / 6.1 * 236, 30, col, rx=4)
        s.text(x0 + 16, y + 64, f"{num(gb, lang, 2 if gb < 1 else 1)} GB", 20, NAVY, 700)
    return s


def fig02(t, lang):
    s = Svg(1000, 330, t["f02_t"])
    header(s, t["f02_t"], t["f02_s"])
    x0, x1, y = 120, 880, 190
    def X(v):
        return x0 + (v + 2) / 4 * (x1 - x0)
    s.line(x0, y, x1, y, MUTED, 2)
    for v in [-2, -1, 0, 1, 2]:
        s.line(X(v), y - 5, X(v), y + 5, MUTED, 1)
        s.text(X(v), y + 24, num(v, lang), 12, MUTED, 400, "middle")
    levels = [-1.5, -0.45, 0.45, 1.5]
    codes = ["00", "01", "10", "11"]
    for c, code in zip(levels, codes):
        s.circle(X(c), y, 9, ORANGE)
        s.text(X(c), y - 20, code, 15, ORANGE, 700, "middle", mono=True)
    s.text(X(0), y - 52, t["f02_levels"], 14, ORANGE, 600, "middle")
    v = 0.71
    s.circle(X(v), y, 7, BLUE)
    s.path(f"M{X(v):.1f},{y + 12} C{X(v):.1f},{y + 52} {X(0.45):.1f},{y + 52} {X(0.45):.1f},{y + 14}", BLUE, sw=2)
    s.text((X(v) + X(0.45)) / 2, y + 62, t["f02_err"], 12.5, BLUE, 400, "middle")
    s.rect(32, 96, 230, 52, CARD, BORDER)
    s.text(44, 118, t["f02_orig"], 12.5, MUTED)
    s.text(44, 140, num(0.7109375, lang, 7), 16, BLUE, 600, mono=True)
    s.rect(738, 96, 230, 52, PEACHBG, BORDER)
    s.text(750, 118, t["f02_code"], 12.5, MUTED)
    s.text(750, 140, "10", 16, ORANGE, 700, mono=True)
    s.lines(32, 296, t["f02_note"], 14, INK, width=120)
    return s


def fig03(t, lang):
    s = Svg(1000, 400, t["f03_t"])
    header(s, t["f03_t"], t["f03_s"])
    xs, w = [32, 352, 672], 296
    data = [(t["f03_q"], t["f03_qd"], t["f03_qb"]), (t["f03_p"], t["f03_pd"], t["f03_pb"]), (t["f03_t3"], t["f03_td"], t["f03_tb"])]
    s.line(40, 116, 960, 116, MUTED, 2)
    for i, (x, (nm, dt, body)) in enumerate(zip(xs, data)):
        dark = i == 2
        s.circle(x + 16, 116, 8, ORANGE if dark else NAVY)
        s.text(x + 30, 104, dt, 13, MUTED, mono=True)
        s.rect(x, 134, w, 186, NAVY if dark else CARD, None if dark else BORDER)
        s.text(x + 16, 166, nm, 22, PEACH if dark else NAVY, 700)
        s.lines(x + 16, 194, body, 13.5, "#D9DEE6" if dark else "#3A4556", width=40)
    s.path("M180,322 C180,362 820,362 820,328", ORANGE, sw=2, dash="6 4")
    s.add('<polygon points="814,332 820,322 826,332" fill="#B4521A"/>')
    s.text(500, 384, t["f03_reuse"], 13.5, ORANGE, 600, "middle")
    return s


def fig04(t, lang):
    s = Svg(1000, 360, t["f04_t"])
    header(s, t["f04_t"], t["f04_s"])
    rows = [(t["f04_int4"], 512, 128, False), (t["f04_tq4"], 512, 16, True), (t["f04_int2"], 256, 128, False), (t["f04_tq2"], 256, 16, True)]
    sc = 0.85
    for i, (lbl, code, over, tq) in enumerate(rows):
        y = 100 + i * 52 + (14 if i >= 2 else 0)
        s.text(32, y + 22, lbl, 14, NAVY, 600 if tq else 400)
        s.rect(230, y + 4, code * sc, 28, BLUE, rx=3)
        s.text(236, y + 23, f"{code} {t['f04_codes']}", 12.5, "#FFFFFF", 600)
        s.rect(230 + code * sc, y + 4, over * sc, 28, ORANGE if not tq else PEACH, rx=3)
        s.text(230 + (code + over) * sc + 10, y + 23, f"+{over}  =  {num((code + over) / 128, lang, 3)} {t['f04_bpc']}", 13, INK, 600)
    s.rect(230, 322, 16, 14, BLUE, rx=2); s.text(252, 334, t["f04_codes"], 12.5, INK)
    s.rect(340, 322, 16, 14, ORANGE, rx=2); s.text(362, 334, t["f04_scales"], 12.5, INK)
    s.rect(620, 322, 16, 14, PEACH, rx=2); s.text(642, 334, t["f04_norm"], 12.5, INK)
    return s


def fig05(t, lang):
    d = 64
    g = np.random.default_rng(3)
    x = g.normal(0, 0.15, d)
    x[[5, 23, 41]] = [3.2, -2.6, 2.1]
    pi = random_rotation(d, seed=11).numpy()
    y = pi @ (x / np.linalg.norm(x)) * math.sqrt(d)
    xs = x / np.linalg.norm(x) * math.sqrt(d)
    s = Svg(1000, 360, t["f05_t"])
    header(s, t["f05_t"], t["f05_s"])
    for k, (vals, lbl, x0) in enumerate([(xs, t["f05_b"], 32), (y, t["f05_a"], 520)]):
        s.rect(x0, 92, 448, 220, CARD, BORDER)
        s.text(x0 + 16, 118, lbl, 14.5, NAVY, 600)
        base, sc = 215, 14
        s.line(x0 + 14, base, x0 + 434, base, MUTED, 1)
        for sd in (-2, 2):
            s.line(x0 + 14, base - sd * sc, x0 + 434, base - sd * sc, GREY, 1, dash="4 4")
        for j, v in enumerate(vals):
            v = float(np.clip(v, -6.5, 6.5))
            bx = x0 + 18 + j * 6.5
            h = abs(v) * sc
            col = ORANGE if abs(v) > 3 else BLUE
            s.rect(bx, base - h if v > 0 else base, 4.5, max(h, 0.5), col, rx=1)
        s.text(x0 + 438, base - 2 * sc - 3, "±2", 11, MUTED, 400, "end")
    s.arrow(484, 200, 514, 200)
    s.text(499, 186, "Π", 18, ORANGE, 700, "middle")
    s.lines(32, 338, t["f05_note"], 13.5, MUTED, width=140)
    return s


def fig06(t, lang):
    c = lloyd_max_codebook(128, 2) * math.sqrt(128)
    bnd = 0.5 * (c[1:] + c[:-1])
    s = Svg(1000, 380, t["f06_t"])
    header(s, t["f06_t"], t["f06_s"])
    x0, x1, yb, H = 80, 920, 300, 180
    def X(v):
        return x0 + (v + 3.5) / 7 * (x1 - x0)
    def Y(p):
        return yb - p / 0.4 * H
    edges = [-3.5] + list(bnd) + [3.5]
    fills = [BLUEBG, PEACHBG, BLUEBG, PEACHBG]
    for i in range(4):
        pts = [(v, math.exp(-v * v / 2) / math.sqrt(2 * math.pi)) for v in np.linspace(edges[i], edges[i + 1], 60)]
        d = f"M{X(edges[i]):.1f},{yb} " + " ".join(f"L{X(v):.1f},{Y(p):.1f}" for v, p in pts) + f" L{X(edges[i + 1]):.1f},{yb} Z"
        s.path(d, "none", fills[i])
        mid = (max(edges[i], -2.6) + min(edges[i + 1], 2.6)) / 2
        s.text(X(mid), yb - 18, ["00", "01", "10", "11"][i], 16, NAVY, 700, "middle", mono=True)
    pts = [(v, math.exp(-v * v / 2) / math.sqrt(2 * math.pi)) for v in np.linspace(-3.5, 3.5, 200)]
    s.path("M" + " L".join(f"{X(v):.1f},{Y(p):.1f}" for v, p in pts), NAVY, sw=2.5)
    s.line(x0, yb, x1, yb, MUTED, 1.5)
    for b in bnd:
        b = 0.0 if abs(b) < 1e-9 else b
        s.line(X(b), yb + 4, X(b), Y(0.42), MUTED, 1.5, dash="5 4")
        s.text(X(b), Y(0.42) - 6, num(b, lang, 2), 12, MUTED, 400, "middle")
    for v in c:
        s.circle(X(v), yb, 7, ORANGE)
        s.text(X(v), yb + 24, num(v, lang, 2), 13, ORANGE, 700, "middle")
    s.circle(X(0.70), yb, 5, BLUE)
    s.text(x1, yb + 54, t["f06_axis"], 12, MUTED, 400, "end")
    s.circle(92, 362, 6, ORANGE); s.text(104, 367, t["f06_c"], 12.5, INK)
    s.line(300, 362, 322, 362, MUTED, 1.5, dash="5 4"); s.text(330, 367, t["f06_bd"], 12.5, INK)
    s.circle(452, 362, 5, BLUE); s.text(464, 367, t["f06_ex"], 12.5, INK)
    return s


def fig07(t, lang):
    s = Svg(1000, 380, t["f07_t"])
    header(s, t["f07_t"])
    s.text(32, 92, t["f07_enc"], 15, ORANGE, 700)
    w, gap = 168, 28
    enc = [("f07_e1", "f07_e1b"), ("f07_e2", "f07_e2b"), ("f07_e3", "f07_e3b"), ("f07_e4", "f07_e4b"), ("f07_e5", "f07_e5b")]
    for i, (a, b) in enumerate(enc):
        x = 32 + i * (w + gap)
        s.box(x, 104, w, 96, t[a], t[b], dark=(i == 4), mono=i in (1, 2, 4))
        if i < 4:
            s.arrow(x + w + 3, 152, x + w + gap - 3, 152)
    s.text(32, 236, t["f07_dec"], 15, BLUE, 700)
    w2 = 215
    dec = [("f07_d1", "f07_d1b"), ("f07_d2", "f07_d2b"), ("f07_d3", "f07_d3b"), ("f07_d4", "f07_d4b")]
    for i, (a, b) in enumerate(dec):
        x = 32 + i * (w2 + 33)
        s.box(x, 248, w2, 80, t[a], t[b], accent=(i == 3), mono=True)
        if i < 3:
            s.arrow(x + w2 + 3, 288, x + w2 + 30, 288, BLUE)
    s.text(32, 360, t["f07_foot"], 13, MUTED, italic=True)
    return s


def _bias_data():
    torch.manual_seed(0)
    d, n = 128, 20000
    x = torch.randn(n, d); x = x / x.norm(dim=1, keepdim=True)
    y = torch.randn(n, d); y = y / y.norm(dim=1, keepdim=True)
    y = 0.6 * x + 0.8 * y
    qm, qp = TurboQuant(d, 2, "mse", seed=3), TurboQuant(d, 2, "prod", seed=3)
    ip = (y * x).sum(1)
    em = ((y * qm.roundtrip(x)).sum(1) - ip).numpy()
    ep = ((y * qp.roundtrip(x)).sum(1) - ip).numpy()
    return em, ep


def fig08(t, lang, data):
    em, ep = data
    s = Svg(1000, 500, t["f08_t"])
    header(s, t["f08_t"], t["f08_s"])
    gap = 22
    ws = [196, 196, 196, 300]
    keys = [("f08_1", "f08_1b"), ("f08_2", "f08_2b"), ("f08_3", "f08_3b"), ("f08_4", "f08_4b")]
    x = 32
    for i, ((a, b), w) in enumerate(zip(keys, ws)):
        s.box(x, 92, w, 100, t[a], t[b], dark=(i == 3), mono=True, bsize=12.5)
        if i < 3:
            s.arrow(x + w + 2, 142, x + w + gap - 2, 142)
        x += w + gap
    s.text(32, 230, t["f08_h"], 14, NAVY, 600)
    lo_v = float(min(np.quantile(em, 0.001), np.quantile(ep, 0.001)))
    hi_v = float(max(np.quantile(em, 0.999), np.quantile(ep, 0.999)))
    bins = np.linspace(math.floor(lo_v * 20) / 20, math.ceil(hi_v * 20) / 20, 81)
    hm, _ = np.histogram(em, bins); hp, _ = np.histogram(ep, bins)
    top = max(hm.max(), hp.max())
    x0, x1, yb, H = 60, 940, 460, 150
    def X(v):
        return x0 + (v - bins[0]) / (bins[-1] - bins[0]) * (x1 - x0)
    bw = (x1 - x0) / (len(bins) - 1)
    for h, col in [(hm, BLUE), (hp, ORANGE)]:
        for j, c in enumerate(h):
            hh = c / top * H
            s.rect(X(bins[j]) + 0.5, yb - hh, bw - 1, hh, col, rx=0, opacity=0.6)
    s.line(x0, yb, x1, yb, MUTED, 1.5)
    s.line(X(0), yb + 4, X(0), yb - H - 8, INK, 1.5, dash="4 3")
    s.text(X(0), yb + 20, "0", 12, INK, 600, "middle")
    for v in np.arange(-0.4, 0.45, 0.1):
        if abs(v) > 1e-9 and bins[0] <= v <= bins[-1]:
            s.text(X(v), yb + 20, num(v, lang, 1), 12, MUTED, 400, "middle")
    s.line(X(em.mean()), yb - H - 6, X(em.mean()), yb, BLUE, 2)
    mean = "mean" if lang == "en" else "media"
    s.rect(x0 + 6, 246, 16, 12, BLUE, rx=2, opacity=0.6); s.text(x0 + 28, 257, f"{t['f08_mse']}  ({mean} {num(em.mean(), lang, 3)})", 12.5, INK)
    s.rect(x0 + 6, 266, 16, 12, ORANGE, rx=2, opacity=0.6); s.text(x0 + 28, 277, f"{t['f08_prod']}  ({mean} {num(ep.mean(), lang, 3)})", 12.5, INK)
    return s


def fig09(t, lang):
    xs = torch.randn(20000, 128); xs = xs / xs.norm(dim=1, keepdim=True)
    meas = [((xs - TurboQuant(128, b, seed=1).roundtrip(xs)) ** 2).sum(1).mean().item() for b in range(1, 6)]
    s = Svg(1000, 420, t["f09_t"])
    header(s, t["f09_t"], t["f09_s"])
    x0, x1, y0, y1 = 110, 640, 100, 360
    lo_e, hi_e = -4, 0
    def X(b):
        return x0 + (b - 1) / 4 * (x1 - x0)
    def Y(v):
        return y1 - (math.log10(v) - lo_e) / (hi_e - lo_e) * (y1 - y0)
    s.rect(x0, y0, x1 - x0, y1 - y0, CARD, BORDER, rx=4)
    for e in range(lo_e, hi_e + 1):
        s.line(x0, Y(10 ** e), x1, Y(10 ** e), BORDER, 1)
        s.text(x0 - 8, Y(10 ** e) + 4, f"1e{e}" if e else "1", 12, MUTED, 400, "end")
    for b in range(1, 6):
        s.text(X(b), y1 + 20, str(b), 12.5, MUTED, 400, "middle")
    s.text((x0 + x1) / 2, y1 + 42, t["f09_x"], 13, INK, 400, "middle")
    s.add(f'<text x="40" y="{(y0 + y1) / 2}" font-family="{FONT}" font-size="13" fill="{INK}" text-anchor="middle" transform="rotate(-90 40 {(y0 + y1) / 2})">{escape(t["f09_y"])}</text>')
    up = [math.sqrt(3) * math.pi / 2 * 4.0 ** -b for b in range(1, 6)]
    lo = [4.0 ** -b for b in range(1, 6)]
    for vals, col, dash in [(up, ORANGE, "6 4"), (lo, NAVY, "2 4")]:
        s.path("M" + " L".join(f"{X(b + 1):.1f},{Y(v):.1f}" for b, v in enumerate(vals)), col, sw=2, dash=dash)
    s.path("M" + " L".join(f"{X(b + 1):.1f},{Y(v):.1f}" for b, v in enumerate(meas)), BLUE, sw=3)
    for b, v in enumerate(meas):
        s.circle(X(b + 1), Y(v), 5, BLUE)
    # legend + table
    lx = 672
    s.line(lx, 118, lx + 30, 118, ORANGE, 2, dash="6 4"); s.lines(lx + 38, 122, t["f09_up"], 12.5, INK, width=36)
    s.line(lx, 170, lx + 30, 170, BLUE, 3); s.text(lx + 38, 174, t["f09_me"], 12.5, INK)
    s.line(lx, 202, lx + 30, 202, NAVY, 2, dash="2 4"); s.lines(lx + 38, 206, t["f09_lo"], 12.5, INK, width=36)
    ty = 262
    s.text(lx, ty, "b", 12.5, MUTED, 600, mono=True)
    for j, hdr in enumerate(["4⁻ᵇ", "meas.", "paper"] if lang == "en" else ["4⁻ᵇ", "medido", "artículo"]):
        s.text(lx + 60 + j * 84, ty, hdr, 12.5, MUTED, 600, "end" if False else "start")
    paper = [0.36, 0.117, 0.03, 0.009, None]
    for b in range(5):
        yy = ty + 22 + b * 20
        s.text(lx, yy, str(b + 1), 12.5, INK, 400, mono=True)
        s.text(lx + 60, yy, num(4.0 ** -(b + 1), lang, 4), 12.5, INK, mono=True)
        s.text(lx + 144, yy, num(meas[b], lang, 4), 12.5, BLUE, 700, mono=True)
        s.text(lx + 228, yy, num(paper[b], lang, 3) if paper[b] else "–", 12.5, INK, mono=True)
    return s, meas


def fig10(t, lang):
    s = Svg(1000, 470, t["f10_t"])
    header(s, t["f10_t"], t["f10_s"])
    toks = t["f10_tokens"]
    x0, w, gap = 40, 108, 12
    s.rect(28, 92, 5 * (w + gap) + 12, 244, CARD, BORDER)
    s.text(40, 116, t["f10_cache"], 13, NAVY, 600)
    for i, tok in enumerate(toks):
        x = x0 + i * (w + gap)
        s.rect(x, 128, w, 34, "#E9E6DD", None, rx=6)
        s.text(x + w / 2, 151, tok, 15, INK, 600, "middle")
        for k, (lbl, col) in enumerate([("K", BLUE), ("V", "#3C8C5A")]):
            yy = 178 + k * 70
            s.rect(x, yy, w, 54, col, None, rx=6, opacity=0.15)
            s.text(x + 10, yy + 22, lbl, 14, col, 700)
            s.text(x + 10, yy + 43, "1011 0010 …", 12, col, 600, mono=True)
    # new token side: q -> attention (reads the cache) -> next word
    qx, qw = 700, 270
    s.rect(qx + 75, 100, 120, 34, PEACHBG, ORANGE, rx=6)
    s.text(qx + 135, 123, t["f10_q"], 13.5, ORANGE, 700, "middle")
    s.arrow(qx + 135, 136, qx + 135, 168, ORANGE, 2)
    s.box(qx, 172, qw, 86, t["f10_att"], None, dark=True, tsize=14.5)
    s.arrow(qx - 4, 215, 652, 215, ORANGE, 2.5)
    s.arrow(qx + 135, 262, qx + 135, 292, ORANGE, 2)
    s.rect(qx + 75, 296, 120, 34, NAVY, None, rx=6)
    s.text(qx + 135, 319, t["f10_next"], 16, PEACH, 700, "middle")
    # write / read band
    s.rect(28, 352, 944, 70, PEACHBG, None, rx=10)
    s.text(44, 380, "→ " + t["f10_w"], 14, ORANGE, 700)
    s.text(44, 406, "← " + t["f10_r"], 14, BLUE, 700)
    s.text(32, 452, t["f10_foot"], 13, MUTED, italic=True)
    return s


def fig11(t, lang):
    s = Svg(1000, 300, t["f11_t"])
    header(s, t["f11_t"], t["f11_s"])
    x0, cw = 32, 7.0
    g = np.random.default_rng(5)
    out = set(g.choice(128, 32, replace=False).tolist())
    for j in range(128):
        o = j in out
        s.rect(x0 + j * cw, 100 if o else 112, cw - 1.5, 52 if o else 40, ORANGE if o else BLUE, rx=1)
    s.rect(x0, 172, 16, 14, ORANGE, rx=2); s.text(x0 + 24, 184, t["f11_o"], 13.5, INK)
    s.rect(x0 + 330, 172, 16, 14, BLUE, rx=2); s.text(x0 + 354, 184, t["f11_r"], 13.5, INK)
    s.text(x0, 224, t["f11_eq"], 15, NAVY, 600, mono=True)
    s.text(x0, 250, t["f11_eq2"], 15, ORANGE, 700, mono=True)
    s.text(x0, 284, t["f11_35"], 13, MUTED, italic=True)
    return s


def fig12(t, lang):
    s = Svg(1000, 470, t["f12_t"])
    header(s, t["f12_t"], t["f12_s"])
    # top row: ingest
    s.box(32, 96, 150, 70, t["f12_docs"])
    s.arrow(186, 130, 214, 130)
    s.box(218, 96, 190, 70, t["f12_emb"])
    s.arrow(412, 130, 440, 130)
    s.box(444, 96, 300, 70, t["f12_idx"], t["f12_idxb"], dark=True)
    # bottom row: query
    s.box(32, 222, 150, 104, t["f12_q"], accent=True)
    s.arrow(186, 262, 214, 262)
    s.box(218, 222, 190, 104, t["f12_emb"])
    s.arrow(412, 262, 440, 262)
    s.box(444, 222, 170, 104, t["f12_scan"], t["f12_scanb"])
    s.arrow(560, 170, 530, 218, NAVY, 2)
    s.arrow(618, 262, 646, 262)
    s.box(650, 222, 196, 104, t["f12_rr"], t["f12_rrb"])
    s.arrow(850, 262, 868, 262)
    s.box(872, 222, 100, 104, t["f12_top"], t["f12_llm"], accent=True)
    # indexing comparison
    s.text(32, 362, t["f12_ing"], 15, NAVY, 700)
    s.rect(32, 374, 940, 40, "#E9E6DD", None, rx=8)
    s.text(48, 399, t["f12_pq"], 14, INK)
    s.rect(32, 420, 940, 40, PEACHBG, None, rx=8)
    s.text(48, 445, t["f12_tq"], 14, ORANGE, 700)
    return s


def fig13(t, lang):
    s = Svg(1000, 330, t["f13_t"])
    header(s, t["f13_t"], t["f13_s"])
    budget = 8 * 2 ** 30
    cfg = [(16, 0), (4, 128), (4, 16), (3.5, 32), (2.5, 32)]
    vals = [int(budget / (2 * 32 * 8 * (b * 128 + e) / 8)) for b, e in cfg]
    mx = max(vals)
    for i, (lbl, v) in enumerate(zip(t["f13_lbl"], vals)):
        y = 96 + i * 44
        tq = i >= 2
        s.text(32, y + 22, lbl, 14, NAVY, 600 if tq else 400)
        s.rect(240, y + 6, v / mx * 600, 26, ORANGE if tq else GREY, rx=4)
        s.text(240 + v / mx * 600 + 10, y + 25, f"{num(v, lang, 0)} {t['f13_unit']}", 14, INK, 700)
    return s, vals


def fig14(t, lang):
    s = Svg(1000, 430, t["f14_t"])
    header(s, t["f14_t"], t["f14_s"])
    r4 = [0.944, 0.912, 0.871, 0.818, 0.814, 0.728]
    r2 = [0.799, 0.715, 0.623, 0.612, None, 0.536]
    for k, (title, rr, x0) in enumerate([(t["f14_b4"], r4, 32), (t["f14_b2"], r2, 520)]):
        s.text(x0, 104, title, 15, NAVY, 700)
        row = 0
        for j, (m, v) in enumerate(zip(t["f14_m"], rr)):
            if v is None:
                continue
            y = 118 + row * 42
            row += 1
            tq = j < 2
            s.text(x0, y + 20, m, 13, INK, 600 if tq else 400)
            s.rect(x0 + 168, y + 5, v * 250, 24, ORANGE if tq else BLUE, rx=3)
            s.text(x0 + 168 + v * 250 + 6, y + 22, num(v, lang, 3), 13, INK, 700)
    s.rect(32, 392, 16, 14, ORANGE, rx=2); s.text(54, 404, t["f14_leg1"], 12.5, INK)
    s.rect(300, 392, 16, 14, BLUE, rx=2); s.text(322, 404, t["f14_leg2"], 12.5, INK)
    return s


def main():
    bias = _bias_data()
    torch.manual_seed(0)
    for lang, t in T.items():
        d = os.path.join(OUT, lang)
        figs = {
            "fig00_overview": fig00(t, lang), "fig01_memory": fig01(t, lang), "fig02_quantization": fig02(t, lang),
            "fig03_papers": fig03(t, lang), "fig04_overhead": fig04(t, lang), "fig05_rotation": fig05(t, lang),
            "fig06_codebook": fig06(t, lang), "fig07_pipeline": fig07(t, lang), "fig08_unbiased": fig08(t, lang, bias),
            "fig10_kvcache": fig10(t, lang), "fig11_outliers": fig11(t, lang), "fig12_vector_search": fig12(t, lang),
            "fig14_recall": fig14(t, lang),
        }
        torch.manual_seed(0)
        figs["fig09_bounds"], meas = fig09(t, lang)
        figs["fig13_capacity"], vals = fig13(t, lang)
        for name, svg in figs.items():
            svg.save(os.path.join(d, name + ".svg"))
        print(lang, len(figs), "figures | measured MSE", [round(m, 4) for m in meas], "| 8 GB contexts", vals,
              "| bias mse/prod", round(float(bias[0].mean()), 4), round(float(bias[1].mean()), 4))


if __name__ == "__main__":
    main()
