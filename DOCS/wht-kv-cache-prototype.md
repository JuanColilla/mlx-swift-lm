# Prototipo experimental de KV cache con Walsh-Hadamard

Fecha: 2026-07-15.

## Estado y activación

Este bloque implementa un prototipo **experimental, opt-in y reversible**. No
cambia el KV cache por defecto ni los esquemas `affine4`/`affine8`.

Se activa únicamente con:

```swift
var parameters = GenerateParameters()
parameters.kvScheme = "wht4" // o "wht8"
```

La conversión dinámica respeta `quantizedKVStart`. Si el cache no está poblado,
el umbral aún no se ha cruzado o las dimensiones no son compatibles, el cache
permanece como `KVCacheSimple`.

En este prototipo el selector se propaga por la generación autoregresiva normal.
Las rutas speculative/MTP y la medición aislada de prefill aún sólo reenvían
`kvBits`; por tanto no activan `wht4`/`wht8`. Se mantiene esta limitación explícita
en vez de ampliar otros hot paths dentro de un experimento de KV cache.

## Invariante matemático

El camino cuantizado existente calcula la atención directamente con
`quantizedMM`. No se puede almacenar `H(K)`/`H(V)` en ese camino sin transformar
también `Q` y aplicar la inversa a la salida:

```text
(Q H) (K H)ᵀ = Q Kᵀ
(A V H) Hᵀ = A V
```

El prototipo evita introducir esos hooks en todos los modelos. Almacena
`affine(H(K))` y `affine(H(V))`, pero antes de la atención reconstruye:

```text
K' = H(dequantize(affine(H(K))))
V' = H(dequantize(affine(H(V))))
```

`MLX.hadamardTransform` usa por defecto escala `1 / sqrt(n)`, de modo que la
matriz es ortonormal y `H⁻¹ = H`. La atención recibe así la base original; la
única desviación intencionada es el error de cuantización afín.

Por este motivo `WalshHadamardQuantizedKVCache` no conforma
`QuantizedKVCacheProtocol`: hacerlo lo enviaría al camino `quantizedMM` sin
compensar la base y produciría resultados matemáticamente incorrectos.

## Dimensiones y fallback

El kernel nativo de MLX admite dimensiones `m · 2^k` para
`m ∈ {1, 12, 20, 28}`, pero la API pública sólo expone la transformación hacia
delante. Para los casos compuestos no se puede asumir que repetir esa operación
sea la inversa (el test real con dimensión 80 confirma que no lo es).

Por corrección, el prototipo usa el subconjunto reversible:

```text
n = 2^k
```

El factor `2^k` se limita a 8192 para `float32` y 16384 para
`float16`/`bfloat16`. Además, keys y values deben admitir un group size afín de
32, 64 o 128. La selección usa el group size compatible más cercano a 64, igual
que el cache afín existente.

Una incompatibilidad durante la conversión devuelve `KVCacheError`; la política
dinámica la trata como fallback recuperable y conserva el cache original. No se
intenta completar dimensiones con padding porque alteraría el contrato del head
y ocultaría costes adicionales.

## Persistencia

`savePromptCache` escribe el identificador inequívoco
`WalshHadamardQuantizedKVCache` y el formato `wht-affine-v1`. `loadPromptCache`
valida ambos antes de reconstruir el estado. Esto evita que un estado WHT se
interprete silenciosamente como un `KVCacheSimple` o `QuantizedKVCache` afín.

## Cobertura

`WalshHadamardKVCacheTests` verifica:

- que `H(H(x)) ≈ x` en head dimensions 64 y 128;
- que 80/96 se rechazan aunque el kernel forward las admita;
- el contrato de dimensiones y dtypes del kernel;
- equivalencia de atención dentro de tolerancia affine8;
- reconstrucción de la secuencia completa tras varias actualizaciones;
- que WHT sólo se activa mediante `wht4`/`wht8`;
- que el comportamiento por defecto y `affine4` no cambian;
- fallback y error recuperable para dimensiones incompatibles;
- round-trip de persistencia y continuación determinista.

## Limitaciones y criterio para promocionarlo

Este diseño reduce el estado persistente del KV cache, pero en cada paso
dequantiza y aplica WHT inversa a todo el contexto. Por tanto, no promete una
mejora de latencia, ancho de banda ni memoria pico; puede ser más lento que
`affine4`. `estimateKVCacheBytes` tampoco interpreta `kvScheme`: sin `kvBits`
paralelo conserva una estimación full-precision, conservadora para admisión.

Antes de promoverlo fuera de `EXPERIMENTAL` hacen falta:

1. benchmarks físicos de TTFT, decode TPS, memoria pico y temperatura;
2. evaluación de calidad long-context en modelos con head dimensions distintos;
3. comparación del error WHT+affine frente a affine puro con datos KV reales;
4. si la calidad lo justifica, un protocolo de atención transformada que aplique
   `H(Q)` y `H⁻¹(output)` sin dequantizar todo el cache.

Hasta entonces `wht4`/`wht8` son superficies de investigación y no una
recomendación de producción.
