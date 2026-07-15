# Nota de investigación: `kvScheme` más allá de affine4/8

Item de backlog #12 (`DOCS/tech-debt-and-research-backlog.md`): "Affine4/8 ya son base.
Explorar WHT, product quantization o compresión híbrida si MLX Swift lo permite
eficientemente."

Esta nota es investigación, no implementación. No hay código de producción aquí,
siguiendo el mismo estándar de honestidad que `RotatingKVCache.quantized(...)`
(`Libraries/MLXLMCommon/KVCache.swift:798-808`), que lanza
`KVCacheError("RotatingKVCache quantization not yet implemented ...")` en vez de
fingir soporte.

## 1. Qué expone hoy mlx-swift

La pasada de release fija el checkout en
`JuanColilla/mlx-swift@5e27a4cb2604599c72615cf058e09801c123b831`;
ver `Package.swift` y `DOCS/bonsai-1bit-compatibility.md`.

### 1.1 Primitivas de cuantización afín

`Source/MLX/Ops.swift` expone tres funciones públicas usadas ya en este repo:

- `quantized(_:groupSize:bits:mode:globalScale:stream:)` — `Ops.swift:2374`
- `dequantized(_:scales:biases:groupSize:bits:mode:globalScale:dtype:stream:)` — `Ops.swift:1146`
- `quantizedMM(_:_:scales:biases:transpose:groupSize:bits:mode:stream:)` — `Ops.swift:2433`

Todas aceptan un `QuantizationMode` (`Ops.swift:1097`) con casos `.affine`, `.mxfp4`,
`.mxfp8`, `.nvfp4`. Solo `.affine` se usa hoy en el KV cache.

Uso real en este repo:

- `QuantizedKVCache.initQuant`/`updateQuantized` llama a `quantized(...)` directamente
  para cuantizar keys/values entrantes (`Libraries/MLXLMCommon/KVCache.swift:884`,
  `:978-979`).
- El scoring/salida cuantizados usan `quantizedMM(...)` en el camino de atención
  cuantizada (`Libraries/MLXLMCommon/KVCache.swift:1961`, `:2001`).
- `resolveAffineScheme(_:)` (`KVCache.swift:2021-2027`) solo resuelve `"affine4"` →
  `(4, 64)` y `"affine8"` → `(8, 64)`; cualquier otro string cae a `nil` y
  `maybeQuantizeKVCache` no hace nada con él (`KVCache.swift:2052-2060`), dejando la
  puerta abierta a que un cache custom interprete el string por su cuenta.

### 1.2 Transformada de Hadamard: SÍ existe, y está expuesta a nivel Swift

Hallazgo relevante que corrige una suposición implícita del ticket: mlx-swift **sí**
tiene un kernel Hadamard/Walsh-Hadamard nativo, fusionado en Metal, con binding
público en Swift:

```swift
// Source/MLX/Ops.swift:1574-1590
/// Perform the Walsh-Hadamard transform along the final axis.
///
/// Supports sizes `n = m*2^k` for `m` in `(1, 12, 20, 28)` and `2^k <= 8192`
/// for `DType.float32` and `2^k <= 16384` for `DType.float16` and `DType.bfloat16`.
public func hadamardTransform(
    _ array: MLXArray, scale: Float? = nil, stream: StreamOrDevice = .default
) -> MLXArray
```

Cadena de implementación: `Ops.swift:1583` → `mlx_hadamard_transform` (C API,
`Source/Cmlx/include-framework/mlx-c-ops.h:499`) → `hadamard_transform` (C++,
`Source/Cmlx/include-framework/mlx-ops.h:150-154`) → kernel Metal fusionado en
`Source/Cmlx/mlx-generated/metal/hadamard.h` (funciones `hadamard_n`/`hadamard_m`,
soporta tamaños compuestos `N = m·2^k`).

`hadamardTransform` **no se usa hoy en ningún fichero Swift de este repo**
(`grep -rn "hadamardTransform" --include="*.swift" .` fuera de `.build` no devuelve
resultados). Es un op disponible y sin usar, no algo que haya que construir desde
cero.

### 1.3 Precedente de "rotar antes de cuantizar" ya en este repo

`Libraries/MLXLMCommon/ParoQuant/RotateQuantizedLinear.swift` implementa exactamente
el patrón "rotación + cuantización afín" — pero para **pesos de capas lineales**, no
para el KV cache. Aplica rotaciones de Givens por pares (aprendidas, no WHT) mediante
un kernel Metal custom (`MLXFast.metalKernel`, `RotateQuantizedLinear.swift:94-110`)
antes de `quantizedMM` (`RotateQuantizedLinear.swift:253-257`). Es la prueba de que
(a) este codebase ya sabe escribir y cachear kernels Metal custom vía
`MLXFast.metalKernel`, y (b) el patrón "rotar activaciones → cuantización afín
estándar" ya tiene un camino de carga de checkpoint dedicado
(`ParoQuantLoader.swift`). No es reutilizable directamente para KV cache: opera sobre
pesos/activaciones de proyección con parámetros de rotación aprendidos y persistidos
en el checkpoint, no sobre tensores de keys/values generados en tiempo de inferencia.

### 1.4 Product quantization / codebooks

No hay primitiva de PQ ni de codebook/vector-quantization en `Source/MLX/`. Sí existen
los bloques de construcción necesarios para montarla a mano: `argmin`/`argmax`,
`take`/`takeAlong`, y álgebra lineal (`qr`, `svd`, `cholesky` en
`Source/MLX/Linalg.swift:667-752`) para un paso de k-means o de reducción de rango.
No hay k-means nativo en mlx-swift.

## 2. Evaluación por técnica

### 2.1 Walsh-Hadamard Transform (WHT) como pre-rotación

**Qué requeriría:** aplicar `hadamardTransform` a keys/values (o a la proyección que
las produce) antes de pasar por `quantized(...)`, y `hadamardTransform` inverso
(la WHT es su propia inversa salvo el factor de escala) tras `dequantized(...)`. La
motivación estándar (QuaRot/SpinQuant/QuIP#-style) es que la rotación reparte outliers
por canal y hace la distribución más uniforme, reduciendo el error de cuantización
afín por grupo — mismo espíritu que ya aplica `RotateQuantizedLinear` con rotaciones
de Givens en vez de WHT.

Restricción real a validar: `hadamardTransform` exige `n = m·2^k` con
`m ∈ {1, 12, 20, 28}` y `2^k` acotado por dtype (`Ops.swift:1576-1577`). El head_dim
típico de KV cache (64, 80, 96, 128...) hay que verificarlo caso por caso contra esa
familia de tamaños — 64 y 128 son `2^k` puros y caen dentro del rango soportado; 80 y
96 no son potencias de 2 pero sí son `m·2^k` (80 = 20·4, 96 no encaja con ningún `m`
de la lista salvo 12·8=96, que sí encaja). Este cálculo debe rehacerse por arquitectura
antes de comprometerse, no asumirse.

**Veredicto: feasible-with-moderate-work.** El kernel fusionado ya existe y está
expuesto en Swift — no hace falta escribir Metal a mano, a diferencia de lo que
sugiere el fraseo del ticket original ("si MLX Swift lo permite eficientemente"). El
trabajo real está en (a) verificar compatibilidad de tamaño por head_dim/arquitectura,
(b) decidir si la rotación se aplica por cabeza o sobre el head_dim completo, (c)
medir si el error de cuantización realmente baja para las distribuciones de KV reales
de los modelos soportados, y (d) diseñar el nuevo caso de `QuantizedKVCacheProtocol`
(ver sección 3). Es un experimento de una–dos semanas, no una investigación abierta.

### 2.2 Product Quantization (PQ)

**Qué requeriría:** partir cada head_dim en subvectores, entrenar (u obtener
offline) codebooks por subespacio vía k-means, y en tiempo de inferencia sustituir
cada subvector de key/value por el índice del centroide más cercano
(`argmin` sobre distancias contra el codebook, disponible) más una tabla de lookup en
la fase de dequantización/matmul. A diferencia de la cuantización afín, PQ no tiene
una forma cerrada por grupo — necesita entrenamiento de codebook (offline, por modelo,
posiblemente por capa) y una tabla de lookup asimétrica para no pagar el coste de
dequantizar todo el cache en cada paso de atención.

No hay soporte de `quantizedMM`-equivalente para codebooks en mlx-swift: la matmul
cuantizada existente asume el formato afín empaquetado en enteros de `bits` bits, no
una tabla de índices con lookup. Reimplementar el análogo de `quantizedMM` para PQ
(asymmetric distance computation contra centroides) sería un kernel Metal nuevo,
distinto del que ya existe para rotación.

**Veredicto: needs-upstream-mlx-work (o kernel Metal custom sustancial).** El coste no
es solo el codebook (eso se puede entrenar offline con NumPy/PyTorch y cargar como
constante), sino la ausencia de un camino de matmul-con-codebook eficiente en MLX
Swift/Metal. Sin eso, cada paso de atención pagaría una dequantización completa del
cache, lo que anula buena parte del ahorro de ancho de banda que PQ promete frente a
affine4/8. No recomendado como siguiente paso — es la técnica de mayor incertidumbre
y mayor coste de las tres.

### 2.3 Compresión híbrida

Interpretado como combinar affine (bits variables por profundidad/posición, ya
soportado vía `quantizedKVStart` y potencialmente `kvBits` variable) con una
transformación previa (WHT) y/o esquemas mixtos por capa (p.ej. capas tempranas sin
cuantizar, capas profundas con affine4 + WHT). No requiere ninguna primitiva nueva más
allá de lo ya cubierto en 2.1 — es composición de piezas existentes
(`quantizedKVStart`, `resolveAffineScheme`, y un futuro esquema WHT) más lógica de
selección de política por capa/posición.

**Veredicto: feasible-with-moderate-work**, condicionado a que WHT (2.1) se implemente
primero — es la técnica que menos primitivas nuevas de MLX necesita, pero la de mayor
superficie de diseño (política de selección por capa, testing de calidad por
combinación). Es efectivamente un paso posterior a 2.1, no una tercera vía
independiente.

## 3. Recomendación: menor siguiente paso viable

`GenerateParameters.kvScheme: String?` (`Evaluate.swift:79`) y
`resolveAffineScheme(_:)` (`KVCache.swift:2021`) ya están diseñados para exactamente
este caso — el propio doc comment de `kvScheme` dice "Extensible for custom schemes
(e.g. WHT-based compression)" (`Evaluate.swift:78`), y `maybeQuantizeKVCache` ya deja
pasar sin efecto cualquier string que `resolveAffineScheme` no reconozca
(`KVCache.swift:2052-2060`), precisamente para que un cache custom lo intercepte.

Forma más pequeña de prototipar WHT sin tocar la API pública:

1. Un nuevo tipo que conforme `QuantizedKVCacheProtocol`
   (`KVCache.swift:145-160`) — análogo a `QuantizedKVCache` pero que aplique
   `hadamardTransform` a keys/values antes de `quantized(...)` en
   `updateQuantized(keys:values:)`, y la inversa tras `dequantized(...)` en el camino
   de lectura.
2. Un resolver de scheme análogo a `resolveAffineScheme`, p.ej.
   `resolveWHTScheme(_ scheme: String?) -> (bits: Int, groupSize: Int)?` reconociendo
   un string nuevo (p.ej. `"wht4"`/`"wht8"`), consultado desde `maybeQuantizeKVCache`
   antes o después de `resolveAffineScheme` (mismo patrón de precedencia que ya existe
   entre `kvScheme` y `kvBits`, `KVCache.swift:2049-2060`).
3. Reutilizar `resolvedKVQuantizationGroupSize` (`KVCache.swift:811-829`) para
   mantener la misma lógica de group size compatible por head_dim, más una
   verificación adicional de que el head_dim cumple la restricción `m·2^k` de
   `hadamardTransform` — si no la cumple, hacer fallback a affine puro (mismo patrón
   defensivo que ya usa `maybeQuantizeKVCache` al descartar caches no cuantizables).
4. No tocar `RotatingKVCache` en esta primera vuelta — su `quantized` ya declara
   explícitamente que no está implementado; un esquema WHT debería nacer solo para
   `KVCacheSimple → QuantizedKVCache`-equivalente, igual que affine4/8 hoy.

No se necesita ningún cambio en mlx-swift/mlx-core para este primer prototipo — todo
lo que hace falta (`hadamardTransform`, `quantized`, `dequantized`, `quantizedMM`) ya
está expuesto. El riesgo principal no es de plataforma sino de resultado empírico: si
la rotación no mejora la calidad medible en las tareas de long-context relevantes, el
esquema queda en `DOCS/tech-debt-and-research-backlog.md` como descartado con
evidencia, no como pendiente indefinido.

PQ (2.2) no debería intentarse hasta que exista un camino de matmul-con-codebook en
mlx-swift, o se acepte pagar el coste de dequantizar el cache completo en cada paso —
lo cual, en la práctica, elimina la ventaja de memoria que se buscaba.
