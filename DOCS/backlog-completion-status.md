# Estado de finalización del backlog

Fecha de inicio de la implementación: 2026-07-15.

Estado actual: 11 de 11 bloques terminados. La composición final supera los
gates globales de release ejecutables sin red ni dispositivo físico.

Este documento es el tracker operativo de la continuación descrita en
`backlog-continuation-plan.md`. Distingue implementación, verificación local y
validación física para que una nota de alcance no se contabilice como código
terminado.

## Criterio de cierre

Un bloque se considera terminado únicamente cuando:

1. la API y el comportamiento están implementados sin romper consumidores 3.x;
2. los caminos deterministas tienen tests unitarios;
3. los hot paths de generación tienen tests de integración cuando requieren un
   modelo real;
4. la documentación describe el comportamiento entregado y sus límites reales;
5. pasan formato, build, suite completa y el diagnóstico de API pública.

## Bloques de implementación

| Bloque | Entregable | Estado | Evidencia requerida |
|---|---|---|---|
| 1 | Estimación de bytes de KV cache y composición con presupuesto wired | Terminado | `17aa803`; aritmética, overflow, cuantización y admisión validados |
| 2 | Regresiones de long-context y cachés rotatorias/cuantizadas | Terminado | `17aa803`; trim, máscara, wraparound y combinación de parámetros validados |
| 3 | Presupuesto/ticket de memoria en `ChatSession` | Terminado | `dac4b57`; propagación normal/speculative, reemplazo por turno y cancelación validados |
| 4 | Persistencia segura y prefill sin generación visible | Terminado | `dac4b57`; prefill, metadata, versión y round-trip validados |
| 5 | Speculative decoding adaptativo | Terminado | `ae40937`; política opt-in, fallback sticky, telemetría y equivalencia validados |
| 6 | MTP de alto nivel en `ChatSession` | Terminado | `0daf9b4`; config pública, gates de memoria, multiturno, fallback y benchmark JSON validados |
| 7 | Metadata tipada de thinking | Terminado | `53ade69`; Codable, precedencia, registros e introspección conservadora validados |
| 8 | Conformance ejecutable para LLM/VLM/long-context | Terminado | `e4296d2`; reporte estructurado, estados reales y automatización documentada |
| 9 | Detokenizer y materialización de media | Terminado | `cdb466a`; microbenchmark determinista, fallback 3.x, ABI y 15 tests validados |
| 10 | Prototipo WHT para KV cache | Terminado | `ed7616c`; equivalencia numérica, fallback, persistencia y límites documentados |
| 11 | Gaps VLM verificables | Terminado | `e4296d2`; cap de frames, tests de processor y gates de checkpoints concretos |

## Validación física Bonsai

El usuario confirmó el 2026-07-15 que la validación física descrita en
`bonsai-1bit-compatibility.md` está completada. Esta confirmación cierra el punto
6 del plan de continuación. Las métricas concretas de dispositivo, memoria y
rendimiento solo se incorporarán si se aportan como datos reproducibles; no se
infieren en este documento.

## Historial de commits

| Commit | Bloques | Verificación aislada | Límites pendientes |
|---|---|---|---|
| `17aa803` | 1–2: memoria KV, política wired y long-context | 20 tests en macOS, 0 fallos (`KVCacheMemoryTests`, `LongContextKVCacheTests`, `WiredMemoryPolicyTests`) | El lint integral de `KVCache.swift` conserva avisos preexistentes fuera del diff |
| `ae40937` | 5: speculative decoding adaptativo | 8 tests lógicos/9 ejecuciones en macOS, 0 fallos | La política es opt-in y sus umbrales deben proceder de benchmarks reales |
| `53ade69` | 7: metadata tipada de thinking | 11 tests lógicos/13 ejecuciones en macOS, 0 fallos | Describe capacidad; no modifica automáticamente la plantilla ni el stream |
| `e4296d2` | 8 y 11: dashboard de conformance y guardas VLM | 5 tests Python, 7 tests de media, build macOS e `iphoneos` | Checkpoints remotos e iPhone físico permanecen `not_run` hasta ejecutarlos realmente |
| `ed7616c` | 10: prototipo WHT de KV cache | 7 tests en macOS, 0 fallos | Experimental; dequantiza y aplica WHT inversa por paso, sin promesa de rendimiento |
| `dac4b57` | 3–4: ticket, prefill y persistencia en `ChatSession` | 31 `ChatSessionTests` en macOS, 0 fallos | `LMOutput.State` se conserva en sesión pero no forma parte del archivo persistido |
| `cdb466a` | 9: detokenización acotada y menor materialización transitoria de media | 15 tests en macOS, 0 fallos; macro consumidor y ABI validados | El resultado multimedia final sigue materializado completo; el ahorro RSS exacto requiere Instruments |
| `0daf9b4` | 6: MTP de producción en `ChatSession` | 61 tests lógicos/72 ejecuciones enfocadas en macOS, 0 fallos | MTP cuantizado cae a target-only; los números reales requieren checkpoint y dispositivo físico |

Las pruebas se ejecutaron con `xcodebuild` en clones locales limpios para excluir
los ficheros duplicados no versionados del árbol principal.

## Gates globales de la composición final

Base comparada: `main` en `ac178f6`. Cabeza validada: `0daf9b4`.

- `git diff --check main...HEAD`: limpio.
- `swift format lint --strict` sobre todos los Swift modificados: limpio.
- Build completo `mlx-swift-lm-Package` para macOS: aprobado.
- Suite completa: 454 tests lógicos/489 ejecuciones en macOS, 0 fallos y
  0 skips, comprobados mediante el `.xcresult` estructurado.
- `diagnose-api-breaking-changes main --targets MLXLMCommon`: sin breaking
  changes.
- Dashboard offline: 5 tests Python aprobados y script shell sintácticamente
  válido.
- `IntegrationTesting`: `build-for-testing` aprobado para macOS y para
  `generic/platform=iOS` sin firma.

Los builds conservan avisos preexistentes de Metal, Qwen2VL/Qwen2.5VL,
Gemma4 y un test ParoQuant; ninguno procede de los bloques nuevos. No se
ejecutaron checkpoints remotos ni tests de rendimiento en iPhone como parte de
este gate. La validación física Bonsai se registra por separado porque fue
confirmada por el usuario.
