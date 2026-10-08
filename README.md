# CRO agéntico

Backend que responde preguntas abiertas de negocio cruzando varias fuentes de
datos, mediante un LLM que decide qué consultar. Expone los resultados por API
para que un frontend externo los renderice.

## Sobre el workshop

Este repositorio es el sistema que se construye en **CRO Agéntico: desarrollo con
IA de un sistema de Insights Inteligente para tu negocio**, workshop de
[Conversion Talks](https://conversiontalks.com/workshops/) impartido por **Joaquín
Ruiz**, AI & Innovation Director.

El workshop recorre el camino completo: de una pregunta de negocio en lenguaje
natural a una respuesta que cruza tres fuentes, nombra una causa y deja cada
cifra trazada hasta la herramienta que la devolvió. Aquí está el código final,
con sus tests y su set de evaluación.

**Lo que no hace:** no renderiza HTML, no escribe en ninguna fuente, no calcula
aritmética dentro del LLM, no genera el layout del cuadro de mando, no autentica
usuarios finales y no persiste datos de negocio — solo trazas y caché.

---

## Arranque

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"

uv run python -m fixtures.validate       # el dataset cumple sus invariantes
uv run pytest -q                         # 108 tests, sin red
uv run python -m evals.run               # 31 casos de evaluación
uv run uvicorn api.main:app --factory --reload
```

Por defecto arranca en `CRO_MODE=fixtures`: no toca la red y no necesita ninguna
credencial.

```bash
curl -s 'localhost:8000/insights?persona=ceo&window=last_14d' | jq
curl -s -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"La conversión en móvil ha caído. ¿Por qué?","persona":"analyst"}' | jq
```

---

## Cómo está montado

```
pregunta ──▶ orquestador ──▶ tools (capa semántica) ──▶ conectores ──▶ fuentes
                │                                          │
                │                                     Observation
                ▼                                          │
          validación post-hoc ◀── traza (ToolCall + Observation)
                │
                ▼
          persona ──▶ JSON
```

Cuatro reglas sostienen el diseño:

1. **El LLM no calcula.** Toda cifra nace en Python, dentro de una `Observation`.
   `compare_windows` hace las restas, los porcentajes y el contraste estadístico.
2. **Toda cifra es trazable.** El validador numérico extrae cada número del texto
   del modelo y comprueba que aparece en algún resultado de tool del mismo run.
   Si no aparece, el insight se degrada (`agent/validators.py`).
3. **Una causa exige dos fuentes.** `causal_claim=true` con evidencia de una sola
   fuente se degrada automáticamente a correlación.
4. **El modo no cambia el razonamiento.** `CRO_MODE=fixtures` sustituye solo la
   capa de *transporte* de los conectores. La capa semántica y el orquestador son
   idénticos en fixtures y en real.

### Las ocho tools

| Tool | Para qué | Fuente |
|---|---|---|
| `get_funnel_metrics` | en qué escalón se pierde el usuario | GA4 |
| `get_traffic_breakdown` | si el cambio es de un segmento o transversal | GA4 |
| `get_orders_summary` | pedidos, ingresos y AOV | Shopify |
| `get_product_performance` | unidades, ingresos de línea y attach rate | Shopify |
| `get_conversion_rate` | conversión derivada (pedidos/sesiones) | derivada |
| `compare_windows` | deltas y significación entre dos periodos | la que toque |
| `get_support_signal` | tickets por tema | soporte |
| `get_data_health` | hasta qué día hay dato consolidado | todas |

La descripción de cada tool dice para qué sirve **y para qué no**: es lo único
que tiene el modelo para elegir.

### Ventanas

Las resuelve el conector, en la zona horaria de la fuente, nunca en UTC y nunca
en el LLM (`connectors/windows.py`). **El día de hoy no entra en ninguna ventana:**

| Ventana | Rango | Ventana | Rango |
|---|---|---|---|
| `last_7d` | hoy-7 … hoy-1 | `prev_7d` | hoy-14 … hoy-8 |
| `last_14d` | hoy-14 … hoy-1 | `prev_14d` | hoy-28 … hoy-15 |
| `last_30d` | hoy-30 … hoy-1 | `prev_30d` | hoy-60 … hoy-31 |
| `same_period_last_year` | `last_14d` desplazada 364 días (mismo día de la semana) | | |

---

## Credenciales y scopes

Ninguna credencial vive en el repositorio. Todo entra por variable de entorno;
copia `.env.example` a `.env`. Los scopes son **de solo lectura, sin excepción**:

| Fuente | Autenticación | Scopes exactos |
|---|---|---|
| GA4 | Application Default Credentials (`gcloud auth application-default login`) | `https://www.googleapis.com/auth/analytics.readonly` |
| Shopify | Admin API access token en `X-Shopify-Access-Token` | `read_orders`, `read_products` |
| Soporte (Zendesk) | Bearer token | `read` |
| Vertex AI | ADC del proyecto | `https://www.googleapis.com/auth/cloud-platform` (región UE, retención cero) |

La zona horaria de la propiedad de GA4 se declara en `CRO_GA4_TIMEZONE` (por
defecto `Europe/Madrid`) y es la que se usa para resolver las ventanas.

La PII (email, nombre, teléfono, dirección, `customer_id`) **no sale del
conector**: se descarta en el mapeo a `Observation`, antes de cualquier log
(`connectors/base.py::strip_pii`), y hay un test que recorre la salida
serializada de todos los conectores buscando patrones de email y teléfono.

---

## API

Todas las respuestas llevan la cabecera `X-Run-Id`.

```
GET  /insights?persona={ceo|ecommerce_manager|analyst}&window={window}
POST /ask                 {question, persona, window|null}
GET  /evidence/{insight_id}
GET  /trace/{run_id}
GET  /connectors/health
```

Errores, siempre con el mismo cuerpo `{"error": {"code", "message", "run_id"}}`:

| HTTP | `code` |
|---|---|
| 400 | `invalid_persona` · `invalid_window` |
| 404 | `insight_not_found` · `run_not_found` |
| 422 | `question_too_long` (> 500 caracteres) |
| 424 | `source_unavailable` |
| 429 | `rate_limited` |
| 504 | `run_timeout` |

**Un run degradado no es un error:** devuelve 200 con `degraded: true`. El sistema
nunca inventa una causa para evitar un estado degradado.

Esquemas completos en `/docs` (OpenAPI) con el servidor arrancado.

---

## Personas

Una persona solo modifica la síntesis y la agregación de salida: se aplica sobre
los insights canónicos ya validados, así que el mismo `run_id` renderizado con
las tres devuelve, por construcción, las mismas cifras.

| | `ceo` | `ecommerce_manager` | `analyst` |
|---|---|---|---|
| Máx. insights | 3 | 6 | sin límite |
| `proposed_action` | opcional | obligatorio | opcional |
| `evidence` | resumida | resumida | completa |
| `assumptions` | ocultas | visibles | visibles |
| Confianza | etiqueta | etiqueta | etiqueta + `n` |

---

## Dataset de referencia

`fixtures/` contiene un escenario coherente que sirve a la vez de doble de test y
de entorno de desarrollo sin credenciales: **el umbral de envío gratuito sube de
50 € a 75 € a mitad de `last_14d`**. Los carritos pequeños abandonan en checkout,
la conversión cae, el AOV sube y los ingresos quedan casi planos. Ninguna fuente
aislada revela la causa.

Los datos se indexan por `day_offset` (1 = ayer), no por fecha absoluta: el
dataset es válido cualquier día que se ejecute.

```bash
uv run python -m fixtures.generate    # regenera (determinista)
uv run python -m fixtures.validate    # comprueba las invariantes
```

Invariantes verificadas en cada `pytest`: `revenue/orders == aov` (±0,01 €),
`orders == Σ orders por aov_bucket`, `conversion_rate == orders/sessions`
(±0,0001) y funnel monótono decreciente por ventana **y** por día. Los fixtures
se generan, no se editan a mano.

---

## Evaluación

```bash
uv run python -m evals.run                     # set completo
uv run python -m evals.run --family refusal -v
uv run python -m evals.run --json informe.json
```

31 casos en tres familias: `causal`, `refusal` y `factual`. El runner reporta
% de casos superados, coste medio por run, p50/p95 de latencia y media de
`tool_calls`, y sale con código 1 si falla alguno.

La familia `refusal` es la que no se puede eliminar: un sistema que nunca dice
"no tengo ese dato" está inventando.

---

## Modo real

```bash
export CRO_MODE=live
export CRO_LLM_PROVIDER=vertex
uv pip install -e ".[vertex,ga4]"
```

El SDK del proveedor solo lo importa `agent/providers/vertex.py`, y de forma
perezosa: el sistema arranca, se testea y ejecuta evaluaciones sin esa
dependencia instalada.

**Servidores MCP:** `container.register_mcp_servers()` los registra como
conectores adicionales. Un servidor sin `MCPMapping` declarado —o que exponga una
herramienta que el mapping no cubre— hace fallar el arranque con un mensaje
explícito. El LLM nunca recibe las herramientas del servidor MCP: solo ve las
ocho tools canónicas.

---

## Estructura

```
connectors/     transporte + mapeo a Observation (aquí se descarta la PII)
semantic/       tipos canónicos, fuente de verdad por métrica, las ocho tools
agent/          orquestador, guardarraíles, validadores, personas, proveedores
api/            HTTP, formato único de error, X-Run-Id
storage/        persistencia de trazas (SQLite)
fixtures/       dataset de referencia, generador y validador de invariantes
evals/          set de evaluación y runner
tests/          108 tests, sin red
```

Si tocas algo, el orden en el que se rompe todo es: tipos → fixtures →
conectores → tools → orquestador → validadores → evaluación → personas → API.
