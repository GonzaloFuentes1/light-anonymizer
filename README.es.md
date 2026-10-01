# Light Anonymizer

**Español** · [English](README.md)

Herramienta de escritorio que anonimiza PDF e imágenes **en el propio computador**, pensada para funcionarios públicos chilenos que publican documentos por transparencia (CoP 33 / SmartGORE). Encuentra datos personales —RUT, correo, teléfono, URL, nombres de una lista, rostros y texto dentro de escaneos y fotos—, los elimina de verdad (no con rectángulos negros dibujados encima), limpia los metadatos y después revisa la salida para detectar fugas. Nada sale del computador.

> **Estado: fase 0 terminada; fases 1 a 3 pendientes.** El repositorio contiene el banco de pruebas (un conjunto de prueba ficticio con verdad de terreno exacta, el evaluador y las líneas base) y un **prototipo** del motor. Todavía no hay una aplicación para usuarios finales. La revisión humana de cada documento antes de publicarlo es obligatoria, siempre.

## Principios

1. **Recall sobre precisión.** Ante la duda, se censura. Las validaciones (dígito verificador del RUT, confianza del OCR, puntaje del detector de rostros) solo ordenan la revisión; nunca descartan un hallazgo.
2. **Censura real.** El contenido bajo una zona censurada se elimina del archivo (texto, píxeles de imágenes, trazos vectoriales) y el archivo se reescribe completo, sin versiones anteriores.
3. **Limpieza de metadatos.** Se eliminan EXIF (incluidos GPS y miniaturas), XMP, metadatos del PDF, anotaciones, adjuntos, capas ocultas, formularios, marcadores y JavaScript.
4. **Verificación automática de fugas.** La salida se vuelve a revisar con los mismos detectores y con comprobaciones de bytes y de estructura; todo lo que aparezca se informa como fuga.
5. **Sin conexión.** Sin llamadas de red al usarla, con los modelos incluidos y sin telemetría.
6. **Software libre.** El proyecto es AGPL-3.0; cada componente incluido tiene una licencia compatible con la AGPL y nada es de uso no comercial (ver [LICENSES.md](LICENSES.md)).
7. **La revisión humana es obligatoria.** La aplicación nunca dice "documento limpio"; dice "revisión lista para confirmar".

## Estructura del repositorio

```
anonymizer/      motor (engine/real.py: detección, aplicación y verificación de fugas), API local y aplicación preliminar
test_bench/      herramientas de desarrollo: generadores del conjunto de prueba, evaluador, líneas base, prototipo
  generators/    un módulo por familia de documentos (PDF con texto, escaneos, imágenes giradas, EXIF, cédula, pantallazos, rostros…)
  evaluation/    comprobaciones de recall y fugas (texto, bytes, píxeles, imágenes tapadas, imágenes huérfanas, trazos, metadatos)
  baselines/     identidad, oráculo, cuaderno y prototipo (ejecuta el motor real)
scripts/         generate_test_data.py, download_models.py, demo_notebook_leak.py
tests/           pruebas con pytest
docs/            definición de la métrica
test_data/       conjunto de prueba generado y cachés (no se versiona)
models/          modelos ONNX (no se versionan; se descargan con un script)
results/         salidas de las corridas locales (no se versiona)
```

## Para empezar (desarrollo)

Requiere [uv](https://docs.astral.sh/uv/) y Python 3.12.

```bash
uv sync --all-groups                        # dependencias
uv run python scripts/download_models.py    # modelo de rostros YuNet, con verificación SHA-256
uv run python scripts/generate_test_data.py # conjunto de prueba ficticio -> test_data/generated

# correr un sistema sobre el conjunto de prueba y evaluarlo
uv run python -m test_bench.baseline prototype --manifest test_data/generated/manifest.json --output results/details/prototype --processes 3
uv run python -m test_bench.evaluate --manifest test_data/generated/manifest.json \
    --redaction-report results/details/prototype/report.json \
    --outputs-dir results/details/prototype/files --report-dir results/details/prototype

uv run python -m test_bench.compare         # láminas de antes y después -> results/examples
uv run python -m test_bench.process_folder <carpeta con tus documentos> --output results/gore
uv run pytest -q                            # pruebas
```

En un equipo con 8 GB de RAM, usa como máximo 3 procesos en paralelo (cada proceso de OCR ocupa unos 600 MB).

## El banco de pruebas

- **102 archivos ficticios** en 10 familias, con **971 datos personales** de ubicación exacta conocida, **44 metadatos sensibles escondidos** y 7 archivos que deben rechazarse (con contraseña, corruptos, vacíos o con formato falso).
- Los rostros vienen de fuentes con licencia libre documentada (Face Research Lab London Set, Open Images, retratos de dominio público de EE. UU.); se descargan, se verifican por SHA-256 y nunca se distribuyen.
- El evaluador mide el **recall** (¿lo encontró?) y las **fugas** (¿se puede recuperar todavía del archivo de salida?); ver [docs/metrics.md](docs/metrics.md). Se valida a sí mismo: la línea base *identidad* debe filtrar todo y el *oráculo*, nada.
- Criterio de aceptación de la fase 1: cero fugas de RUT, correo y teléfono en el nivel base, y cero fugas de metadatos.

## Privacidad

Nunca subas documentos reales ni nada derivado de ellos (nombres, montos, frases). Los documentos reales viven solo en `test_data/real/`, que git ignora, igual que todos los resultados. Las pruebas y los ejemplos usan solo datos inventados.

## Documentación

- [PLAN.md](PLAN.md): plan detallado, hallazgos y decisiones pendientes para las fases 1 a 3.
- [docs/metrics.md](docs/metrics.md): cómo se miden el recall y las fugas.
- [LICENSES.md](LICENSES.md): licencia de cada dependencia, modelo y fuente de datos de prueba.
- El manual para usuarios finales (en español) se escribirá en la fase 3.

## Licencia

GNU Affero General Public License v3.0 o posterior (ver [LICENSE](LICENSE)). Quien reciba la aplicación tiene derecho a su código fuente, que es este repositorio. Los componentes de terceros mantienen sus propias licencias (ver [LICENSES.md](LICENSES.md)).
