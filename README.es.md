# Light Anonymizer

**Español** · [English](README.md)

Herramienta de escritorio que anonimiza PDF e imágenes **en el propio computador**, pensada para funcionarios públicos chilenos que publican documentos por transparencia. Encuentra datos personales —RUT, correo, teléfono, URL, nombres de una lista, rostros, firmas manuscritas y dibujadas, y texto dentro de escaneos y fotos—, los elimina de verdad (no con rectángulos negros dibujados encima), limpia los metadatos y después revisa la salida para detectar fugas. Nada sale del computador.

La revisión humana de cada documento antes de publicarlo es obligatoria: la aplicación propone y quien revisa decide.

**Plataforma:** Windows 10 y 11 (64 bits).

## Cómo funciona

1. **Elegir archivos**: PDF (con texto o escaneados), JPG, PNG, WEBP y TIFF, o carpetas completas.
2. **Procesar**: la aplicación revisa cada archivo; puedes empezar a revisar los listos mientras el resto termina.
3. **Revisar**: todas las páginas aparecen en un solo scroll, con el original y sus zonas marcadas a la izquierda y, a la derecha, el resultado tal como se exportará. Los hallazgos dudosos van primero; puedes quitar una censura (con un motivo), agregar una dibujando una zona, o censurar lo que no se censura por defecto.
4. **Exportar**: los archivos anonimizados se guardan en la carpeta que elijas, nunca sobre los originales, junto con un informe de auditoría (PDF y JSON) de lo censurado y lo que cambiaste.

## Principios

1. **Recall antes que precisión.** Ante la duda, se censura. Las validaciones (dígito verificador del RUT, confianza del OCR, puntaje de rostros) solo ordenan la revisión; nunca descartan un hallazgo.
2. **Censura real.** El contenido bajo una zona censurada se borra del archivo (texto, píxeles de imágenes, trazos vectoriales); el archivo se reescribe completo, sin versiones anteriores.
3. **Limpieza de metadatos.** Se eliminan EXIF (incluidos GPS y miniaturas), XMP, metadatos del PDF, anotaciones, adjuntos, capas ocultas, formularios, marcadores y JavaScript.
4. **Verificación automática de fugas.** La salida se revisa de nuevo; un archivo con datos todavía legibles no se exporta.
5. **Sin conexión.** No hay llamadas de red al usarla, los modelos vienen incluidos y no hay telemetría.
6. **Software libre.** AGPL-3.0; cada componente incluido tiene una licencia compatible (ver [LICENSES.md](LICENSES.md)).
7. **La revisión humana es obligatoria.** La aplicación nunca dice «documento limpio»; dice «revisión lista para confirmar».

**Algunas páginas de un PDF pueden exportarse como imagen.** Cuando la aplicación no puede asegurar que no quedó nada dibujado bajo un rectángulo negro (una letra dibujada como trazo que el rectángulo corta, un trazo que cruza su borde, una trama debajo), esa página se exporta como una sola imagen de la página censurada, sin capa de texto ni contenido vectorial. La revisión ya la muestra así, y el informe de auditoría dice qué páginas y por qué.

**Los RUT dudosos se muestran, pero no se censuran por defecto.** Un número escrito solo con dígitos (sin puntos ni guion), cuyo dígito verificador no coincide y sin la palabra «RUT» antes, suele ser un folio o un código: aparece entre lo que no se censura por defecto, para que quien revisa decida. Los RUT con puntos o guion y dígito verificador incorrecto se censuran y se marcan como dudosos.

## Instalación

Ejecuta `LightAnonymizer-<versión>-setup.exe`. Se instala para el usuario actual, sin permisos de administrador, y agrega «Anonimizador» y su manual al menú Inicio. El instalador todavía no tiene firma digital, así que Windows puede mostrar «Windows protegió su PC»: haz clic en «Más información» y luego en «Ejecutar de todas formas». Se desinstala desde Configuración > Aplicaciones > Aplicaciones instaladas.

El manual de usuario: [docs/user-manual/manual-de-usuario.pdf](docs/user-manual/manual-de-usuario.pdf).

## Desarrollo

Requiere [uv](https://docs.astral.sh/uv/) y Python 3.12.

```bash
uv sync --all-groups                            # dependencias
uv run python scripts/download_models.py        # modelo de rostros YuNet, con verificación SHA-256
uv run python scripts/generate_test_data.py     # conjunto de prueba ficticio -> test_data/generated
uv run python scripts/generate_practice_docs.py # documentos de práctica ficticios -> test_data/practice
uv run python -m anonymizer.app                 # aplicación de escritorio (--browser para abrirla en el navegador)
uv run pytest -q                                # pruebas

# correr el motor sobre el conjunto de prueba y evaluarlo
uv run python -m test_bench.baseline prototype --manifest test_data/generated/manifest.json --output results/details/prototype --processes 3
uv run python -m test_bench.evaluate --manifest test_data/generated/manifest.json \
    --redaction-report results/details/prototype/report.json \
    --outputs-dir results/details/prototype/files --report-dir results/details/prototype
```

Usa como máximo 3 procesos en paralelo en un equipo con 8 GB de RAM (cada proceso de OCR usa unos 600 MB).

```
anonymizer/      motor, API local y aplicación de escritorio
test_bench/      generadores del conjunto de prueba, evaluador y líneas base
scripts/         datos de prueba, modelos, PDF del manual, ejecutable e instalador
packaging/       spec de PyInstaller, lanzador, script de Inno Setup y textos de licencias
tests/           suite de pytest
docs/            manual de usuario
```

**El banco de pruebas** tiene 102 archivos ficticios en 10 familias, con 971 datos personales de ubicación exacta conocida, 44 metadatos ocultos y 7 archivos que deben rechazarse. El evaluador mide el **recall** (¿lo encontró?) y las **fugas** (¿se puede recuperar todavía del archivo de salida?).

**Privacidad.** Nunca subas documentos reales ni nada derivado de ellos. Los documentos reales viven solo en `test_data/real/`, que git ignora, igual que todos los resultados. Las pruebas y los ejemplos usan solo datos inventados.

## Generar el instalador de Windows

```bash
uv run python scripts/download_models.py                       # una vez
uv run --group build python scripts/build_exe.py --installer   # desde una copia de trabajo sin cambios pendientes
```

Deja en `dist/` la carpeta de la aplicación (`LightAnonymizer/`, con `LEEME.txt`, las licencias y el manual), un zip de ella y `LightAnonymizer-<versión>-setup.exe` (hecho con [Inno Setup 6](https://jrsoftware.org/isinfo.php)). La AGPL exige ofrecer el código fuente exacto de lo que se entrega, así que la compilación rechaza una copia de trabajo con cambios sin commit; `--allow-dirty` hace una compilación de prueba, marcada como no apta para distribuir. También verifica los modelos y que no se incluya nada de desarrollo ni ningún documento.

Instalación silenciosa: `LightAnonymizer-<versión>-setup.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART`.

## Tráfico de red

La aplicación no hace llamadas de red: su servidor local escucha solo en 127.0.0.1, los modelos vienen incluidos, la interfaz no carga nada de afuera y la telemetría de las bibliotecas está desactivada. La ventana es WebView2 de Microsoft, un componente de Windows; la aplicación desactiva sus servicios en segundo plano, el inicio de sesión y los informes de fallos, pero las actualizaciones de WebView2 y las revisiones propias de Windows (SmartScreen, Defender) son del sistema operativo. Un equipo que deba tener tráfico cero necesita una regla de firewall de salida para `LightAnonymizer.exe`.

## Licencia

GNU Affero General Public License v3.0 o posterior (ver [LICENSE](LICENSE)). Quien recibe la aplicación tiene derecho a su código fuente, que es este repositorio. Los componentes de terceros conservan sus propias licencias (ver [LICENSES.md](LICENSES.md)).
