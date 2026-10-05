# Light Anonymizer

**Español** · [English](README.md)

Herramienta de escritorio que anonimiza PDF e imágenes **en el propio computador**, pensada para funcionarios públicos chilenos que publican documentos por transparencia (CoP 33 / SmartGORE). Encuentra datos personales —RUT, correo, teléfono, URL, nombres de una lista, rostros y texto dentro de escaneos y fotos—, los elimina de verdad (no con rectángulos negros dibujados encima), limpia los metadatos y después revisa la salida para detectar fugas. Nada sale del computador.

> **Estado: fase 0 terminada, fases 1 y 2 en curso, fase 3 pendiente.** El repositorio contiene el banco de pruebas (un conjunto de prueba ficticio con verdad de terreno exacta, el evaluador y las líneas base), el motor y una aplicación de escritorio preliminar que lo usa, con un instalador para Windows. La revisión humana de cada documento antes de publicarlo es obligatoria, siempre.

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
scripts/         generate_test_data.py, download_models.py, build_exe.py, build_installer.py, demo_notebook_leak.py
packaging/       especificación de PyInstaller, lanzador y script del instalador (Inno Setup) para Windows, y los textos de licencia que faltan en los wheels
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
uv run python -m anonymizer.app             # aplicación de escritorio (--browser para abrirla en el navegador)

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

## Compilar el ejecutable para Windows

Una versión preliminar para Windows, hecha con PyInstaller en modo carpeta, que se entrega como instalador o como .zip:

```bash
uv run python scripts/download_models.py                       # una vez: modelo de rostros YuNet, con verificación SHA-256
uv run --group build python scripts/build_exe.py --installer   # desde una copia sin cambios pendientes; sin --installer, no genera el instalador
```

La AGPL exige ofrecer el código fuente exacto de lo que se entrega, así que el script rechaza una copia de trabajo con cambios sin confirmar; `--allow-dirty` hace una compilación de prueba, que su `LEEME.txt` (y su instalador) marca como no distribuible. El script revisa los modelos (el SHA-256 de YuNet y los modelos PP-OCR que trae `rapidocr`), ejecuta PyInstaller con `packaging/light_anonymizer.spec`, reúne las licencias de terceros (`scripts/collect_licenses.py`: los archivos de licencia de cada paquete incluido, más los textos de `packaging/licenses/` que faltan en sus wheels; la compilación falla si un paquete incluido no tiene ninguno), comprueba que no se haya incluido nada propio del desarrollo ni ningún documento (banco de pruebas, datos de prueba, pytest, matplotlib, la DLL de FFmpeg de OpenCV, cualquier PDF, imagen o archivo de oficina, un RUT con puntos en el código de la aplicación…) y genera:

- `dist/LightAnonymizer/LightAnonymizer.exe` junto con su carpeta `_internal/` (unos 260 MB): la aplicación, sin ventana de consola. A su lado quedan `LEEME.txt` (cómo abrirla, la licencia, la ausencia de garantía y el commit y la URL de su código fuente), `LICENSE.txt`, `LICENSES.md` y `THIRD_PARTY_LICENSES/` (con `INDEX.txt`: cada componente, su versión, licencia y código fuente). Copia siempre la carpeta completa, no solo el `.exe`.
- `dist/LightAnonymizer-<versión>-windows.zip` (unos 120 MB): la misma carpeta comprimida, para entregarla.
- `dist/LightAnonymizer-build.json`: la versión de la compilación, su commit, si tenía cambios sin confirmar y el SHA-256 del ejecutable, para el instalador.
- Con `--installer`, `dist/LightAnonymizer-<versión>-setup.exe` (unos 90 MB): el instalador (ver más abajo).

Se usa una carpeta y no un archivo único porque así la aplicación abre en uno o dos segundos, en vez de descomprimir cientos de MB en `%TEMP%` cada vez que se abre, y los antivirus la marcan menos (la primera vez que se abre después de instalarla o descomprimirla tarda más, mientras el antivirus la revisa). Necesita el componente WebView2, que ya viene en Windows 10 y 11, y no se conecta a internet (WebView2 corre con sus servicios en segundo plano y su inicio de sesión con la cuenta de Windows desactivados: con ese inicio de sesión activo, se conectaba a Microsoft 365 cada vez que se abría). Descomprímela en una ruta corta y fuera de OneDrive, como `C:\Apps\LightAnonymizer`: con rutas largas Windows puede no cargar las bibliotecas nativas (el instalador evita ese problema).

### El instalador

`build_exe.py --installer`, o `uv run python scripts/build_installer.py` después de compilar, compila `packaging/installer.iss` con [Inno Setup 6](https://jrsoftware.org/isinfo.php) (`winget install --id JRSoftware.InnoSetup -e`; su licencia permite el uso comercial y entregar los instaladores, ver [LICENSES.md](LICENSES.md)). Compilarlo toma de uno a tres minutos (LZMA2, compresión máxima). El instalador:

- instala solo para el usuario actual, sin permisos de administrador, en `%LOCALAPPDATA%\Programs\LightAnonymizer`: una ruta corta y fuera de OneDrive;
- está en español; muestra la AGPL en una página "Licencia" que no pide "Acepto" (la AGPL no tiene que aceptarse para recibir ni usar el programa); agrega "Anonimizador" al menú Inicio, un acceso directo en el escritorio solo si se elige (desmarcado por omisión) y una entrada en Configuración > Aplicaciones > Aplicaciones instaladas, y termina con la casilla "Abrir el Anonimizador";
- instala la carpeta completa, con `LEEME.txt`, `LICENSE.txt`, `LICENSES.md` y `THIRD_PARTY_LICENSES/`, para que los avisos de licencia y la oferta del código fuente viajen con el programa;
- actualiza sobre la versión instalada: un instalador más nuevo reemplaza el programa y antes borra los antiguos `_internal/` y `THIRD_PARTY_LICENSES/`, para que nunca se mezclen bibliotecas de dos versiones;
- nunca toca los datos del usuario en `%LOCALAPPDATA%\Anonimizador` (registro técnico, estimaciones de tiempo, perfiles de WebView2; nunca documentos) al instalar ni al actualizar. Al desinstalar pregunta si borrarlos también, con No como respuesta por omisión; una desinstalación silenciosa siempre los conserva;
- no reemplaza ni borra archivos de una aplicación abierta: cada instancia mantiene el mutex con nombre `LightAnonymizer.Running` (`packaging/launcher.py`), y el instalador y el desinstalador piden cerrarla antes (en modo silencioso, desisten).

Las reglas de la AGPL también valen para el instalador: toma la versión y el commit de la compilación de `dist/LightAnonymizer`, que debe coincidir con `dist/LightAnonymizer-build.json`; el instalador de una compilación de prueba dice "compilación de prueba: no distribuir" (página de bienvenida, título de la ventana, propiedades del archivo, entrada en las aplicaciones instaladas); y `build_installer.py` se niega a compilar si las fuentes propias del instalador (`packaging/installer.iss`, `LICENSE`, el propio script) difieren de las del commit de la compilación, salvo con `--allow-dirty` (instalador de prueba). La documentación confirmada después de compilar no importa.

Instalación silenciosa, para los equipos de informática: `LightAnonymizer-<versión>-setup.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART` (más `/DIR=<carpeta>`, `/NOICONS` para no crear el acceso del menú Inicio, `/LOG=<archivo>`); desinstalación silenciosa: `"%LOCALAPPDATA%\Programs\LightAnonymizer\unins000.exe" /VERYSILENT /SUPPRESSMSGBOXES`.

**Todavía sin firma digital.** El instalador y el ejecutable no están firmados, así que la primera vez Windows SmartScreen muestra "Windows protegió su PC" con "Editor desconocido": hay que hacer clic en "Más información" y luego en "Ejecutar de todas formas", como explica `LEEME.txt`. El Control inteligente de aplicaciones o las políticas de la institución pueden bloquear del todo los programas sin firma. Para firmarlos se necesita un certificado de firma de código emitido a nombre de la institución (por una autoridad certificadora o un servicio de firma); con él se firmaría `LightAnonymizer.exe` durante la compilación, y el instalador y el desinstalador mediante la directiva `SignTool` de Inno Setup.

El ejecutable acepta las mismas opciones que `python -m anonymizer.app` (`--browser`, `--no-open`, `--engine`), salvo que siempre usa el motor definitivo: rechaza `--engine fake`, ignora `ANONYMIZER_ENGINE` y, si faltan componentes del motor (un antivirus puede poner uno en cuarentena), muestra un error en vez de pasar al motor de prueba, que exportaría páginas escaneadas y fotos sin censurar. Los errores al abrir aparecen en un mensaje en español, y cerrar la ventana con archivos en proceso o revisados sin exportar pide confirmación. Como no tiene consola, `--url-file RUTA` escribe la URL y el token de sesión en un archivo JSON para las pruebas automáticas; no se escribe nada si no se usa la opción, y el archivo se borra al cerrar la aplicación. Con `--browser`, borrar ese archivo cierra la aplicación de forma ordenada (también se borra su carpeta de trabajo), ya que el ejecutable no recibe Ctrl+C.

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
