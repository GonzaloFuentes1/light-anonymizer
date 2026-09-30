# Licencias

Política del proyecto: todo lo que se distribuye con la aplicación (código, bibliotecas y
modelos) debe tener licencia MIT, Apache 2.0, BSD o similar, compatible con uso público y
estatal. Nada con licencia no comercial y nada de InsightFace.

Este archivo distingue tres cosas: lo que **se distribuirá** con la aplicación, lo que solo se
usa **para desarrollar y probar**, y los **datos de prueba**. Las licencias se verificaron el
2026-09-30 contra los archivos de licencia dentro de cada paquete (no solo el campo de PyPI,
que a veces es incorrecto) y cada conclusión crítica se verificó de forma independiente.

Al empaquetar (fase 3), este archivo se completará automáticamente con los textos de licencia
de cada paquete incluido (`dist-info/licenses`, `ThirdPartyNotices.txt` de onnxruntime,
`LICENSE-3RD-PARTY.txt` de OpenCV, `BUILD_LICENSES` de pypdfium2 y la licencia del SDK de
WebView2), y una prueba del empaquetado fallará si aparece un archivo prohibido.

## 1. Componentes previstos para la aplicación

Estado: ✅ compatible · ⚠️ compatible con condiciones · ⛔ incompatible con la política.

| Componente | Versión | Licencia | Estado | Notas |
|---|---|---|---|---|
| **PyMuPDF** (MuPDF) | 1.28.2 | AGPL-3.0 o licencia comercial de Artifex | ⛔ | Distribuir el ejecutable a otros servicios obliga a licenciar toda la aplicación bajo AGPL-3.0 y entregar el código fuente completo. La alternativa es una licencia comercial de Artifex, que es de pago. **Decisión pendiente** (PLAN.md, D1). |
| pypdfium2 (PDFium) | 5.13.0 (PDFium 153.0.7999.0) | BSD-3-Clause / Apache-2.0 | ✅ | Trae incluidas, entre otras, freetype (se usa bajo la opción FTL), ICU, lcms, libjpeg-turbo, openjpeg, libpng, libtiff y zlib, todas permisivas. |
| pypdf | 6.19.0 | BSD-3-Clause | ✅ | Candidato para limpiar la estructura del PDF (metadatos, XMP, anotaciones, adjuntos, JavaScript, capas). |
| opencv-python-headless | 5.0.0.93 | Apache-2.0 (OpenCV), MIT (empaquetado) | ⚠️ Windows / ⛔ macOS | Windows: trae FFmpeg (LGPL-2.1) en un solo DLL de video que se puede eliminar (se verificó que YuNet sigue funcionando), y enlaza Intel IPP ICV bajo la *Intel Simplified Software License* (no OSI). macOS: las ruedas de PyPI enlazan un FFmpeg **GPL-3.0** con x264/x265 que no se puede quitar. Alternativa: compilar OpenCV sin FFmpeg ni IPP, o ejecutar YuNet directamente con onnxruntime. |
| onnxruntime | 1.30.0 | MIT | ✅ | Incluye Eigen (MPL-2.0, solo cabeceras), que se cumple conservando el aviso. Requiere macOS 14 o superior (Apple Silicon). **Trae telemetría de Microsoft** (ETW en Windows; subida HTTPS en macOS desde 1.29), que la app desactiva al iniciar (PLAN.md, 5.6). |
| rapidocr | 3.9.2 | Apache-2.0 | ⚠️ | El paquete sí es compatible, pero exige opencv-python con interfaz gráfica (Qt y FFmpeg), shapely (GEOS, LGPL-2.1), requests y certifi (MPL-2.0) y tqdm (MPL-2.0). Además, incluye código que descarga modelos y abre URLs. Propuesta: usar sus modelos y portar solo la inferencia (PLAN.md, sección 5.4). |
| Modelos PaddleOCR (ONNX) | PP-OCR | Apache-2.0 | ✅ | Detalle por modelo en la sección 2. |
| YuNet (opencv_zoo) | 2023mar / 2026may | MIT | ✅ | Copyright (c) 2020 Shiqi Yu. |
| numpy | 2.5.3 | BSD-3-Clause y otras permisivas | ✅ | OpenBLAS (BSD-3); runtime de GCC con excepción GCC. |
| Pillow | 12.3.0 | MIT-CMU | ✅ | Trae freetype (FTL), harfbuzz, lcms2, libjpeg-turbo, libpng, libwebp, openjpeg, libtiff, zlib-ng, xz, todas permisivas. No lee HEIC. |
| pillow-heif | 1.8.0 | BSD-3 (fuente) / **GPL-2.0 (ruedas, por x265)** | ⛔ | No se usará. |
| pi-heif | 1.4.0 | BSD-3 (fuente) / LGPL-3.0 (libheif, libde265) | ⚠️ | Solo decodifica. Sería aceptable únicamente si se aprueba LGPL, con empaquetado en carpeta (no en un solo archivo). **Decisión pendiente** (D5). |
| rapidfuzz | 3.14.6 | MIT | ✅ | Coincidencia difusa de nombres. |
| pyclipper | 1.4.0 | MIT | ✅ | Expansión de polígonos del detector de texto (reemplaza a shapely). |
| FastAPI | 0.142.2 | MIT | ✅ | Ahora exige `opentelemetry-api` (Apache-2.0), que no envía nada sin el SDK. Se agregará una prueba de que no hay tráfico de red. |
| Starlette / Uvicorn / Pydantic / python-multipart | 1.7.0 / 0.54.0 / 2.13.5 / 0.0.32 | BSD-3 / BSD-3 / MIT / Apache-2.0 | ✅ | |
| pywebview | 6.2.1 | BSD-3-Clause | ✅ | Incluye el SDK de WebView2 de Microsoft (licencia tipo BSD, se reproduce el aviso). En Windows usa pythonnet (MIT), clr-loader (MIT), proxy-tools (MIT) y bottle (MIT). En macOS usa pyobjc (MIT). |
| PyInstaller | 6.22.3 | GPL-2.0 con excepción para el cargador | ✅ | Es herramienta de construcción; la excepción cubre lo que queda dentro del ejecutable. |
| reportlab | 5.0.1 | BSD | ⚠️ | Incluye la fuente DarkGarden (GPL-2.0), que se excluirá del paquete si se usa para el informe PDF. |
| piexif | 1.1.3 | MIT | ✅ | |
| qrcode / segno | 8.2 / 1.6.6 | BSD-3 / BSD-3 | ✅ | Solo si hace falta generar QR. La lectura de QR la hace OpenCV. |
| python-phonenumbers | 9.0.40 | Apache-2.0 | ✅ | Opcional, como apoyo al detector de teléfonos. |

Componentes con copyleft débil que podrían colarse como dependencias transitivas y que se
excluirán o reemplazarán: shapely/GEOS (LGPL-2.1), certifi y tqdm (MPL-2.0), img2pdf
(LGPL-3.0), pyzbar/zbar (LGPL-2.1).

## 2. Modelos

| Modelo | Origen | Licencia | Uso |
|---|---|---|---|
| `face_detection_yunet_2023mar.onnx` (SHA-256 `8f2383e4…2fa4`) | opencv_zoo, `models/face_detection_yunet` | MIT | Detección de rostros (entrada de tamaño fijo). |
| `face_detection_yunet_2026may.onnx` (SHA-256 `ebafce4e…f0f0`, 229 738 bytes) | opencv_zoo | MIT | Detección de rostros (entrada dinámica, recomendado para OpenCV 5). |
| `PP-OCRv6_det_small.onnx` (9 929 594 bytes) y `ch_PP-OCRv5_det_mobile.onnx` (4 819 576 bytes) | PaddleOCR, convertidos a ONNX por RapidAI (incluido en `rapidocr` 3.9.2 / ModelScope) | Apache-2.0 | Detector de texto; se elige uno en la fase 1. |
| `PP-OCRv6_rec_small.onnx` (21 234 383 bytes, diccionario de 18 708 caracteres incrustado) | idem | Apache-2.0 | Reconocedor multilingüe con tildes y ñ/Ñ. |
| `ch_ppocr_mobile_v2.0_cls_mobile.onnx` (585 532 bytes) | idem | Apache-2.0 | Clasificador de orientación 0/180. |

Los modelos no se suben al repositorio: `scripts/descargar_modelos.py` (fase 1) los descarga
para desarrollo y verifica su SHA-256; el empaquetado los incluye dentro del ejecutable.

## 3. Herramientas de desarrollo y pruebas (no se distribuyen)

| Herramienta | Licencia | Uso |
|---|---|---|
| PyMuPDF | AGPL-3.0 | Solo en el banco de pruebas: generar PDF de prueba (capas, anotaciones, revisiones incrementales) y como uno de los dos extractores independientes del evaluador. No se incluye en la aplicación. |
| pypdfium2 | BSD-3 / Apache-2.0 | Segundo extractor independiente del evaluador. |
| matplotlib | Matplotlib License (PSF) | Solo por sus fuentes DejaVu y STIX para dibujar los datos de prueba. |
| piexif, OpenCV, numpy, Pillow | ver arriba | Generación de datos de prueba. |
| pytest, ruff | MIT | Pruebas y estilo. |

## 4. Datos de prueba (no se distribuyen)

Todos los datos personales del conjunto de prueba son inventados. Las imágenes de rostros se
descargan a `datos_prueba/cache/rostros/` (fuera del control de versiones), se verifican por
SHA-256 y **no se incluyen en la aplicación ni en el instalador**. El catálogo completo, con
la URL, el SHA-256 y la atribución de cada imagen, está en `banco_pruebas/fuentes_rostros.json`.

| Fuente | Imágenes | Licencia | Atribución |
|---|---|---|---|
| Face Research Lab London Set | 16 (8 de frente, 5 de tres cuartos, 3 de perfil) | CC BY 4.0 | DeBruine, L. M. & Jones, B. C. (2017). *Face Research Lab London Set*. figshare. doi:10.6084/m9.figshare.5047666.v5. Las personas firmaron consentimiento para uso "en estudios de laboratorio y web, en su forma original o alterada, y para ilustrar investigación". Son los únicos rostros que se pegan en documentos ficticios (cédula, fichas). |
| Open Images V7 (validación) | 5 fotos grupales (70 rostros anotados a mano) | Imágenes CC BY 2.0 (autores en Flickr, ver catálogo); cajas CC BY 4.0 (Google LLC) | Cada imagen lleva autor, título y URL original en el catálogo. Se usan tal cual, sin pegarlas en documentos. |
| Retratos oficiales de EE. UU. (Congreso, Senado) y NASA | 4 (incluye una foto grupal de 43 personas) | Dominio público (obra del gobierno de EE. UU.) | United States Congress / United States Senate / NASA, vía Wikimedia Commons. El dominio público cubre el derecho de autor, no los derechos de imagen: se usan solo como fotos de prueba internas, nunca en documentos ficticios. |
| Fuentes DejaVu y STIX (en matplotlib) | — | Bitstream Vera / dominio público (DejaVu); SIL OFL 1.1 (STIX) | Tipografías con que se dibujan los documentos ficticios. |

Se descartaron por su licencia: DigiFace-1M y FaceSynthetics (Microsoft, solo investigación no
comercial), FFHQ (CC BY-NC-SA) y cualquier dato derivado de InsightFace. SFHQ (rostros
sintéticos, etiquetado CC0) queda fuera por defecto, porque fue generado con StyleGAN2, cuyo
código tiene licencia no comercial.
