# Métrica de evaluación del anonimizador

Este documento define cómo se mide si el anonimizador funciona. La métrica es la misma en
todas las fases y la calcula `banco_pruebas.evaluar` sobre el conjunto de prueba ficticio
(`datos_prueba/generado/`, generado por `scripts/generar_datos_prueba.py`).

La idea central viene del cuaderno: **no basta con que el documento se vea censurado; hay que
comprobar en el archivo de salida que el dato ya no está.** Por eso se miden dos cosas
distintas:

- **Recall** (¿lo encontró?): qué fracción de los datos personales quedó cubierta por las zonas
  de censura que el sistema propuso.
- **Fugas** (¿se puede recuperar?): qué datos siguen presentes en el archivo de salida, por
  cualquier vía: texto extraíble, bytes del archivo, píxeles visibles, imágenes originales que
  quedaron dentro del archivo, imágenes tapadas por
  un rectángulo, trazos vectoriales o metadatos.

Un sistema puede tener recall alto y aun así fugas (por ejemplo, si dibuja un rectángulo negro
en vez de eliminar el contenido). El criterio de aceptación exige ambas cosas.

## 1. Verdad de terreno

`datos_prueba/generado/manifiesto.json` describe cada archivo y cada dato que contiene
(esquema en `banco_pruebas/esquema.py`).

**Elemento**: un dato en un lugar concreto del archivo.

| Campo | Significado |
|---|---|
| `tipo` | `rut`, `correo`, `telefono`, `url`, `nombre`, `direccion`, `rostro`, `firma`, `qr`, o `texto` (texto neutro de referencia, sin dato personal) |
| `nivel` | `base` (realista y legible), `estres` (difícil: diminuto, espejado, formatos antiguos, mucho ruido), `fuera_de_alcance` (límite documentado: nombre que no está en la lista, firma, manuscrito) |
| `capa` | `texto` (capa de texto PDF), `raster` (píxeles), `vector` (glifos convertidos en trazos), `oculto` (presente pero invisible: capa opcional apagada, fuera de la página, blanco sobre blanco, bajo una imagen) |
| `poligono` | contorno exacto del dato (cuadrilátero rotado si corresponde) |
| `nucleo` | solo rostros: ojos, nariz y boca |
| `valor` | el texto exacto, que es único en todo el conjunto (funciona como *canario*) |

**Metadato sensible**: un dato en metadatos o estructuras no visibles (EXIF, GPS, miniatura
EXIF, XMP, bloques de texto PNG, etiquetas TIFF, metadatos PDF, anotaciones, adjuntos, capas
opcionales, formularios, marcadores, JavaScript, revisiones anteriores de un PDF).

**Archivo con error esperado**: protegido con contraseña, corrupto, vacío o con formato
engañoso. El sistema debe rechazarlo con un código de error comprensible.

Coordenadas: en imágenes, píxeles de la geometría ya corregida por EXIF (la de la salida); en
PDF, puntos en el espacio de página sin rotar de PyMuPDF.

## 2. Lo que entrega el sistema evaluado

1. Una carpeta de salida con los archivos censurados (con la misma ruta relativa).
2. Un informe JSON (`InformeCensura` en `banco_pruebas/esquema.py`) con, para cada archivo:
   ruta de salida, zonas censuradas (página, polígono, tipo, detector, score, estado),
   error si se rechazó, tiempo de proceso y páginas procesadas.

El motor de la fase 1 producirá este informe como parte de su informe de auditoría.

## 3. Comprobaciones por elemento

| Código | Comprobación | Dónde se aplica |
|---|---|---|
| **C** Cobertura | Fracción del polígono cubierta por la unión de las zonas censuradas activas (estado distinto de `descartado`) en esa página. Se calcula rasterizando con sobremuestreo. Umbral: ≥ 0,95. Rostros: núcleo ≥ 0,95 y cara completa ≥ 0,80. | todas las capas visibles |
| **T** Texto extraíble | Se extrae todo el texto de la salida con dos motores independientes (PyMuPDF, sin recortar a la página, y pdfium) y se normaliza. Hay fuga si aparece el valor completo **o un fragmento crítico**: cuerpo del RUT sin dígito verificador, últimos 7 dígitos del teléfono, parte local del correo con `@`, apellidos del nombre, calle y número de la dirección. | PDF |
| **B** Bytes | Se buscan el valor y sus fragmentos críticos en los bytes del archivo, en todos los flujos descomprimidos y en las cadenas de todos los objetos PDF, en UTF-8, Latin-1, UTF-16 BE y UTF-16 LE. | todos |
| **P** Píxeles | Se dibuja la salida (PDF a 144 ppp, en su orientación visible) y se exige que ≥ 95 % de los píxeles del polígono tengan el color de relleno uniforme de la zona (90 % en polígonos de menos de 20 px²). Además, si la zona parece uniforme pero su correlación en gris con la entrada es ≥ 0,90, los píxeles siguen siendo los originales y la comprobación falla: así no pasan por "censurados" un rostro oscuro en sombra, la foto fantasma de la cédula o texto bajo un reflejo. | capas visibles |
| **I** Imágenes tapadas | En PDF, para cada imagen de la página que se superpone al polígono, se extrae la imagen, se lleva el polígono a sus coordenadas y se exige la misma uniformidad. Detecta el error de dibujar un rectángulo encima de una imagen sin borrar sus píxeles. | raster en PDF, rostros en PDF |
| **O** Imágenes originales | En PDF, ninguna imagen de la entrada que contenga un dato puede aparecer intacta (con los mismos píxeles) en la salida, esté o no dibujada en alguna página. Detecta la imagen original sin censurar que queda como objeto huérfano cuando el PDF se guarda sin reescribirlo por completo, un problema que se comprobó en el cuaderno. | raster en PDF, rostros en PDF |
| **V** Trazos vectoriales | En PDF, no deben quedar trazos con curvas (glifos) dentro del polígono. Detecta un rectángulo dibujado sobre texto vectorizado. | vector |

Un elemento está **censurado** si pasan todas las comprobaciones que le corresponden según su
capa (los rostros siguen la regla de su capa `raster`):

| Capa | Comprobaciones |
|---|---|
| `texto` | C, T, B, P |
| `oculto` | T, B |
| `vector` | C, P, V |
| `raster` en imagen | C, P |
| `raster` en PDF | C, P, I, O |

Un elemento que no está censurado es una **fuga**. El informe indica qué comprobación falló.

**Geometría.** La salida debe tener el mismo número de páginas y el mismo tamaño *visible* que
la entrada (una página con `/Rotate 90` reconstruida como página apaisada sin `/Rotate` es
válida). Si no, los elementos de esas páginas cuentan como fuga con el motivo
`geometria_distinta`, porque no se puede verificar que no filtren. La comprobación O necesita
la carpeta de entrada del conjunto (la del manifiesto) para comparar las imágenes originales.

**Metadato sensible**: hay fuga si el canario aparece en la salida (lectores estructurados de
EXIF/XMP/PNG/TIFF/PDF más la búsqueda en bytes), o, para datos no textuales, si la estructura
sigue presente (bloque GPS, miniatura EXIF, adjunto, capa opcional, revisión anterior).

Además se listan como **advertencias** los metadatos residuales que no son canarios (por
ejemplo, un campo `Producer` o un perfil de color), sin contarlos como fuga.

## 4. Métricas

**Recall** por tipo y nivel:

```
recall(tipo, nivel) = elementos con cobertura suficiente / elementos del tipo y nivel
```

Para la capa `oculto` (que no tiene zona visible), "detectado" equivale a "eliminado" (T y B).

**Fugas** por tipo y nivel: número y porcentaje de elementos no censurados, con el detalle de
cada uno (archivo, página, valor, comprobación que falló).

**Fugas de metadatos**: número de metadatos sensibles presentes en la salida.

**Errores esperados**: archivos que debían rechazarse y se rechazaron con el código correcto.
Los archivos procesables que el sistema rechazó se listan aparte; cuentan como no detectados
en el recall, no como fugas (no hay archivo de salida que publicar).

**Desgloses** (informe de recall de rostros y OCR):
- Rostros: por tamaño de la cara en la salida (alto < 24, 24–48, 48–96, 96–192, ≥ 192 px),
  pose (frente, tres cuartos, perfil), ángulo y origen (foto suelta, PDF, escaneo, cédula).
- Texto en imágenes: por ángulo (0, 90, 180, 270, 15, 45, espejado), alto de letra en
  píxeles, fuente, degradación (ruido, desenfoque, JPEG, perspectiva) y categoría de archivo.
- RUT: con dígito verificador válido o inválido (debe dar igual: el dígito no decide nada).

**Métricas informativas** (se reportan, nunca se optimizan a costa del recall):
- *Sobrecensura*: fracción de elementos `texto` neutros (incluidos montos y fechas, que son
  señuelos) cubiertos en ≥ 50 %; zonas censuradas que no tocan ningún dato personal.
- *Conservación del texto* (PDF): fracción de elementos `texto` neutros de la capa de texto
  que siguen siendo extraíbles. Mide si el documento sigue siendo útil y accesible.
- *Tiempo*: segundos por archivo, por página de PDF y por imagen (mediana, p95, máximo), y
  memoria máxima si el informe la trae. Se reporta el equipo en que se midió.

**Modo "censurar todo el texto de esta imagen"**: en los archivos donde el sistema lo activó,
los elementos `texto` pasan a ser objetivos y se reporta su recall aparte.

## 5. Criterio de aceptación de la fase 1

1. **Cero fugas** de `rut`, `correo` y `telefono` en el nivel `base`, en todas las
   categorías (PDF con texto, escaneos, imágenes giradas, cédula, pantallazos, TIFF).
2. **Cero fugas de metadatos** sensibles.
3. Todos los archivos con error esperado rechazados con un mensaje comprensible.
4. Informe de recall de rostros y de OCR con los desgloses de la sección 4 (sin umbral fijo
   en esta fase; los números se discuten con la revisión).
5. Tiempos por imagen y por página en CPU.

Los niveles `estres` y `fuera_de_alcance` se reportan siempre, pero no bloquean. Su función
es dejar los límites a la vista.

## 6. Validación del propio evaluador

Una métrica que no detecta fugas es peor que no tener métrica. Por eso el evaluador se valida
con tres líneas base (`banco_pruebas/lineas_base/`):

| Línea base | Qué hace | Resultado exigido |
|---|---|---|
| `identidad` | copia los archivos sin tocarlos | **todos** los elementos y metadatos deben aparecer como fuga. Si alguno no, el evaluador es ciego para ese caso y se corrige. |
| `oraculo` | censura exactamente la verdad de terreno (rasteriza y rellena) y elimina metadatos | recall 100 % y cero fugas. Si no, la verdad de terreno o las coordenadas están mal. |
| `cuaderno` | la lógica del cuaderno CoP 33 tal cual | referencia: muestra qué cubre hoy y qué no |

Estas tres corridas son pruebas automáticas (`tests/banco_pruebas/`).
