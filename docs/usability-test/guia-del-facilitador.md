# Prueba de usabilidad del Anonimizador: guía del facilitador

Esta guía explica cómo preparar y conducir una prueba de usabilidad corta del Anonimizador con funcionarios que publican documentos por transparencia. Junto a ella están:

- [tareas.md](tareas.md): las tarjetas de tareas para el participante y, para ti, el criterio de éxito de cada una.
- [hoja-de-observacion.md](hoja-de-observacion.md): la hoja donde el observador anota lo que pasa.
- [cuestionario-final.md](cuestionario-final.md): el cuestionario SUS y tres preguntas abiertas.
- [El manual de usuario](../user-manual/manual-de-usuario.pdf), por si el participante lo pide.

## 1. Objetivo

Saber si una persona que publica documentos por transparencia, sin ayuda, puede:

1. Procesar una carpeta de documentos.
2. Encontrar y revisar un hallazgo dudoso.
3. Entender por qué una página se exportará como imagen.
4. Quitar una censura que no corresponde, con su motivo.
5. Cubrir un dato que la aplicación no encontró, dibujando una zona.
6. Exportar y encontrar el informe de auditoría.

Buscamos los puntos donde la gente se confunde, duda o se equivoca. **Evaluamos la aplicación, no a las personas.** Lo más grave que puede pasar es que un dato personal quede visible sin que el participante lo note: anótalo siempre.

## 2. Participantes y duración

- **2 o 3 participantes.** Funcionarios que publican o revisan documentos por transparencia. Idealmente con distinta experiencia en computación. No deben haber usado la aplicación ni participado en su desarrollo.
- **30 a 40 minutos por persona:** 5 de bienvenida, 25 de tareas y 5 a 10 de cuestionario y conversación final.
- **Dos personas del equipo:** un facilitador, que conduce la sesión y habla con el participante, y un observador, que toma notas y el tiempo. Si están solos, el facilitador también anota, pero la sesión toma más tiempo.

## 3. Preparación (el día antes)

### El computador

1. Un computador con Windows 10 u 11, con mouse. Usa una cuenta de Windows de prueba, sin correo ni archivos personales.
2. Instala el Anonimizador con el instalador (sección 2 del manual). Ábrelo una vez para pasar el aviso de Windows SmartScreen antes de la sesión.
3. Activa el modo "No molestar" de Windows y cierra los demás programas (correo, chat, navegador), para que no aparezcan notificaciones con datos reales en la pantalla.

### Los documentos de prueba

Usa **solo documentos ficticios**. Nunca uses documentos reales de la institución, ni siquiera "de ejemplo".

En un computador con el código del proyecto, genera los documentos una sola vez. Se necesita el entorno de desarrollo (README, sección "Para empezar"):

```
uv sync --all-groups
uv run python scripts/generate_test_data.py
uv run python scripts/generate_practice_docs.py
```

Crea en el escritorio del computador de prueba una carpeta llamada **Documentos de prueba** y copia en ella estos 8 archivos:

| Archivo | Ruta en el proyecto | Para qué sirve |
|---|---|---|
| `acta_con_timbre.pdf` | `test_data/practice/acta_con_timbre.pdf` | Tareas 2 y 3: un número de inventario que parece RUT dudoso (no se censura por defecto), un enlace institucional y un timbre sobre un nombre que hace que la página se exporte como imagen. |
| `oficio_fotografiado.jpg` | `test_data/practice/oficio_fotografiado.jpg` | Tarea 5: una firma a mano que la aplicación **no** encuentra. |
| `mixto_texto_y_escaneo.pdf` | `test_data/generated/pdf_scanned/mixto_texto_y_escaneo.pdf` | Tarea 4: "JEFATURA DIVISIÓN DE ADMINISTRACIÓN Y FINANZAS" aparece marcado como nombre de persona. |
| `informe_honorarios_01.pdf` | `test_data/generated/pdf_text/informe_honorarios_01.pdf` | Documento de relleno con muchos hallazgos, nombres dudosos y dos enlaces institucionales. |
| `resolucion_exenta.pdf` | `test_data/generated/pdf_text/resolucion_exenta.pdf` | Documento de relleno de dos páginas. |
| `timbre_y_firma.pdf` | `test_data/generated/pdf_scanned/timbre_y_firma.pdf` | Un escaneo con una firma que la aplicación sí encuentra (marcada como dudosa). |
| `tarjeta_0.png` | `test_data/generated/rotated_images/tarjeta_0.png` | Una imagen con un enlace no personal. |
| `protegido_contrasena.pdf` | `test_data/generated/errors/protegido_contrasena.pdf` | Tarea 1: un archivo con contraseña que la aplicación no puede abrir ("Necesita tu ayuda"). |

Procesar la carpeta completa toma alrededor de un minuto.

### Ensayo

Haz la prueba completa tú mismo una vez, en el mismo computador, y comprueba que:

- `protegido_contrasena.pdf` queda en "Necesita tu ayuda".
- En `acta_con_timbre.pdf`, el grupo "No se censuran por defecto" tiene un RUT dudoso y un enlace, y la columna Después muestra "Esta página se exportará como imagen".
- En `mixto_texto_y_escaneo.pdf`, "JEFATURA DIVISIÓN DE ADMINISTRACIÓN Y FINANZAS" aparece en "Para revisar primero".
- En `oficio_fotografiado.jpg` no hay ninguna zona sobre la firma.

Si algo no coincide (por ejemplo, después de actualizar la aplicación), ajusta las tareas antes de la sesión. La prueba automática `tests/test_practice_docs.py` revisa los dos documentos de práctica.

Después del ensayo, cierra la aplicación y borra la carpeta de exportación.

### Materiales impresos

- Las tarjetas de tareas ([tareas.md](tareas.md)), una por hoja, sin la parte "Para el facilitador".
- Una hoja de observación por participante.
- Un cuestionario por participante.
- El manual de usuario impreso o abierto en otra ventana, a mano si el participante lo pide.
- Un cronómetro.

## 4. Antes de cada participante

1. Cierra el Anonimizador si está abierto. Al cerrarlo se borran los archivos cargados, las listas y la revisión anterior.
2. Borra la carpeta **Exportados** del escritorio, si existe.
3. Comprueba que la carpeta **Documentos de prueba** tiene los 8 archivos.
4. Abre el Anonimizador y déjalo en la pantalla de inicio.
5. Ten lista una hoja de observación con el código del participante (P1, P2, P3).

## 5. Durante la sesión

### Bienvenida (5 minutos)

Puedes decir algo así:

> "Gracias por venir. Estamos probando una aplicación que ayuda a censurar datos personales antes de publicar documentos por transparencia. Queremos ver si es fácil de usar. **Estamos evaluando la aplicación, no a ti**: si algo no te resulta, es un problema de la aplicación y nos sirve mucho saberlo.
>
> Te voy a pedir que hagas algunas tareas. Mientras las haces, **piensa en voz alta**: cuéntame qué estás buscando, qué esperas que pase y qué te sorprende. Si te quedas en silencio, te voy a preguntar qué estás pensando.
>
> Todos los documentos son inventados: ningún dato es de una persona real. No vamos a grabar tu cara ni tu nombre. Puedes parar cuando quieras. ¿Tienes alguna pregunta?"

Pide su consentimiento verbal y anótalo en la hoja (sin su nombre).

### Las tareas (25 minutos)

- Entrega una tarjeta a la vez y pide que la lea en voz alta.
- Pon en marcha el cronómetro cuando termine de leer. Detenlo cuando diga que terminó o cuando se cumpla el tiempo límite.
- Las tareas usan el estado que dejó la tarea anterior: no cierres la aplicación entre tareas.

**Qué decir:**

- "¿Qué estás pensando?"
- "¿Qué esperabas que pasara?"
- "¿Qué harías ahora si estuvieras solo en tu puesto?"
- "Cuéntame más sobre eso."
- Si pregunta algo: "¿Tú qué crees?" o "¿Qué harías si yo no estuviera?".

**Qué no decir:**

- No expliques la pantalla ni digas dónde hacer clic.
- No digas "bien", "correcto" ni "no, así no". Usa "gracias" o "sigamos".
- No uses las palabras de la pantalla antes que el participante (por ejemplo, no digas "dibujar zona" ni "no se censuran por defecto").
- No te rías ni suspires, aunque algo sea gracioso o lento.

**Si se queda atascado:** espera. Si pasan dos minutos sin avance o se cumple el tiempo límite, anota la tarea como no completada y ofrece seguir: "Gracias, con esto ya aprendimos mucho. Pasemos a la siguiente". Si la tarea siguiente depende de ella, haz tú el paso que faltaba sin explicarlo y anótalo como "ayuda dada".

**Si pide el manual:** entrégaselo y anota en qué tarea y qué buscó. También es un dato útil.

**Después de cada tarea:** pregunta "Del 1 al 7, ¿qué tan fácil o difícil fue esta tarea?" (1 = muy difícil, 7 = muy fácil) y anótalo.

### Cuestionario y cierre (5 a 10 minutos)

1. Entrega el [cuestionario final](cuestionario-final.md). Pide que lo responda solo, sin pensar mucho cada respuesta.
2. Conversa sobre las preguntas abiertas y anota frases textuales.
3. Agradece y cierra el Anonimizador.

## 6. Cómo registrar sin registrar datos personales

- **Sin nombres.** Identifica a cada participante con un código (P1, P2, P3). No escribas su nombre, su RUT, su cargo exacto ni su unidad en ninguna hoja. Si necesitas una lista con los nombres para coordinar las sesiones, guárdala aparte y bórrala al terminar.
- **Sin video de la persona.** No grabes su cara ni su voz. Si quieres grabar, graba solo la ventana del Anonimizador y solo con su permiso.
- **Citas sin datos.** Anota frases textuales, pero quita cualquier cosa que identifique al participante o a otra persona (nombres, casos reales que mencione).
- **Si aparece un documento real** (por ejemplo, el participante abre un archivo propio), detén la sesión, cierra la aplicación y no lo anotes.
- **Guarda las hojas** en una carpeta de la institución con acceso restringido. Borra las grabaciones cuando termine el análisis.

## 7. Después de las sesiones: análisis

1. Para cada tarea, cuenta cuántos participantes la completaron, sin ayuda y con ayuda, y el tiempo de cada uno.
2. Junta los problemas observados. Para cada uno, anota cuántos participantes lo tuvieron y su gravedad:

| Gravedad | Significado |
|---|---|
| 1. Cosmético | Molesta, pero no cambia el resultado. |
| 2. Menor | Hace perder tiempo o genera dudas, pero la persona lo resuelve sola. |
| 3. Mayor | La persona no puede terminar la tarea sin ayuda, o termina con un resultado distinto al que quería. |
| 4. Crítico | Un dato personal quedaría publicado, o se censuraría información pública sin que la persona lo note. |

3. Calcula el puntaje SUS de cada participante (instrucciones en el cuestionario). Con 2 o 3 personas, el puntaje sirve solo como referencia; lo más valioso son los problemas observados y las frases textuales.
4. Escribe un resumen de una página: los 3 a 5 problemas más importantes, con su gravedad, y una propuesta para cada uno.
