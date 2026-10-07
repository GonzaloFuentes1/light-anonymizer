# Prueba de usabilidad del Anonimizador: tareas

Seis tareas, en este orden. Cada una usa lo que dejó la anterior: no cierres la aplicación entre tareas.

Imprime cada tarjeta en una hoja aparte y entrega al participante **solo el recuadro "Tarjeta"**. La parte "Para el facilitador" es para ti y para el observador.

Tiempo total de las tareas: unos 25 minutos.

---

## Tarea 1. Procesar una carpeta

> **Tarjeta**
>
> Te llegó una carpeta con documentos que hay que publicar en el portal de transparencia. Está en el escritorio y se llama "Documentos de prueba".
>
> Pide a la aplicación que busque los datos personales de todos los documentos de esa carpeta. Avísame cuando termine. Si algún documento tiene un problema, cuéntame qué harías.

**Para el facilitador**

- **Punto de partida:** la aplicación abierta en la pantalla de inicio, sin archivos.
- **Tiempo límite:** 5 minutos.
- **Criterio de éxito:** agrega los 8 archivos (con "Elegir una carpeta", arrastrando la carpeta o eligiendo los archivos), hace clic en "Procesar" sin apagar ningún grupo de "Qué buscar", y explica qué pasa con `protegido_contrasena.pdf` (tiene contraseña: hay que pedir una versión sin contraseña, u omitirlo).
- **Éxito parcial:** procesa los archivos pero no nota el archivo con problema, o lo nota pero no entiende el mensaje.
- **Observa:** si encuentra "Elegir una carpeta"; si lee o cambia el panel "Qué buscar" y por qué; si espera a que terminen todos o empieza a revisar antes; cómo interpreta "Necesita tu ayuda" y el mensaje "Ingresa la contraseña" (la aplicación no tiene dónde escribirla).

---

## Tarea 2. Revisar un dato dudoso

> **Tarjeta**
>
> Abre el "Acta de entrega de equipo computacional" (`acta_con_timbre.pdf`).
>
> En el acta hay un número que la aplicación encontró y que podría ser un RUT, pero que no censuró. Encuéntralo, compáralo con el documento original y decide si se debe censurar o no. Explícame por qué.

**Para el facilitador**

- **Punto de partida:** los archivos ya procesados (tarea 1), en cualquier pantalla.
- **Tiempo límite:** 5 minutos.
- **Qué hay en el documento:** en el grupo "No se censuran por defecto" hay un RUT dudoso: el **número de inventario del equipo**, escrito sin puntos ni guion, cuyo dígito verificador no coincide. No es un dato personal.
- **Criterio de éxito:** encuentra el número en la lista o en el documento, lo mira en la columna Antes (la línea dice "N° de inventario del equipo") y decide **dejarlo visible** con una razón que tiene sentido ("es un número de inventario, no un RUT").
- **Éxito parcial:** lo encuentra pero lo censura sin mirar el original, o necesita más de un intento para entender por qué no está censurado.
- **Observa:** si va primero a "Para revisar primero" o a "No se censuran por defecto"; si entiende el texto "RUT dudoso sin puntos ni guion"; si usa J y K; si confunde "Censurar" con "Quitar…"; si nota el enlace del mismo grupo y qué hace con él.

---

## Tarea 3. Una página que se exportará como imagen

> **Tarjeta**
>
> Sigue en el acta de entrega. Mira cómo quedará el documento cuando lo publiques.
>
> ¿Hay algo distinto en esta página? Cuéntame qué crees que va a pasar con ella al guardarla y si eso te parece un problema.

**Para el facilitador**

- **Punto de partida:** `acta_con_timbre.pdf` abierto en Revisar.
- **Tiempo límite:** 3 minutos.
- **Qué hay en el documento:** el timbre redondo cruza el nombre de quien firma. La columna Después muestra el aviso "Esta página se exportará como imagen" con el motivo "un trazo dibujado bajo una zona no se pudo quitar con certeza".
- **Criterio de éxito:** encuentra el aviso y explica con sus palabras que la página se guardará como una foto de la página ya censurada (por ejemplo: "ya no se podrá copiar el texto", "es para que no quede nada debajo del timbre"), y que no tiene que hacer nada más.
- **Éxito parcial:** ve el aviso pero no sabe qué significa, o cree que tiene que corregir algo.
- **Observa:** si mira la columna Después; si el aviso le genera desconfianza o miedo; si piensa que es un error; si busca ayuda en el manual.

---

## Tarea 4. Quitar una censura que no corresponde

> **Tarjeta**
>
> Abre el oficio `mixto_texto_y_escaneo.pdf`.
>
> En el remitente del oficio, la aplicación censuró el nombre de una unidad del gobierno regional como si fuera una persona. El nombre de una unidad no es un dato personal y debe quedar visible. Corrígelo y deja registrado por qué lo hiciste.

**Para el facilitador**

- **Punto de partida:** Revisar, con otro archivo abierto.
- **Tiempo límite:** 4 minutos.
- **Qué hay en el documento:** "JEFATURA DIVISIÓN DE ADMINISTRACIÓN Y FINANZAS" está marcado como Nombre, en "Para revisar primero".
- **Criterio de éxito:** quita esa censura con "Quitar…" (o la tecla Supr), elige el motivo "No es un dato personal" (u "Otro motivo" con un comentario) y confirma con "Quitar censura". En la lista, el hallazgo queda tachado con el botón "Restaurar", y en la columna Después el texto queda visible.
- **Éxito parcial:** quita la censura sin fijarse en el motivo, o quita por error otra censura (anótalo: quitar un nombre real es gravedad 4).
- **Observa:** cómo cambia de archivo; si entiende la ventana "¿Quitar esta censura?"; si escribe un comentario; si comprueba el resultado en la columna Después.

---

## Tarea 5. Cubrir un dato que la aplicación no encontró

> **Tarjeta**
>
> Abre la foto del oficio `oficio_fotografiado.jpg`.
>
> Revisa la foto como si la fueras a publicar hoy. Asegúrate de que no quede ningún dato personal a la vista y corrige lo que haga falta.

**Para el facilitador**

- **Punto de partida:** Revisar, con otro archivo abierto.
- **Tiempo límite:** 5 minutos.
- **Qué hay en el documento:** la firma a mano sobre el nombre de quien firma **no** está marcada. El nombre del destinatario, el correo y el nombre de quien firma sí están censurados.
- **Criterio de éxito:** nota que la firma quedó visible, usa "Dibujar zona" (o la tecla D) y arrastra un rectángulo que cubre toda la firma en la columna Antes. En la columna Después la firma queda en negro.
- **Éxito parcial:** cubre solo una parte de la firma, o necesita varios intentos para dibujar la zona.
- **No completada:** dice que la foto está lista sin cubrir la firma. Es un problema de gravedad 4: anótalo con detalle (qué miró, qué no miró).
- **Observa:** si compara Antes y Después; si encuentra "Dibujar zona" o intenta otra cosa (hacer clic, buscar un botón "agregar"); si entiende que hay que arrastrar en la columna Antes; si sale del modo dibujo.

---

## Tarea 6. Exportar y encontrar el informe de auditoría

> **Tarjeta**
>
> Ya revisaste el acta de entrega, el oficio `mixto_texto_y_escaneo.pdf` y la foto del oficio. Guarda las versiones censuradas de esos tres documentos en una carpeta nueva del escritorio llamada "Exportados".
>
> Después, abre el informe de la exportación y muéstrame dónde dice qué censura quitaste en la tarea 4 y por qué.

**Para el facilitador**

- **Punto de partida:** Revisar.
- **Tiempo límite:** 5 minutos.
- **Criterio de éxito:** confirma los tres archivos ("Confirmar este archivo", y "Confirmar igual" si quedan dudosos sin abrir), va a Exportar, elige o escribe la carpeta `Exportados` del escritorio, exporta (los tres dicen "Sin fugas" y "Exportado"), abre `informe_auditoria_<fecha>_<hora>.pdf` desde esa carpeta y señala la sección "Censuras quitadas por quien revisó" con el motivo "No es un dato personal".
- **Éxito parcial:** exporta pero no encuentra el informe, o encuentra el informe pero no la sección.
- **Observa:** si entiende por qué un archivo dice "Falta confirmar la revisión"; cómo reacciona al aviso de dudosos sin abrir (¿vuelve a revisar o confirma igual?); si cambia la carpeta de destino o deja la de por defecto; cómo encuentra el informe (la ruta en pantalla, el Explorador de archivos); si le preocupa que el informe muestre datos.
