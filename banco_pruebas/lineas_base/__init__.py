"""Líneas base que validan al evaluador (``docs/metricas.md``, sección 6).

Cada módulo expone ``procesar(archivo, manifiesto, carpeta, detalles) -> ResultadoArchivo``:

- ``archivo``: el registro del manifiesto (``esquema.Archivo``).
- ``manifiesto``: el manifiesto completo (su ``raiz`` es la carpeta de los archivos de entrada).
- ``carpeta``: carpeta ``<salida>/archivos``; el archivo de salida va en ``carpeta / archivo.ruta``.
- ``detalles``: diccionario libre que termina en ``InformeCensura.detalles`` (información de apoyo).

El recorrido de todos los archivos, la medición de tiempo y la escritura del informe están en
``banco_pruebas.linea_base``.
"""

SISTEMAS = ("identidad", "oraculo", "cuaderno", "prototipo")
