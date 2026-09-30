"""Evaluador del anonimizador: compara la salida de un sistema con la verdad de terreno.

La métrica está definida en ``docs/metricas.md``. Punto de entrada programático::

    from banco_pruebas.evaluacion import evaluar
    resultado = evaluar(manifiesto, informe, carpeta_salida)

y por línea de comandos ``uv run python -m banco_pruebas.evaluar``.
"""

from banco_pruebas.evaluacion.nucleo import EvalArchivo, EvalElemento, EvalMetadato, evaluar, evaluar_archivo

__all__ = ["EvalArchivo", "EvalElemento", "EvalMetadato", "evaluar", "evaluar_archivo"]
