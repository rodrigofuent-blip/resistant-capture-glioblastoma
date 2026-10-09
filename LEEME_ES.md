# Reproducibilidad computacional

Repositorio asociado al artículo *Finite-Time Resistant Capture and Resistance-Aware Control in a Reduced Phenotypic Glioblastoma Model*. Contiene implementaciones Python, datos de entrada, resultados numéricos, figuras de referencia y procedimientos verificables. El artículo y su información suplementaria se entregan a la revista por separado.

`data/` almacena los datos de referencia; `src/`, las implementaciones; `scripts/`, los programas ejecutables; `results/`, los resultados y gráficos; `docs/`, los índices de correspondencia y verificaciones de integridad. Los archivos generados en nuevas ejecuciones se escriben en `work/`, directorio excluido de Git.

Instalación: entorno Python compatible con `requirements.txt`.

```bash
python -m pip install -r requirements.txt
python scripts/run_reproducibility.py preflight
python scripts/run_reproducibility.py smoke
```

Las ejecuciones completas están disponibles mediante `grid-full`, `prcc-full` y `control-full`. La documentación principal y el mapa de correspondencia entre manuscritos y resultados figuran en `README.md` y `docs/MANUSCRIPT_CROSS_REFERENCE.md`. Los cálculos emplean tiempo y presión terapéutica adimensionales; no constituyen calendarios clínicos calibrados.
