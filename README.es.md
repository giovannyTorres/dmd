# modaldyn

Librería de Dynamic Mode Decomposition para CPU con NumPy y, opcionalmente,
CPU/CUDA con PyTorch. Las columnas representan snapshots:
**`(variables, instantes)`**. Versión inicial de investigación: 0.1.0.

```bash
python -m pip install 'git+https://github.com/giovannyTorres/dmd.git@v0.1.0'
```

```python
import numpy as np
from modaldyn import DMD

t = np.arange(160) * 0.1
X = np.vstack((np.cos(1.3*t), np.sin(1.3*t)))
modelo = DMD(rank=2, dt=0.1, backend="numpy").fit(X)
reconstruccion = modelo.reconstruct()
futuro = modelo.forecast(20)
print(modelo.diagnostics())
```

Para GPU instala una versión de PyTorch compatible con tu driver y ejecuta
`DMD(rank=2, backend="torch", device="cuda").fit(X)`. Los resultados permanecen
como tensores del backend seleccionado. CUDA no disponible produce un error;
no se cambia silenciosamente a CPU. El soporte CUDA implementado necesita
validación sobre hardware: pruebas omitidas por falta de GPU no lo verifican.

Incluye DMD exacto, TLS, forward/backward experimental, randomized, Hankel con
retardos, DMD con control, RPCA+DMD y streaming mediante estadísticos suficientes.
No incluye todas las variantes de la literatura. Streaming conserva memoria
O(variables²); Hankel y RPCA pueden aumentar mucho el costo.

PyDMD ya tiene un catálogo amplio y flowTorch ya ejecuta DMD en CPU/GPU. El valor
propuesto es combinar instalación CPU ligera, selección explícita del backend,
diagnósticos numéricos y mediciones reproducibles. No se afirma que sea la
primera librería GPU, la más rápida ni la más precisa. Consulta la
[comparación con fuentes](docs/comparison.md), la [API](docs/api.md), las
[limitaciones](docs/limitations.md) y la [metodología de benchmark](docs/benchmarking.md).

`forecast(n)` inicia después del último snapshot de entrenamiento;
`reconstruct()` incluye el instante cero. En DMDc se necesitan controles futuros
explícitos. En streaming se proporciona el estado inicial y no se unen las
fronteras de los lotes automáticamente.

El código y los artefactos de distribución pueden publicarse en GitHub. PyPI
necesita una cuenta y un Trusted Publisher configurados: preparar el workflow
no equivale a completar esa publicación. [Guía de publicación](docs/releasing.md).

Autor: Giovanny Torres. Licencia MIT.
