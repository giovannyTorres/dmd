# modaldyn

Librer�a de Dynamic Mode Decomposition para CPU con NumPy y, opcionalmente,
CPU/CUDA con PyTorch. Las columnas representan snapshots:
**`(variables, instantes)`**. Versi�n **0.1.2**.
Incluye correcciones de precisi�n mixta y convergencia robusta; consulta
[CHANGELOG.md](CHANGELOG.md) y [la gu�a de publicaci�n](docs/releasing.md).

```bash
python -m pip install modaldyn
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

Para GPU instala una versi�n de PyTorch compatible con tu driver y ejecuta
`DMD(rank=2, backend="torch", device="cuda").fit(X)`. Los resultados permanecen
como tensores del backend seleccionado. CUDA no disponible produce un error;
no se cambia silenciosamente a CPU. El soporte CUDA implementado necesita
validaci�n sobre hardware: pruebas omitidas por falta de GPU no lo verifican.

Incluye DMD exacto, TLS, forward/backward experimental, randomized, Hankel con
retardos, DMD con control, RPCA+DMD y streaming mediante estad�sticos suficientes.
No incluye todas las variantes de la literatura. Streaming conserva memoria
O(variables�); Hankel y RPCA pueden aumentar mucho el costo.

PyDMD ya tiene un cat�logo amplio y flowTorch ya ejecuta DMD en CPU/GPU. El valor
propuesto es combinar instalaci�n CPU ligera, selecci�n expl�cita del backend,
diagn�sticos num�ricos y mediciones reproducibles. No se afirma que sea la
primera librer�a GPU, la m�s r�pida ni la m�s precisa. Consulta la
[comparaci�n con fuentes](docs/comparison.md), la [API](docs/api.md), las
[limitaciones](docs/limitations.md) y la [metodolog�a de benchmark](docs/benchmarking.md).

`forecast(n)` inicia despu�s del �ltimo snapshot de entrenamiento;
`reconstruct()` incluye el instante cero. En DMDc se necesitan controles futuros
expl�citos. En streaming se proporciona el estado inicial y no se unen las
fronteras de los lotes autom�ticamente.

El c�digo y los artefactos de distribuci�n pueden publicarse en GitHub. PyPI
necesita una cuenta y un Trusted Publisher configurados: preparar el workflow
no equivale a completar esa publicaci�n. [Gu�a de publicaci�n](docs/releasing.md).

Autor: Giovanny Torres. Licencia MIT.
