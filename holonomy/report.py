"""Utilidades de los guiones: comprobaciones con salida OK/FALLO y rutas."""
import csv
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"


class Checker:
    """Acumula comprobaciones; `finish()` sale con código 1 si alguna falla."""

    def __init__(self):
        self.failures = 0

    def check(self, ok, msg):
        print(("OK    " if ok else "FALLO ") + msg, flush=True)
        if not ok:
            self.failures += 1
        return ok

    def close(self, a, b, tol, msg):
        err = abs(a - b)
        return self.check(err <= tol, f"{msg}: {a:.12g} vs {b:.12g} (error {err:.1e}, tol {tol:.0e})")

    def finish(self):
        print(f"\n{'Todo correcto' if not self.failures else f'{self.failures} fallos'}")
        sys.exit(1 if self.failures else 0)


def write_csv(name, header, rows):
    DATA.mkdir(exist_ok=True)
    path = DATA / name
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"escrito {path.relative_to(ROOT)}")
    return path


def read_csv(name):
    with open(DATA / name) as fh:
        rows = list(csv.DictReader(fh))
    return [{k: float(v) for k, v in r.items()} for r in rows]
