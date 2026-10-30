"""Project paths. Runtime artifacts live under ``data/`` (ignored except ``data/summary``)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CORPUS_DIR = DATA / "corpus"
# One DROWSE_HOME per fitted lens: Drowse always writes an R-lens to ``local/relp``, so
# separate homes keep each fit (and Drowse's default home) apart.
LENS_HOMES = DATA / "lenses"
SUMMARY = DATA / "summary"
FITS = DATA / "fits"  # 02_fit.py's lens fits, the authoritative format
LOGS = ROOT / "logs"
# Local copy of the Neuron Persona Library's Qwen data (neuronpersonas.com/data/qwen/: library, combos, combos3)
LIBRARY = DATA / "external" / "qwen_neuron_library" / "library.json"
