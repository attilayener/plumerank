"""Mechanical claim checking: every component a document names must exist in the code.

Architecture inflation is the most repeated loss in this project's corpus -- a README that
names a component the source does not contain. The fix is not editorial care, because
editorial care is exactly what fails at 03:00 two days before a deadline. The fix is a
script that greps the claims out of the documents, greps the reality out of the source,
and exits non-zero on the difference. It runs inside ``make test``, so a document and the
code can only disagree for as long as one test run.

Six checks
----------
1. **Surface table complete.** Every ``cv2.*`` call in the real source is classified in
   ``SURFACES``. An unclassified call FAILS, because the alternative -- dropping it
   quietly -- is how a published count goes wrong in the flattering direction.
2. **OpenCV symbols.** Every OpenCV symbol a document names must appear at a real CALL
   SITE under ``plumerank/``, ``api/`` or ``scripts/``.
3. **Routes.** Both directions against ``api.main.app.routes``: a documented route the app
   does not serve is a lie, and a route the app serves that no document mentions is an
   unreviewed public surface on an unattended box.
4. **Dependencies.** Every package a document names must be in ``requirements.lock`` (or
   the training lockfile, or the standard library).
5. **Makefile targets.** Every ``make <target>`` a document prints must exist.
6. **Claimed source paths.** Every ``plumerank/`` ``api/`` ``scripts/`` ``tests/``
   ``deploy/`` ``web/`` path a document names must exist on disk. This is the check that
   catches a file described for a branch that never ran.

Why this file parses and does not grep
--------------------------------------
Three traps live in the real source, and a ``grep -c 'cv2.dnn.Net.forward'`` gets all
three wrong:

* ``cv2.dnn.Net.forward`` appears in ``escalate.py`` only inside a docstring; the calls are
  ``self._net.forward(`` and ``self._net.setInput(``. A checker matching the doc spelling
  would go green on a docstring -- it would be claim-checking the claim.
* ``cv2.DISOpticalFlow`` appears as a type annotation, and ``from __future__ import
  annotations`` means the interpreter never evaluates those. Counting one is counting a
  symbol nothing resolves, so annotations are skipped here and the constructor
  ``DISOpticalFlow_create`` is what counts.
* ``statistic.py`` and ``stabilise.py`` name OpenCV symbols in comments in order to say
  they are NOT used. Comments do not survive parsing, which is the cheapest possible way
  to get that right.

An AST gets all three right for free, so the extra thirty lines buy a checker that cannot
be satisfied by prose.

The defended-surface count
--------------------------
This script is the authority on it. It prints what it finds TODAY, from the source on
disk, and a document may quote only the number it printed on the day that document was
written. The build plan predicts 23, falling to 22 if the ablation gate deletes S4 and to
21 if GSOC is descoped too -- a prediction is not a measurement, and this file never
prints one.
"""

from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import importlib.util
import json
import pathlib
import re
import sys
from dataclasses import dataclass, field

import cv2

__all__ = [
    "Surface",
    "SourceIndex",
    "Check",
    "Report",
    "SURFACES",
    "collect_source_symbols",
    "collect_doc_symbols",
    "count_surfaces",
    "run",
    "main",
]

# ---------------------------------------------------------------------------------
# Where the truth on each side lives
# ---------------------------------------------------------------------------------

#: Judge-facing documents. Exactly the four the build contract names. FEEDBACK.md is
#: deliberately absent: it quotes other people's words, and a checker that fails because
#: a reviewer said "you should have used SIFT" would teach the team to stop quoting.
DOC_FILES: tuple[str, ...] = ("README.md", "ARCHITECTURE.md", "REPORT.md", "DEMO.md")

#: Directories that count as real source. ``spike/`` is excluded on purpose -- it is the
#: pre-gate scratch corner, and a surface used only there is not a surface the product has.
SOURCE_DIRS: tuple[str, ...] = ("plumerank", "api", "scripts")

#: A documented path under one of these must exist. ``results/`` and ``data/`` are absent
#: because their contents are produced by ``make bench`` and ``make fetch``; failing on a
#: results file before the benchmark has run would make this script unrunnable on a clean
#: clone, which is the one place it matters most.
CLAIMABLE_PATH_ROOTS: tuple[str, ...] = (
    "plumerank",
    "api",
    "scripts",
    "tests",
    "deploy",
    "web",
)

COUNTED = "counted"
FEATURES = "features"
DRAWING = "drawing"
SUPPORT = "support"

#: Print order for the module table. It is the order the architecture doc uses, so a
#: reader comparing the two is comparing rows and not hunting.
MODULE_ORDER: tuple[str, ...] = ("video", "imgproc", "geometry", "bgsegm", "dnn", "core")


@dataclass(frozen=True)
class Surface:
    """One OpenCV surface the project may use, and how to recognise its call site.

    ``key`` is the canonical *source* spelling with the ``cv2.`` root removed, because
    that is the form both sides normalise to: ``dnn.blobFromImage``, ``Sobel``,
    ``bgsegm.createBackgroundSubtractorGSOC``.
    """

    key: str
    module: str
    group: str
    method: str | None = None
    """Set when the call site is a bound method on an object OpenCV handed us, so the
    source never contains the doc spelling at all."""
    requires: str | None = None
    """A bound method is only evidence of *this* surface when the constructor is present
    IN THE SAME FILE -- ``.apply(`` alone is called on CLAHE, on MOG2 and on LSBP, and
    ``.forward(`` is a method name half this codebase could have. What a file-local rule
    proves is that the object OpenCV handed us was built where the method is called; what
    it does not prove is which of two OpenCV objects built in that file received the call.
    Resolving that needs dataflow the checker deliberately does not have, so the residual
    is stated here rather than papered over."""
    aliases: tuple[str, ...] = ()
    """Doc spellings that mean this surface, normalised the same way (``cv::`` and
    ``::`` already resolved)."""


SURFACES: tuple[Surface, ...] = (
    # -- video ---------------------------------------------------------------------
    Surface("createBackgroundSubtractorMOG2", "video", COUNTED),
    Surface(
        "BackgroundSubtractorMOG2.apply",
        "video",
        COUNTED,
        method="apply",
        requires="createBackgroundSubtractorMOG2",
        aliases=("BackgroundSubtractor.apply",),
    ),
    Surface(
        "DISOpticalFlow_create",
        "video",
        COUNTED,
        aliases=("DISOpticalFlow.create",),
    ),
    Surface("calcOpticalFlowPyrLK", "video", COUNTED),
    # -- imgproc -------------------------------------------------------------------
    Surface("phaseCorrelate", "imgproc", COUNTED),
    Surface("phaseCorrelateIterative", "imgproc", COUNTED),
    Surface("createHanningWindow", "imgproc", COUNTED),
    Surface("connectedComponentsWithStats", "imgproc", COUNTED),
    Surface("Sobel", "imgproc", COUNTED),
    Surface("Laplacian", "imgproc", COUNTED),
    Surface("createCLAHE", "imgproc", COUNTED),
    Surface("morphologyEx", "imgproc", COUNTED),
    # -- geometry ------------------------------------------------------------------
    # estimateAffine2D, NOT estimateAffinePartial2D: the 4-DoF variant rejects
    # USAC_MAGSAC on 5.0.0.93, so the 6-DoF spelling is the one that runs and the only
    # one a document may name.
    Surface("estimateAffine2D", "geometry", COUNTED),
    Surface("USAC_MAGSAC", "geometry", COUNTED),
    # -- bgsegm (contrib) ----------------------------------------------------------
    Surface("bgsegm.createBackgroundSubtractorLSBP", "bgsegm", COUNTED),
    Surface("bgsegm.createBackgroundSubtractorGSOC", "bgsegm", COUNTED),
    # -- dnn -----------------------------------------------------------------------
    Surface("dnn.readNetFromONNX", "dnn", COUNTED),
    Surface("dnn.ENGINE_NEW", "dnn", COUNTED),
    Surface("dnn.blobFromImage", "dnn", COUNTED),
    Surface(
        "dnn.Net.forward",
        "dnn",
        COUNTED,
        method="forward",
        requires="dnn.readNetFromONNX",
    ),
    # -- core ----------------------------------------------------------------------
    Surface("getBuildInformation", "core", COUNTED),
    Surface("TickMeter", "core", COUNTED),
    Surface("setNumThreads", "core", COUNTED),
    # -- used, and deliberately NOT counted as analysis depth ----------------------
    # Saying this out loud is cheaper than being caught inflating the number, and it is
    # the difference between a count a judge can audit and a count a judge can attack.
    Surface("goodFeaturesToTrack", FEATURES, FEATURES),
    Surface("line", "imgproc", DRAWING),
    Surface("putText", "imgproc", DRAWING),
    Surface("rectangle", "imgproc", DRAWING),
    Surface("imwrite", "imgcodecs", DRAWING),
    Surface("resize", "imgproc", DRAWING),
    Surface("cvtColor", "imgproc", DRAWING),
    # -- plumbing: real calls, but not defended as analysis surfaces ---------------
    # Every one of these is genuinely called. None is claimed as analysis depth, and the
    # line between the two groups is drawn once, here, rather than per document.
    Surface("warpAffine", "imgproc", SUPPORT),
    Surface("threshold", "imgproc", SUPPORT),
    Surface("GaussianBlur", "imgproc", SUPPORT),
    # cv2.blur is corpus SELECTION, not analysis: it smooths a gradient-magnitude map
    # while build_confusion_set.py decides which non-fire windows are the hard ones. It
    # never runs on the judged path and is deliberately NOT a defended surface -- the
    # confusion set is chosen once, offline, and then only its pixels matter.
    Surface("blur", "imgproc", SUPPORT),
    Surface("absdiff", "core", SUPPORT),
    Surface("countNonZero", "core", SUPPORT),
    Surface("getTickCount", "core", SUPPORT),
    Surface("getTickFrequency", "core", SUPPORT),
    Surface("setRNGSeed", "core", SUPPORT),
    # The read-back beside setNumThreads. setNumThreads is a counted surface because
    # pinning the pool is load-bearing on 2 vCPUs; getNumThreads only reports what took,
    # so a caller can publish the number rather than assume it.
    Surface("getNumThreads", "core", SUPPORT),
    Surface("imencode", "imgcodecs", SUPPORT),
    Surface("imdecode", "imgcodecs", SUPPORT),
    Surface("getStructuringElement", "imgproc", SUPPORT),
    Surface("compare", "core", SUPPORT),
    Surface("imread", "imgcodecs", SUPPORT),
    Surface("error", "core", SUPPORT),
    Surface("dnn.Net.setInput", "dnn", SUPPORT, method="setInput"),
    Surface("dnn.Net.setPreferableBackend", "dnn", SUPPORT, method="setPreferableBackend"),
    Surface("CLAHE.apply", "imgproc", SUPPORT, method="apply"),
    Surface("DISOpticalFlow.calc", "video", SUPPORT, method="calc"),
    Surface("TickMeter.start", "core", SUPPORT, method="start"),
    Surface("TickMeter.stop", "core", SUPPORT, method="stop"),
    Surface("TickMeter.getTimeMilli", "core", SUPPORT, method="getTimeMilli"),
)

_BY_KEY: dict[str, Surface] = {s.key: s for s in SURFACES}
_BY_ALIAS: dict[str, Surface] = {}
for _s in SURFACES:
    _BY_ALIAS[_s.key] = _s
    for _a in _s.aliases:
        _BY_ALIAS[_a] = _s

#: ``cv2.dnn`` and ``cv2.bgsegm`` on their own are module references, not surfaces.
MODULE_REFERENCES: frozenset[str] = frozenset({"dnn", "bgsegm", "utils", "typing"})

#: A trailing ALL_CAPS component is an enum or flag: ``CV_32F``, ``MORPH_OPEN``,
#: ``CC_STAT_AREA``. Passing one to a call is not a surface of its own, so they are
#: summarised rather than listed -- except the two the table claims by name
#: (``USAC_MAGSAC``, ``dnn.ENGINE_NEW``), which are looked up first.
_FLAG_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")

#: Type names that appear in documents only because a signature was quoted. Requiring a
#: call site for ``cv::Mat`` would ban quoting a signature, which is the opposite of what
#: this script is for.
IGNORED_DOC_SYMBOLS: frozenset[str] = frozenset(
    {
        "Mat", "UMat", "InputArray", "OutputArray", "InputOutputArray", "Ptr", "Size",
        "Point", "Point2d", "Point2f", "Rect", "Scalar", "Vec", "Vec2f", "noArray",
        "String", "Range", "TermCriteria", "Algorithm", "Exception", "cv", "cv2",
    }
)

#: Bare words that are cv2 attributes AND ordinary English. A document must write the
#: ``cv2.`` prefix to claim one of these -- otherwise the sentence "we take the mean of
#: the ring" is a claim about ``cv2.mean``.
AMBIGUOUS_BARE_WORDS: frozenset[str] = frozenset(
    {
        "mean", "sum", "min", "max", "add", "divide", "multiply", "subtract", "split",
        "merge", "sort", "reduce", "transform", "compare", "flip", "rotate", "norm",
        "trace", "solve", "invert", "pow", "sqrt", "exp", "log", "abs", "repeat",
        "determinant", "magnitude", "phase", "polarToCart", "line", "circle", "rectangle",
        "resize", "error", "version", "have",
    }
)

# ---------------------------------------------------------------------------------
# Dependency vocabulary
# ---------------------------------------------------------------------------------

#: Import name -> distribution name, for the handful where they differ. Without this a
#: document that writes ``import cv2`` claims a package no lockfile has ever contained.
IMPORT_ALIASES: dict[str, str] = {
    "cv2": "opencv-contrib-python-headless",
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "yaml": "pyyaml",
    "dateutil": "python-dateutil",
}

#: Packages a document might name that would each be a new dependency. The set is finite
#: on purpose: the failure mode this guards is a REPORT.md sentence saying "we fit the
#: curve with scipy", not an exhaustive audit of PyPI.
WATCHED_PACKAGES: frozenset[str] = frozenset(
    {
        "scipy", "pandas", "matplotlib", "seaborn", "plotly", "bokeh", "altair",
        "torch", "torchvision", "onnx", "onnxruntime", "tensorflow", "keras", "jax",
        "scikit-learn", "sklearn", "scikit-image", "skimage", "pillow", "PIL",
        "requests", "aiohttp", "flask", "django", "starlette", "redis", "sqlalchemy",
        "celery", "tqdm", "ultralytics", "transformers", "langchain", "openai",
        "anthropic", "supervision", "shapely", "numba", "polars", "duckdb",
        "streamlit", "gradio", "joblib", "einops", "timm", "albumentations",
    }
)

# Framework-supplied routes. They exist on every FastAPI app and nobody documents them.
FRAMEWORK_ROUTES: frozenset[str] = frozenset(
    {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
)


# ---------------------------------------------------------------------------------
# Source side -- what the code actually calls
# ---------------------------------------------------------------------------------


@dataclass
class SourceIndex:
    """Every OpenCV call site in the real source, with where each one was found."""

    attributes: dict[str, list[str]] = field(default_factory=dict)
    """``cv2``-rooted dotted names, root stripped: ``Sobel`` -> ["plumerank/statistic.py:285"]."""
    methods: dict[str, list[str]] = field(default_factory=dict)
    """Bound-method call names, e.g. ``forward`` -> ["plumerank/escalate.py:257"]."""

    def sites(self, surface: Surface) -> list[str]:
        """Where ``surface`` is called, or an empty list when it is not called at all."""
        if surface.method is not None:
            hits = list(self.methods.get(surface.method, ()))
            if surface.requires is not None:
                # File-local, not repo-global. A `net.forward()` in one module and a
                # `cv2.dnn.readNetFromONNX` in another are not the same net, and crediting
                # a defended surface to that pair is exactly the inflation this file is
                # supposed to make impossible.
                built_in = _files(self.attributes.get(surface.requires, []))
                hits = [h for h in hits if _file_of(h) in built_in]
            return hits
        return list(self.attributes.get(surface.key, ()))

    def has(self, surface: Surface) -> bool:
        return bool(self.sites(surface))


def _file_of(site: str) -> str:
    """``plumerank/segment.py:216`` -> ``plumerank/segment.py``."""
    return site.rsplit(":", 1)[0]


def _files(sites: list[str]) -> set[str]:
    return {_file_of(s) for s in sites}


class _CallSiteVisitor(ast.NodeVisitor):
    """Collects call sites, and only call sites.

    Comments never reach the AST, docstrings are ``Constant`` nodes with no ``Attribute``
    inside them, and annotations are skipped explicitly below -- so the three ways this
    repo names an OpenCV symbol without calling it are all invisible here by construction
    rather than by a list of exceptions somebody has to maintain.
    """

    def __init__(self, index: SourceIndex, where: str) -> None:
        self._index = index
        self._where = where

    def _record(self, table: dict[str, list[str]], key: str, line: int) -> None:
        table.setdefault(key, []).append(f"{self._where}:{line}")

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # The annotation is a string at runtime under `from __future__ import
        # annotations`; nothing resolves it, so nothing in it is a call site.
        self.visit(node.target)
        if node.value is not None:
            self.visit(node.value)

    def visit_arg(self, node: ast.arg) -> None:
        return

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in node.args.defaults + [d for d in node.args.kw_defaults if d]:
            self.visit(default)
        for stmt in node.body:
            self.visit(stmt)

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def visit_Attribute(self, node: ast.Attribute) -> None:
        dotted = _dotted_name(node)
        if dotted is not None and dotted.split(".", 1)[0] == "cv2":
            tail = dotted.split(".", 1)[1] if "." in dotted else ""
            if tail:
                self._record(self._index.attributes, tail, node.lineno)
            return  # the inner chain is this same symbol; recording it twice is noise
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute):
            dotted = _dotted_name(func)
            if dotted is None or dotted.split(".", 1)[0] != "cv2":
                self._record(self._index.methods, func.attr, node.lineno)
        self.generic_visit(node)


def _dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def iter_source_files(root: pathlib.Path) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for name in SOURCE_DIRS:
        directory = root / name
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            files.append(path)
    return files


def collect_source_symbols(root: pathlib.Path) -> SourceIndex:
    """Parse every source file and index its OpenCV call sites."""
    index = SourceIndex()
    for path in iter_source_files(root):
        rel = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:  # a file that does not parse is a file that does not run
            raise SystemExit(f"check_claims: cannot parse {rel}: {exc}") from exc
        _CallSiteVisitor(index, rel).visit(tree)
    return index


def classify(key: str) -> Surface | None:
    """The table entry for a source-side symbol, or ``None`` when it is unclassified."""
    surface = _BY_KEY.get(key)
    if surface is not None:
        return surface
    last = key.rsplit(".", 1)[-1]
    if key in MODULE_REFERENCES:
        return Surface(key, "module", SUPPORT)
    if last.startswith("__"):
        return Surface(key, "attribute", SUPPORT)
    if _FLAG_RE.match(last):
        return Surface(key, "flag", SUPPORT)
    return None


# ---------------------------------------------------------------------------------
# Document side -- what the documents claim
# ---------------------------------------------------------------------------------

_QUALIFIED_RE = re.compile(r"\bcv2?\s*(?:\.|::)\s*([A-Za-z_][A-Za-z0-9_]*(?:\s*(?:\.|::)\s*[A-Za-z_][A-Za-z0-9_]*)*)")
_BACKTICK_RE = re.compile(r"`([^`\n]{1,120})`")
_BARE_SYMBOL_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*(?:(?:\.|::)[A-Za-z_][A-Za-z0-9_]*)*)(?:\(\))?$")


def _normalise_symbol(raw: str) -> str:
    """One spelling for both sides: no ``cv::``/``cv2.`` root, ``::`` becomes ``.``."""
    text = re.sub(r"\s+", "", raw).rstrip("()")
    text = text.replace("::", ".")
    for prefix in ("cv2.", "cv."):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return text


def _resolves_in_cv2(name: str) -> bool:
    """Is this bare name an attribute of cv2, cv2.dnn or cv2.bgsegm?

    Asking the installed library rather than carrying a word list is what makes the
    document side self-maintaining: a symbol OpenCV 5 does not have cannot be claimed as
    an OpenCV 5 symbol, and one it does have is caught the first time a document names it.
    """
    head = name.split(".", 1)[0]
    for namespace in (cv2, getattr(cv2, "dnn", None), getattr(cv2, "bgsegm", None)):
        if namespace is not None and hasattr(namespace, head):
            return True
    return False


def collect_doc_symbols(root: pathlib.Path) -> dict[str, list[str]]:
    """OpenCV symbols each document names, normalised, with the documents that name them."""
    claims: dict[str, list[str]] = {}

    def add(symbol: str, doc: str) -> None:
        if symbol in IGNORED_DOC_SYMBOLS or not symbol:
            return
        if symbol.rsplit(".", 1)[-1] in IGNORED_DOC_SYMBOLS:
            return
        docs = claims.setdefault(symbol, [])
        if doc not in docs:
            docs.append(doc)

    for doc_path in existing_docs(root):
        doc = doc_path.name
        text = doc_path.read_text(encoding="utf-8")
        for match in _QUALIFIED_RE.finditer(text):
            add(_normalise_symbol(match.group(1)), doc)
        for match in _BACKTICK_RE.finditer(text):
            token = match.group(1).strip()
            bare = _BARE_SYMBOL_RE.match(token)
            if bare is None:
                continue
            symbol = _normalise_symbol(bare.group(1))
            if symbol in AMBIGUOUS_BARE_WORDS or symbol.split(".", 1)[0] in AMBIGUOUS_BARE_WORDS:
                continue
            if symbol in _BY_ALIAS or _resolves_in_cv2(symbol):
                add(symbol, doc)
    return claims


def existing_docs(root: pathlib.Path) -> list[pathlib.Path]:
    return [root / name for name in DOC_FILES if (root / name).is_file()]


def doc_text(root: pathlib.Path) -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in existing_docs(root))


_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.DOTALL)


def code_spans(text: str) -> str:
    """Only the parts of a document a reader would actually type.

    ``make sure the box is up`` is English. A checker that reads it as a Makefile target
    fails a judge-facing README for a sentence, and the repair a tired person applies at
    03:00 is to reword the prose rather than to fix a claim -- which is the checker
    training the team out of writing. Every command this project documents is written as
    code, so code is where commands are looked for.
    """
    parts = [m.group(1) for m in _FENCE_RE.finditer(text)]
    parts += [m.group(1) for m in _BACKTICK_RE.finditer(text)]
    return "\n".join(parts)


# ---------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    misses: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "misses": self.misses,
        }


def check_opencv_symbols(index: SourceIndex, claims: dict[str, list[str]]) -> Check:
    misses: list[str] = []
    for symbol in sorted(claims):
        surface = _BY_ALIAS.get(symbol)
        if surface is not None:
            if index.has(surface):
                continue
            misses.append(
                f"{symbol} (named in {', '.join(claims[symbol])}) is in the surface table "
                f"but has no call site under {'/, '.join(SOURCE_DIRS)}/"
            )
            continue
        if symbol in index.attributes:
            continue
        tail = symbol.rsplit(".", 1)[-1]
        if tail in index.methods:
            misses.append(
                f"{symbol} (named in {', '.join(claims[symbol])}) is called only as a bound "
                f"method; add it to SURFACES with method= so the claim is checked deliberately"
            )
            continue
        misses.append(
            f"{symbol} (named in {', '.join(claims[symbol])}) is never called in the source"
        )
    if misses:
        return Check("opencv-symbols", FAIL, f"{len(misses)} claimed symbol(s) not called", misses)
    return Check("opencv-symbols", PASS, f"{len(claims)} claimed symbol(s), all called")


def check_unclassified_surfaces(index: SourceIndex) -> Check:
    """A call the table has never seen is a surface nobody decided about.

    It fails rather than warns because the alternative -- silently dropping it -- is how a
    count becomes wrong in the direction that flatters us.
    """
    unknown = [
        f"cv2.{key} at {index.attributes[key][0]}"
        for key in sorted(index.attributes)
        if classify(key) is None
    ]
    if unknown:
        return Check(
            "surface-table-complete",
            FAIL,
            f"{len(unknown)} OpenCV call(s) missing from SURFACES -- classify each as "
            f"counted, drawing, or support",
            unknown,
        )
    return Check("surface-table-complete", PASS, "every OpenCV call in the source is classified")


def load_app(root: pathlib.Path) -> object | None:
    """Import ``api/main.py`` and hand back its ``app``, or ``None`` when there is no API."""
    main_py = root / "api" / "main.py"
    if not main_py.is_file():
        return None
    # A path-derived module name, because the tests import several different trees in one
    # process and a fixed name would serve the first one back from sys.modules forever.
    digest = hashlib.sha256(str(main_py.resolve()).encode()).hexdigest()[:12]
    module_name = f"_claims_api_main_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, main_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {main_py}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(root))
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(root))
    app = getattr(module, "app", None)
    if app is None:
        raise RuntimeError(f"{main_py} defines no `app`")
    return app


_DOC_ROUTE_RE = re.compile(r"\b(GET|POST|PUT|PATCH|DELETE)\b[\s|`]*(/[A-Za-z0-9_\-./{}]*)")


def check_routes(root: pathlib.Path, text: str) -> Check:
    documented: dict[str, set[str]] = {}
    for method, path in _DOC_ROUTE_RE.findall(text):
        documented.setdefault(path.rstrip("."), set()).add(method)

    try:
        app = load_app(root)
    except Exception as exc:  # an app we cannot import is an app we cannot verify
        return Check("routes", FAIL, f"api/main.py did not import: {exc!r}")

    if app is None:
        if documented:
            return Check(
                "routes",
                FAIL,
                "documents describe routes but api/main.py does not exist",
                sorted(f"{'/'.join(sorted(m))} {p}" for p, m in documented.items()),
            )
        return Check("routes", SKIP, "no api/main.py on disk and no route documented")

    served: dict[str, set[str]] = {}
    for route in getattr(app, "routes", []):
        path = getattr(route, "path", None)
        if path is None or path in FRAMEWORK_ROUTES:
            continue
        # HEAD is synthesised by Starlette for every GET; nobody documents it and
        # demanding that they do would be demanding a lie about the design.
        methods = {m for m in (getattr(route, "methods", None) or set()) if m != "HEAD"}
        served.setdefault(path, set()).update(methods)

    misses: list[str] = []
    for path, methods in sorted(documented.items()):
        if path not in served:
            misses.append(f"documented route {path} is not served by app.routes")
            continue
        for method in sorted(methods):
            if method not in served[path]:
                misses.append(f"documented {method} {path} -- app serves {sorted(served[path])}")
    for path in sorted(served):
        if path not in documented and path not in text:
            misses.append(f"app serves {path}, which no document mentions")

    if misses:
        return Check("routes", FAIL, f"{len(misses)} route mismatch(es)", misses)
    return Check("routes", PASS, f"{len(served)} served route(s), all documented both ways")


def lockfile_names(root: pathlib.Path) -> set[str]:
    names: set[str] = set()
    for lock in sorted(root.glob("requirements*.lock")):
        for line in lock.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            name = re.split(r"[=<>!\[; ]", line, maxsplit=1)[0].strip()
            if name:
                names.add(name.lower().replace("_", "-"))
    return names


_PIP_INSTALL_RE = re.compile(r"pip install\s+([^\n`]+)")
_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)


def check_dependencies(root: pathlib.Path, text: str) -> Check:
    locked = lockfile_names(root)
    if not locked:
        return Check("dependencies", FAIL, "no requirements*.lock to check against")

    named: set[str] = set()
    for match in _PIP_INSTALL_RE.finditer(text):
        for token in match.group(1).split():
            if token.startswith("-"):
                continue
            named.add(re.split(r"[=<>!\[]", token)[0])
    named.update(_IMPORT_RE.findall(text))
    for match in _BACKTICK_RE.finditer(text):
        token = match.group(1).strip()
        if token in WATCHED_PACKAGES:
            named.add(token)

    stdlib = getattr(sys, "stdlib_module_names", frozenset())
    misses: list[str] = []
    for raw in sorted(named):
        if raw in stdlib or raw == "plumerank" or raw == "api" or raw == "tests":
            continue
        canonical = IMPORT_ALIASES.get(raw, raw).lower().replace("_", "-")
        if canonical in locked:
            continue
        if raw in WATCHED_PACKAGES or raw in IMPORT_ALIASES:
            misses.append(f"{raw} is named in the docs but is in no requirements*.lock")
    if misses:
        return Check("dependencies", FAIL, f"{len(misses)} unlocked dependency claim(s)", misses)
    return Check("dependencies", PASS, f"{len(locked)} locked package(s); no doc names another")


_MAKE_RE = re.compile(r"\bmake\s+([a-zA-Z][a-zA-Z0-9_-]*)")
_TARGET_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9_-]*)\s*:(?!=)", re.MULTILINE)


def check_makefile_targets(root: pathlib.Path, text: str) -> Check:
    makefile = root / "Makefile"
    if not makefile.is_file():
        return Check("makefile-targets", FAIL, "no Makefile")
    body = makefile.read_text(encoding="utf-8")
    targets = set(_TARGET_RE.findall(body))
    for line in body.splitlines():
        if line.startswith(".PHONY:"):
            targets.update(line.split(":", 1)[1].split())
    mentioned = set(_MAKE_RE.findall(code_spans(text)))
    misses = [f"docs print `make {t}`, which the Makefile does not define" for t in sorted(mentioned - targets)]
    if misses:
        return Check("makefile-targets", FAIL, f"{len(misses)} missing target(s)", misses)
    return Check(
        "makefile-targets",
        PASS,
        f"{len(mentioned)} target(s) mentioned, all defined ({len(targets)} exist)",
    )


_PATH_RE = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.@-]+)+/?$")


def check_claimed_paths(root: pathlib.Path, text: str) -> Check:
    claimed: set[str] = set()
    for match in _BACKTICK_RE.finditer(text):
        token = match.group(1).strip()
        if not _PATH_RE.match(token):
            continue
        if token.split("/", 1)[0] not in CLAIMABLE_PATH_ROOTS:
            continue
        claimed.add(token)
    misses = [
        f"docs name {token}, which does not exist on disk"
        for token in sorted(claimed)
        if not (root / token).exists()
    ]
    if misses:
        return Check("claimed-paths", FAIL, f"{len(misses)} claimed path(s) missing", misses)
    return Check("claimed-paths", PASS, f"{len(claimed)} claimed path(s), all present")


# ---------------------------------------------------------------------------------
# The count
# ---------------------------------------------------------------------------------


@dataclass
class SurfaceCount:
    """What the source contains today. Never what the plan predicted."""

    by_module: dict[str, list[str]]
    absent: list[str]
    features: list[str]
    drawing: list[str]
    support: list[str]

    @property
    def total(self) -> int:
        return sum(len(v) for v in self.by_module.values())

    @property
    def modules(self) -> int:
        return len(self.by_module)

    def to_json(self) -> dict:
        return {
            "defended_surfaces_found_today": self.total,
            "modules": self.modules,
            "by_module": {m: sorted(v) for m, v in self.by_module.items()},
            "in_table_but_not_called_today": self.absent,
            "used_but_not_counted": {
                "features": self.features,
                "drawing_and_io": self.drawing,
                "support_and_flags": self.support,
            },
        }


def count_surfaces(index: SourceIndex) -> SurfaceCount:
    by_module: dict[str, list[str]] = {}
    absent: list[str] = []
    features: list[str] = []
    drawing: list[str] = []
    support: list[str] = []

    for surface in SURFACES:
        present = index.has(surface)
        if surface.group == COUNTED:
            if present:
                by_module.setdefault(surface.module, []).append(surface.key)
            else:
                absent.append(surface.key)
        elif present and surface.group == FEATURES:
            features.append(f"{surface.module}::{surface.key}")
        elif present and surface.group == DRAWING:
            drawing.append(f"{surface.module}::{surface.key}")
        elif present:
            support.append(f"{surface.module}::{surface.key}")

    # Flags, module references and dunders are grouped by what they are rather than
    # lumped together, so a reader can see at a glance that none of them is a surface.
    for key in sorted(index.attributes):
        if key in _BY_KEY:
            continue
        implicit = classify(key)
        if implicit is not None:
            support.append(f"{implicit.module}::{key}")

    ordered = {m: sorted(by_module[m]) for m in MODULE_ORDER if m in by_module}
    for module in sorted(set(by_module) - set(MODULE_ORDER)):
        ordered[module] = sorted(by_module[module])
    return SurfaceCount(ordered, sorted(absent), sorted(features), sorted(drawing), sorted(support))


# ---------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------


@dataclass
class Report:
    root: pathlib.Path
    checks: list[Check]
    count: SurfaceCount
    docs: list[str]
    generated: str

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status == FAIL]

    @property
    def skipped(self) -> list[Check]:
        return [c for c in self.checks if c.status == SKIP]

    @property
    def exit_code(self) -> int:
        return 1 if self.failed else 0

    def to_json(self) -> dict:
        return {
            "generated": self.generated,
            "root": str(self.root),
            "documents_checked": self.docs,
            "checks": [c.to_json() for c in self.checks],
            "surfaces": self.count.to_json(),
            "ok": not self.failed,
            # `ok` means nothing failed. `complete` means everything ran. Before the first
            # judge-facing document exists the two differ, and a reader who is handed only
            # the first one reads a green tick over a tree nothing examined.
            "complete": not self.skipped,
        }

    def render(self) -> str:
        lines: list[str] = []
        lines.append("PlumeRank claim check")
        lines.append("=" * 72)
        lines.append(
            "documents checked: " + (", ".join(self.docs) if self.docs else "none on disk yet")
        )
        lines.append("")
        for check in self.checks:
            lines.append(f"  [{check.status}] {check.name:<24} {check.detail}")
            for miss in check.misses:
                lines.append(f"          - {miss}")
        lines.append("")
        lines.append("-" * 72)
        lines.append(
            f"DEFENDED OPENCV 5 ANALYSIS SURFACES FOUND TODAY: "
            f"{self.count.total} across {self.count.modules} module(s)"
        )
        for module, keys in self.count.by_module.items():
            lines.append(f"  {module:<10} {len(keys):>2}  {', '.join(keys)}")
        if self.count.absent:
            lines.append(
                "  in the table, not called today: " + ", ".join(self.count.absent)
            )
        lines.append(
            "  This number is measured from the source on disk. A document may quote only"
        )
        lines.append(
            "  the number printed on the day that document was written -- never a forecast."
        )
        lines.append("")
        lines.append("USED, AND DELIBERATELY NOT COUNTED AS ANALYSIS DEPTH:")
        lines.append(
            "  analysis, wrong module: "
            + (", ".join(self.count.features) if self.count.features else "none")
        )
        lines.append(
            "  drawing and IO:         "
            + (", ".join(self.count.drawing) if self.count.drawing else "none")
        )
        lines.append(
            "  support and flags:      "
            + (", ".join(self.count.support) if self.count.support else "none")
        )
        lines.append("-" * 72)
        if self.failed:
            lines.append(f"FAIL -- {len(self.failed)} check(s) failed")
        elif self.skipped:
            # Saying PASS here would be this script committing the exact offence it was
            # written to catch: a green line over checks that never ran.
            lines.append(
                f"PASS with {len(self.skipped)} check(s) SKIPPED -- nothing failed, and "
                f"the document checks did not run"
            )
        else:
            lines.append("PASS -- every claim on disk is backed by code")
        return "\n".join(lines)


def run(root: pathlib.Path) -> Report:
    """Every check, against the tree at ``root``."""
    root = pathlib.Path(root)
    index = collect_source_symbols(root)
    docs = [p.name for p in existing_docs(root)]
    text = doc_text(root)
    checks: list[Check] = [check_unclassified_surfaces(index)]

    if docs:
        checks.append(check_opencv_symbols(index, collect_doc_symbols(root)))
        checks.append(check_routes(root, text))
        checks.append(check_dependencies(root, text))
        checks.append(check_makefile_targets(root, text))
        checks.append(check_claimed_paths(root, text))
    else:
        # Nothing claimed is nothing to check. The count below still prints, because it is
        # read out of the source and is what the documents will have to quote.
        for name in ("opencv-symbols", "routes", "dependencies", "makefile-targets", "claimed-paths"):
            checks.append(Check(name, SKIP, "no judge-facing document on disk yet"))

    return Report(
        root=root,
        checks=checks,
        count=count_surfaces(index),
        docs=docs,
        generated=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    )


def write_json(report: Report, out_dir: pathlib.Path) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.date.today().isoformat()
    path = out_dir / f"claims_{stamp}.json"
    path.write_text(json.dumps(report.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent.parent,
        help="repository root (default: the tree this script lives in)",
    )
    parser.add_argument("--out-dir", type=pathlib.Path, default=None, help="where claims_<date>.json goes")
    parser.add_argument("--no-json", action="store_true", help="print the report, write nothing")
    args = parser.parse_args(argv)

    report = run(args.root)
    print(report.render())
    if not args.no_json:
        out_dir = args.out_dir if args.out_dir is not None else args.root / "results"
        print(f"\nwrote {write_json(report, out_dir)}")
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
