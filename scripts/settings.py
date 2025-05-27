from os.path import abspath, dirname, join

AAV = "aav"
YEAST = "qtls_li_hq"
SMN1 = "smn1"
GB1 = "gb1"

DATASETS = [AAV, YEAST, SMN1, GB1]

AMINOACIDS = [
    "R",
    "K",
    "E",
    "D",
    "Q",
    "N",
    "H",
    "S",
    "T",
    "A",
    "V",
    "I",
    "L",
    "M",
    "P",
    "G",
    "Y",
    "F",
    "W",
    "C",
]
POSITIONS = {
    GB1: ["39", "40", "41", "54"],
    AAV: [str(x) for x in range(561, 589)],
    SMN1: ["-3", "-2", "-1", "+2", "+3", "+4", "+5", "+6"],
}
REF_SEQS = {GB1: "VDGV", SMN1: "CAGUAAGU", AAV: "DEEEIRTTNPVATEQYGSVSTNLQRGNR"}
ALPHABET = {
    GB1: AMINOACIDS,
    AAV: AMINOACIDS,
    YEAST: ["A", "B"],
    SMN1: ["A", "C", "G", "U"],
}
LENGTH = {
    GB1: 4,
    AAV: 28,
    YEAST: 83,
    SMN1: 8,
}

MODELS = [
    "Additive",
    "Global epistasis",
    "Pairwise",
    "Variance Component",
    "Exponential",
    "Connectedness",
    "Jenga",
    "General Product",
]

ORDER = [
    "Additive",
    "Global epistasis",
    "Pairwise",
    "Exponential",
    "Variance Component",
    "Connectedness",
    "Jenga",
    "General Product",
]
MODEL_KEYS = {
    "Global epistasis": "global_epistasis",
    "Variance Component": "VC",
    "General Product": "GeneralProduct",
}
GB1_PEAK_SEQS = ["VDGV", "WWLG", "LICA"]

AAV_BACKGROUNDS = {'WT': 'DEEEIRTTNPVATEQYGSVSTNLQRGNR',
                   'N569Q': 'DEEEIRTTQPVATEQYGSVSTNLQRGNR',
                   'Y576F': 'DEEEIRTTNPVATEQFGSVSTNLQRGNR',
                   'Y576C': 'DEEEIRTTNPVATEQCGSVSTNLQRGNR',
                   'N587E': 'DEEEIRTTNPVATEQYGSVSTNLQRGER',
                   'R585E+N587E': 'DEEEIRTTNPVATEQYGSVSTNLQEGER'}

# File paths
BASEDIR = join(dirname(abspath(__file__)), '..',)
DATADIR = join(BASEDIR, "data")
SPLITSDIR = join(DATADIR, "splits")
OUTDIR = join(BASEDIR, "output")
PARAMSDIR = join(OUTDIR, "models")
RESULTSDIR = join(BASEDIR, "results")
FIGDIR = join(BASEDIR, "figures")