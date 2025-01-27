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
    "gb1": ["39", "40", "41", "54"],
    "aav": [str(x) for x in range(561, 589)],
    "smn1": ["-3", "-2", "-1", "+2", "+3", "+4", "+5", "+6"],
}
REF_SEQS = {"gb1": "VDGV", "smn1": "CAGUAAGU",
            'aav': 'DEEEIRTTNPVATEQYGSVSTNLQRGNR'}
ALPHABET = {
    "gb1": AMINOACIDS,
    "aav": AMINOACIDS,
    "qtls_li_hq": ["A", "B"],
    "smn1": ["A", "C", "G", "U"],
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
