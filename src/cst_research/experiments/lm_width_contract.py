"""Prespecified width and duration screen, without GPU imports."""

from dataclasses import asdict, dataclass

from .ffn_contract import ModelConfig

WIDTHS = (512, 1024, 2048)
DENSE_LRS = (0.0001, 0.0003, 0.001)
DST_LRS = (0.0003, 0.001, 0.003, 0.01)
SHORT_STEPS = 976
LONG_STEPS = 4882
SCHEDULE_STEPS = 20000000 // 1024
EVALUATIONS = tuple(
    sorted({0, 200, 256, 400, 600, 800, 976, *range(1200, LONG_STEPS, 400), LONG_STEPS})
)
CHECKPOINTS = (256, SHORT_STEPS, 2441, LONG_STEPS)
ARMS = (
    {f"dense-lr{lr:g}": ("dense", lr) for lr in DENSE_LRS}
    | {"cst-c64": ("cst", 0.0003)}
    | {f"dst-lr{lr:g}": ("dst", lr) for lr in DST_LRS}
)


@dataclass(frozen=True)
class WidthConfig(ModelConfig):
    """Private width admission leaves the ordinary ModelConfig unchanged."""

    width: int = 512
    heads: int = 8
    coefficient: int = 64
    radius_mode: str = "envelope"
    linear_backend: str = "levels_wide_amplitude"
    amplitude_max: float = 32.0
    tau: float = 0.003
    radius_fraction: float = 0.0
    atom_lr_multiplier: float = 32.0
    radius_fraction_lr_multiplier: float = 32.0

    def __post_init__(self):
        ModelConfig(**(asdict(self) | {"width": 512, "coefficient": 16}))
        if (
            self.width not in WIDTHS
            or type(self.width) is not int
            or self.coefficient != 64
        ):
            raise ValueError("fixed-c64 widths512/1024/2048 only")
        if (self.layers, self.heads, self.context, self.expansion, self.vocabulary) != (
            4,
            8,
            128,
            4,
            50257,
        ):
            raise ValueError(
                "fixed four-block TinyStories GPT2-token protocol required"
            )


def slots_per_map(width):
    return WidthConfig(width=width).atoms_per_projection


def choose(candidates, kind):
    """DEV-selected short-run LR, tie smaller LR; not independent confirmation."""
    rows = [(name, row) for name, row in candidates.items() if ARMS[name][0] == kind]
    if {n for n, _ in rows} != {n for n in ARMS if ARMS[n][0] == kind}:
        raise ValueError("all prespecified LR candidates are required")
    return min(rows, key=lambda item: (item[1]["ce"], ARMS[item[0]][1]))[0]
