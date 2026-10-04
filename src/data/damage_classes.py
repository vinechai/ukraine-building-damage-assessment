# damage labels for the unosat/pwtt ukraine benchmark.
# it's binary: a building is "damaged" if its footprint overlaps a unosat damage point,
# "undamaged" otherwise. unosat has finer damage grades internally but this benchmark
# only keeps yes/no.

UNDAMAGED = 0
DAMAGED = 1

CLASS_NAMES = {
    UNDAMAGED: "undamaged",
    DAMAGED: "damaged",
}
