import mitsuba as mi
import drjit as dr


def pcg(v: mi.UInt32) -> mi.UInt32:
    state = (v * 747796405) + 2891336453
    word = ((state >> ((state >> 28) + 4)) ^ state) * 277803737
    return (word >> 22) ^ word


def uint32_to_uniform_float(v: mi.UInt32) -> mi.Float32:
    return dr.reinterpret_array_v(mi.Float32, (v | 0x3F800000) >> 9) - 1.0
