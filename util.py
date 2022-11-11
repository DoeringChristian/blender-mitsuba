import mitsuba as mi
import drjit as dr


def pcg(state: mi.UInt32) -> mi.UInt32:
    oldstate = state
    state = dr.fma(oldstate, mi.UInt64(0x5851F42D4C957F2D), 0)
    xorshift = mi.UInt32((oldstate ^ (oldstate >> 18)) >> 27)
    rot = mi.UInt32(oldstate >> 59)

    return (xorshift >> rot) | (xorshift << ((-mi.Int32(rot)) & 31))


def uint32_to_uniform_float(v: mi.UInt32) -> mi.Float32:
    return dr.reinterpret_array_v(mi.Float32, (v | 0x3F800000) >> 9) - 1.0
