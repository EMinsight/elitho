from elitho import config
from elitho.utils.backend import get_array_module


def polarization_rotation(k, MX, MY, px, py, sx0, sy0, xp):
    s0 = xp.array([sx0, sy0, -xp.sqrt(k**2 - sx0**2 - sy0**2)])
    p = xp.array([px, py, 0.0])
    pp = xp.array([MX * px, MY * py, -xp.sqrt(k**2 - (MX * px) ** 2 - (MY * py) ** 2)])
    ez = xp.array([0.0, 0.0, 1.0])
    ezp = xp.array([0.0, 0.0, 1.0])
    eps = 1e-5

    if (px**2 + py**2) > eps:
        es = xp.cross(p, s0)
        es = es / xp.linalg.norm(es)
        esp = xp.cross(pp, ezp)
        esp = esp / xp.linalg.norm(esp)

        ps0 = p + s0
        ps0 = ps0 / xp.linalg.norm(ps0)

        em = xp.cross(es, ps0)
        emp = xp.cross(esp, pp) / k

    else:
        es = xp.cross(ez, s0)
        es = es / xp.linalg.norm(es)
        esp = -es
        em = xp.cross(es, s0) / k
        emp = xp.cross(esp, -ezp)

    R = xp.zeros((3, 2))
    scale = xp.sqrt(k / abs(pp[2]))

    for i in range(3):
        for j in range(2):
            R[i, j] = scale * (esp[i] * es[j] + emp[i] * em[j])

    return R


def high_na_electro_field(sc, nsx, nsy, Ax_val, Ay_val, linput, minput, l0s, m0s, xp):
    kxn = sc.dkx * (nsx / sc.ndivX) + sc.dkx * l0s + sc.dkx * linput
    kyn = sc.dky * (nsy / sc.ndivY) + sc.dky * m0s + sc.dky * minput
    EAx = 0.0
    EAy = 0.0
    EAz = 0.0
    p2 = sc.magnification_x**2 * kxn**2 + sc.magnification_y**2 * kyn**2
    if all(
        [
            (sc.NA * sc.k * sc.central_obscuration) ** 2 <= p2,
            p2 <= (sc.NA * sc.k) ** 2,
        ]
    ):
        R = polarization_rotation(
            sc.k, sc.magnification_x, sc.magnification_y, kxn, kyn, sc.kx0, sc.ky0, xp
        )
        EAx = 1j * sc.k * (R[0, 0] * Ax_val + R[0, 1] * Ay_val)
        EAy = 1j * sc.k * (R[1, 0] * Ax_val + R[1, 1] * Ay_val)
        EAz = 1j * sc.k * (R[2, 0] * Ax_val + R[2, 1] * Ay_val)
    return EAx, EAy, EAz


def standard_na_electro_field(sc, kxplus, kyplus, Ax_val, Ay_val, xp):
    kxy2 = kxplus**2 + kyplus**2
    klm = xp.sqrt(sc.k**2 - kxy2)
    EAx = 1j * sc.k * Ax_val - 1j / sc.k * (
        kxplus**2 * Ax_val + kxplus * kyplus * Ay_val
    )
    EAy = 1j * sc.k * Ay_val - 1j / sc.k * (
        kxplus * kyplus * Ax_val + kyplus**2 * Ay_val
    )
    EAz = 1j * klm / sc.k * (kxplus * Ax_val + kyplus * Ay_val)
    return EAx, EAy, EAz


def electro_field(
    sc: config.SimulationConfig,
    polar: config.PolarizationDirection,
    is_high_na: bool,
    nsx: int,
    nsy: int,
    SDIV: int,
    l0s: "xp.ndarray",
    m0s: "xp.ndarray",
    sx0: float,
    sy0: float,
    pupil_coords: "pupil.PupilCoordinates",
    ampxx: "xp.ndarray",
) -> tuple["xp.ndarray", "xp.ndarray", "xp.ndarray"]:
    xp = get_array_module(ampxx)

    Ex0m = xp.zeros((SDIV, pupil_coords.n_coordinates), dtype=complex)
    Ey0m = xp.zeros_like(Ex0m)
    Ez0m = xp.zeros_like(Ex0m)

    for isd in range(SDIV):
        kx = sx0 + sc.dkx * l0s[isd]
        ky = sy0 + sc.dky * m0s[isd]
        ls = l0s[isd] + sc.lsmaxX
        ms = m0s[isd] + sc.lsmaxY
        for i in range(pupil_coords.n_coordinates):
            ip = pupil_coords.linput[i] + sc.lpmaxX
            jp = pupil_coords.minput[i] + sc.lpmaxY

            if polar == config.PolarizationDirection.X:
                Ax_val = ampxx[ls, ms, ip, jp] / xp.sqrt(sc.k**2 - kx**2)
                Ay_val = 0
            elif polar == config.PolarizationDirection.Y:
                Ax_val = 0
                Ay_val = ampxx[ls, ms, ip, jp] / xp.sqrt(sc.k**2 - ky**2)
            else:
                raise ValueError("polar must be 'X' or 'Y'")

            if is_high_na:
                EAx, EAy, EAz = high_na_electro_field(
                    sc,
                    nsx,
                    nsy,
                    Ax_val,
                    Ay_val,
                    pupil_coords.linput[i],
                    pupil_coords.minput[i],
                    l0s[isd],
                    m0s[isd],
                    xp,
                )
            else:
                kxplus = kx + sc.dkx * pupil_coords.linput[i]
                kyplus = ky + sc.dky * pupil_coords.minput[i]
                EAx, EAy, EAz = standard_na_electro_field(
                    sc, kxplus, kyplus, Ax_val, Ay_val, xp
                )

            Ex0m[isd, i] = EAx
            Ey0m[isd, i] = EAy
            Ez0m[isd, i] = EAz

    return Ex0m, Ey0m, Ez0m
