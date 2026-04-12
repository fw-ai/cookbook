from .turboquant import TurboQuantProd
from .lloyd_max import LloydMaxCodebook, solve_lloyd_max
from .rotorquant import RotorQuantMSE
from .clifford import geometric_product, make_random_rotor, rotor_sandwich

try:
    from .triton_kernels import (
        triton_rotor_sandwich,
        triton_rotor_inverse_sandwich,
        pack_rotors_for_triton,
    )
    _triton_available = True
except ImportError:
    _triton_available = False
