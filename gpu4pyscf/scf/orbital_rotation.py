# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Orbital rotation step of a second-order SCF: from the Newton step k to the
updated MO coefficients C.

The MO coefficients C are an (nao, nmo) matrix whose columns are ordered
occupied first and virtual second, i.e. the occupied orbitals are C[:, :n_occ],
the virtual orbitals are C[:, n_occ:], and nmo = n_occ + n_vir.

The rotation vector k is a 1D float64 array of length n_vir*n_occ holding the
occupied-virtual block of the rotation generator in C order, virtual major::

    kappa = k.reshape(n_vir, n_occ)

so that kappa[a, i] == k[a*n_occ + i] is the rotation mixing occupied orbital i
with virtual orbital a (eq. 3.2 / 3.3 and eq. C.5 of the SOSCF derivation).
The antisymmetric generator and the update of the orbitals are

    K = [[0, -kappa.T], [kappa, 0]]     # (nmo, nmo), eq. 8.1
    U = expm(K)                         # orthogonal,          eq. 8.2
    C_new = C @ U

Positive kappa[a, i] mixes virtual orbital a into occupied orbital i, i.e. the
first-order update is C[:, i] <- C[:, i] + kappa[a, i] * C[:, a] up to
normalization (eq. 8.7). This is the same generator as ``x1 - x1.T`` with
``x1[vir, occ] = kappa`` in ``gpu4pyscf/scf/soscf.py`` ``update_rotate_matrix``.
Because U is orthogonal, C_new stays orthonormal, C_new.T S C_new = C.T S C
(eq. 3.10), so in a non-orthogonal AO basis the orthonormality of the input C
(with respect to the overlap S) is preserved exactly.

A small-|k| Taylor expansion of U is available as well (eqs. 8.5-8.7)::

    U = I + K + K^2/2 + O(|k|^3)
    Uoo = I - kappa.T @ kappa / 2 + O(|k|^4)
    Uvo = kappa - kappa @ kappa.T @ kappa / 6 + O(|k|^5)
    Co' = Co + Cv @ kappa - Co @ kappa.T @ kappa / 2 + O(|k|^3)

It avoids the (nmo, nmo) matrix exponential but no longer gives exactly
orthonormal orbitals.

WARNING: the tdscf-ris code uses the opposite, occupied-major layout for the
same occupied-virtual block. ``gpu4pyscf/tdscf/ris.py`` (see ``gen_hdiag_MVP``)
and the Krylov solver in ``gpu4pyscf/tdscf/_krylov_tools.py`` store the vectors
as ``X.reshape(nstates, n_occ, n_vir)``, i.e. flattened with index
``i*n_vir + a``, whereas this module expects the virtual-major index
``a*n_occ + i``. Converting between the two layouts when the step of an SOSCF
driver is taken from the ris Hessian is the caller's responsibility, e.g.
``k = x.reshape(n_occ, n_vir).T.reshape(-1)`` for a 1D ris vector x.
'''

import numpy as np
import cupy as cp
from cupyx.scipy.linalg import expm

def _check_n_occ(n_occ):
    '''Validate the number of occupied orbitals.

    Returns n_occ as a Python int.
    '''
    if isinstance(n_occ, bool) or not isinstance(n_occ, (int, np.integer)):
        raise TypeError(f'n_occ must be an integer, got {type(n_occ).__name__}')
    if n_occ <= 0:
        raise ValueError(f'n_occ must be a positive number of occupied orbitals, got {n_occ}')
    return int(n_occ)

def _check_rotation_vector(k, n_occ):
    '''Validate the rotation vector k of length n_vir*n_occ.

    Returns the array as a cupy array and the number of virtual orbitals n_vir.
    '''
    k = cp.asarray(k)
    assert k.dtype == np.float64, f'k must be float64, got {k.dtype}'
    if k.ndim != 1:
        raise ValueError(f'k must be a 1D vector, got shape {k.shape}')
    if k.size == 0:
        raise ValueError('k must contain at least one occupied-virtual rotation')
    if k.size % n_occ != 0:
        raise ValueError(f'k of length {k.size} is not nvir*nocc: '
                         f'it is not divisible by the number of occupied orbitals {n_occ}')
    return k, k.size // n_occ

def _check_mo_coeff(C, nmo):
    '''Validate the MO coefficient matrix C of shape (nao, nmo).'''
    C = cp.asarray(C)
    assert C.dtype == np.float64, f'C must be float64, got {C.dtype}'
    if C.ndim != 2:
        raise ValueError(f'C must be a 2D array, got shape {C.shape}')
    if C.shape[1] != nmo:
        raise ValueError(f'C has {C.shape[1]} MO columns, expected nmo = n_occ + n_vir = {nmo}')
    return C

def k_to_kappa(k, n_occ):
    '''Reshape the rotation vector k into the (n_vir, n_occ) block kappa.

    kappa[a, i] == k[a*n_occ + i] is the rotation mixing occupied orbital i with
    virtual orbital a (eq. 3.2 / 3.3 and eq. C.5 of the SOSCF derivation).
    '''
    n_occ = _check_n_occ(n_occ)
    k, n_vir = _check_rotation_vector(k, n_occ)
    return k.reshape(n_vir, n_occ)

def kappa_to_k(kappa):
    '''Flatten an (n_vir, n_occ) block kappa into the rotation vector k.

    Inverse of :func:`k_to_kappa`.
    '''
    kappa = cp.asarray(kappa)
    assert kappa.dtype == np.float64, f'kappa must be float64, got {kappa.dtype}'
    if kappa.ndim != 2:
        raise ValueError(f'kappa must be a 2D array, got shape {kappa.shape}')
    return kappa.reshape(-1)

def build_rotation_generator(k, n_occ):
    '''Assemble the antisymmetric rotation generator K from the vector k.

    K = [[0, -kappa.T], [kappa, 0]] with kappa = k.reshape(n_vir, n_occ)
    (eq. 8.1 of the SOSCF derivation). The occupied orbitals come first, so K
    is an (nmo, nmo) matrix with nmo = n_occ + n_vir = n_occ + k.size//n_occ.
    '''
    n_occ = _check_n_occ(n_occ)
    k, n_vir = _check_rotation_vector(k, n_occ)
    nmo = n_occ + n_vir
    kappa = k.reshape(n_vir, n_occ)
    K = cp.zeros((nmo, nmo), dtype=np.float64)
    K[n_occ:, :n_occ] = kappa
    K[:n_occ, n_occ:] = -kappa.T
    return K

def rotation_matrix(k, n_occ):
    '''Orthogonal rotation matrix U = expm(K) associated with the step k.

    K is built by :func:`build_rotation_generator`. The matrix exponential is
    evaluated with cupyx.scipy.linalg.expm, the same way as the CIAH solver in
    gpu4pyscf/scf/soscf.py (eq. 8.2 of the SOSCF derivation).
    '''
    return expm(build_rotation_generator(k, n_occ))

def update_mo_coeff(C, k, n_occ, method='expm', order=2):
    '''Update the MO coefficients C with the orbital rotation step k.

    Args:
        C : (nao, nmo) float64 MO coefficients, occupied first then virtual.
        k : 1D float64 rotation vector of length n_vir*n_occ, virtual major.
        n_occ : number of occupied orbitals, C[:, :n_occ].
        method : 'expm' for C @ expm(K) (eq. 8.2), or 'taylor' for the
            small-step expansion C @ (I + K + ... ) (eqs. 8.5-8.7).
        order : order of the Taylor expansion, 1 or 2. Only used by
            method='taylor'. The order 2 result, C @ (I + K + K^2/2), equals
            Co' = Co + Cv @ kappa - Co @ kappa.T @ kappa / 2 in the occupied
            block and is accurate to O(|k|^3).

    Returns:
        The updated (nao, nmo) float64 cupy array C_new = C @ U. With
        method='expm', C_new.T @ S @ C_new == C.T @ S @ C for any overlap S.
    '''
    n_occ = _check_n_occ(n_occ)
    k, n_vir = _check_rotation_vector(k, n_occ)
    C = _check_mo_coeff(C, n_occ + n_vir)

    if method == 'expm':
        U = rotation_matrix(k, n_occ)
    elif method == 'taylor':
        K = build_rotation_generator(k, n_occ)
        if order == 1:
            U = cp.eye(K.shape[0], dtype=np.float64) + K
        elif order == 2:
            U = cp.eye(K.shape[0], dtype=np.float64) + K + K.dot(K) * .5
        else:
            raise ValueError(f"order must be 1 or 2 for method='taylor', got {order!r}")
    else:
        raise ValueError(f"method must be 'expm' or 'taylor', got {method!r}")
    return C.dot(U)

def mo_orthonormality_error(C, S=None):
    '''Largest absolute deviation of C.T @ S @ C from the identity matrix.

    Args:
        C : (nao, nmo) MO coefficients.
        S : (nao, nao) AO overlap matrix. The identity is used when S is None.

    Returns:
        max(abs(C.T @ S @ C - I)) as a Python float. Squared MO coefficients in
        a non-orthogonal basis should satisfy C.T @ S @ C = I to numerical
        precision (appendix B of the SOSCF derivation).
    '''
    C = cp.asarray(C)
    assert C.dtype == np.float64, f'C must be float64, got {C.dtype}'
    if C.ndim != 2:
        raise ValueError(f'C must be a 2D array, got shape {C.shape}')
    if S is None:
        gram = C.T.dot(C)
    else:
        S = cp.asarray(S)
        assert S.dtype == np.float64, f'S must be float64, got {S.dtype}'
        if S.ndim != 2 or S.shape[0] != S.shape[1]:
            raise ValueError(f'S must be a square 2D overlap matrix, got shape {S.shape}')
        if S.shape[0] != C.shape[0]:
            raise ValueError(f'S of shape {S.shape} does not match the {C.shape[0]} AO basis of C')
        gram = C.T.dot(S.dot(C))
    return float(cp.abs(gram - cp.eye(C.shape[1], dtype=np.float64)).max())
