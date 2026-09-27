#!/usr/bin/env python
# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
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

##############################################################################
#  Example of the orbital rotation step k -> C_new used by the second-order
#  SCF (gpu4pyscf/scf/orbital_rotation.py), validated with GPU RHF on
#  H2O, NH3, C2H6 and unsymmetrical dimethylhydrazine (CH3)2NNH2
##############################################################################

"""
Numerical validation of the MO update of a second-order SCF,

    K  = [[0, -kappa.T], [kappa, 0]]         kappa = k.reshape(n_vir, n_occ)
    U  = expm(K)
    C' = C @ U                               (C' orthonormal w.r.t. the AO overlap S)

for the four molecules H2O, NH3, C2H6 and (CH3)2NNH2 (UDMH) in def2-SVP. All
four RHF calculations are converged on the GPU with density fitting.

Conventions (see the module docstring of gpu4pyscf/scf/orbital_rotation.py):
the MO coefficients are ordered occupied first, and the rotation vector k is
1D and virtual major, kappa[a, i] = k[a*n_occ + i] mixing occupied orbital i
into virtual orbital a. The energy gradient in this coordinate is

    g = vec(4 F_ai),  F = C.T F_AO C,  E(k) = E0 + g.k + O(|k|^2)

so the preconditioned (quasi-Newton) step is
kappa[a, i] = -g_ai / (4 (e_a - e_i)) with the orbital energies e (section 5,
section 7 and appendix A of the SOSCF derivation). Stepping along -g must lower
the energy.

What is checked here, per molecule and for the step norms |k| = 1e-1 ... 1e-4:
  1. the orthonormality of the updated orbitals, max|C'.T S C' - I|, for the
     'expm' and the 'taylor' order 2 updates (machine precision for expm,
     O(|k|^3) for the truncated Taylor expansion),
  2. the GPU update against a CPU reference built from the *same* generator,
     max|C_gpu - C_scipy| with scipy.linalg.expm(K),
  3. the linear energy model E(C') - E0 vs g.k, i.e. the ratio
     (E(C') - E0) / (g.k), which tends to 1 as |k| -> 0, plus the appendix B
     central difference (E(+h) - E(-h)) / (2h) = g.x along the step direction,
  4. the sign of the energy change, a step along -g must lower the energy.

The step is taken from a non-stationary state: at the converged solution
F_ai = 0 (|4 F_ai| ~ 1e-6 - 1e-5 for these molecules), where the linear model
is degenerate -- the finite rotation is then dominated by the quadratic term.
Appendix B of the SOSCF derivation prescribes the model check for arbitrary
non-converged orthogonal orbitals, so the validation uses the state

    C0 = C_scf @ expm(K0)

with K0 built from a fixed (seeded) random kappa0 rescaled to a small
max|kappa0|. C0 is a generic, deterministic non-stationary SCF state, i.e. the
kind of state a second-order SCF step actually operates on. The step state is
orthonormal by construction (expm of an antisymmetric matrix) and its energy
E0 is evaluated with the same density fitted RHF object as everything else.

Acceptance thresholds used by the final summary:
    orthonormality error of the expm update  < 1e-10
    max|C_gpu - C_scipy|                     < 1e-10
    |ratio - 1| at |k| = 1e-4                < 3e-2

The SCF energy and the density are reproducible to ~1e-14, but the *virtual*
MO basis of the converged solution is only defined up to the near degenerate
mixing left by the last Fock diagonalization (~1e-8 in the Fock). E0 is the
energy of a non-stationary state and is therefore sensitive to it (it moves in
the 4th decimal from run to run); the checked quantities -- orthonormality, the
CPU reference and the energy ratios -- do not.
"""

import cupy as cp
import numpy as np
import pyscf
import scipy.linalg

from gpu4pyscf.scf import hf, orbital_rotation

BASIS = 'def2-svp'
STEP_NORMS = (1e-1, 1e-2, 1e-3, 1e-4)
MAX_KAPPA0 = 2.0e-2   # max|kappa0| of the rotation generating the step state
KAPPA0_SEED = 2026    # fixed seed, the step state is reproducible
FD_STEP = 1e-3        # step of the central difference of appendix B.1
ORTH_TOL = 1e-10
REF_TOL = 1e-10
RATIO_TOL = 3.0e-2

# Idealized geometries: r(N-H) = 1.0124, angle(HNH) = 106.7 for NH3;
# r(C-C) = 1.529, r(C-H) = 1.091, staggered for C2H6; r(N-N) = 1.401,
# r(N-C) = 1.451, r(N-H) = 1.021, r(C-H) = 1.091, angle(CNC) = 112 for UDMH.
MOLECULES = (
    ('H2O', '''
        O      0.0000000000     0.0000000000     0.1174000000
        H      0.7570000000     0.0000000000    -0.4696000000
        H     -0.7570000000     0.0000000000    -0.4696000000
    '''),
    ('NH3', '''
        N      0.0000000000     0.0000000000     0.0000000000
        H      0.9380180000     0.0000000000    -0.3808890000
        H     -0.4690090000     0.8123470000    -0.3808890000
        H     -0.4690090000    -0.8123470000    -0.3808890000
    '''),
    ('C2H6', '''
        C      0.0000000000     0.0000000000     0.0000000000
        C      0.0000000000     0.0000000000     1.5290000000
        H      1.0171650000     0.0000000000    -0.3945320000
        H     -0.5085830000     0.8808910000    -0.3945320000
        H     -0.5085830000    -0.8808910000    -0.3945320000
        H      0.5085830000     0.8808910000     1.9235320000
        H     -1.0171650000     0.0000000000     1.9235320000
        H      0.5085830000    -0.8808910000     1.9235320000
    '''),
    ('(CH3)2NNH2', '''
        N      0.0000000000     0.0000000000     0.0000000000
        N      0.0000000000     0.0000000000     1.4010000000
        C      0.6024120000     1.2029340000    -0.5435540000
        C      0.6024120000    -1.2029340000    -0.5435540000
        H     -0.1663420000     1.9649970000    -0.6797780000
        H      1.0638280000     0.9757370000    -1.5057170000
        H      1.3626750000     1.5724920000     0.1461620000
        H      1.0638280000    -0.9757370000    -1.5057170000
        H     -0.1663420000    -1.9649970000    -0.6797780000
        H      1.3626750000    -1.5724920000     0.1461620000
        H      0.4717460000     0.8207380000     1.7834730000
        H      0.4717460000    -0.8207380000     1.7834730000
    '''),
)


def build_rotation_generator(k, n_occ):
    '''CPU reference generator K = [[0, -kappa.T], [kappa, 0]] built from k.

    kappa = k.reshape(n_vir, n_occ) is the occupied-virtual block, so that
    kappa[a, i] mixes occupied orbital i into the virtual orbital a.
    '''
    k = np.asarray(k, dtype=np.float64)
    n_vir = k.size // n_occ
    K = np.zeros((n_occ + n_vir, n_occ + n_vir))
    K[n_occ:, :n_occ] = k.reshape(n_vir, n_occ)
    K[:n_occ, n_occ:] = -K[n_occ:, :n_occ].T
    return K


def rhf_state(mf, C, mo_occ):
    '''Total RHF energy and MO-basis Fock of the state with coefficients C.

    The energy is the exact RHF functional of the density built from the
    occupied columns of C, evaluated with the same (density fitted) mean field
    object that produced the converged solution.
    '''
    dm = mf.make_rdm1(C, mo_occ)
    fock = mf.get_fock(dm=dm)
    return mf.energy_tot(dm=dm), C.T.dot(fock).dot(C)


def validate_molecule(name, atom):
    mol = pyscf.M(atom=atom, basis=BASIS, verbose=0)
    mf = hf.RHF(mol).density_fit()
    e_scf = mf.kernel()
    assert mf.converged, f'{name}: GPU RHF did not converge'

    C_scf = mf.mo_coeff
    mo_occ = mf.mo_occ
    nao, nmo = C_scf.shape
    n_occ = mol.nelectron // 2
    n_vir = nmo - n_occ
    S = cp.asarray(mol.intor('int1e_ovlp'))

    # residual gradient of the converged solution, it must vanish
    F_scf_mo = C_scf.T.dot(mf.get_fock()).dot(C_scf)
    g_scf = float(cp.linalg.norm(4*F_scf_mo[n_occ:, :n_occ]))

    # non-stationary step state C0 = C_scf @ expm(K0), see the module docstring
    rng = np.random.default_rng(KAPPA0_SEED)
    kappa0 = rng.standard_normal((n_vir, n_occ))
    kappa0 *= MAX_KAPPA0/np.abs(kappa0).max()
    C0_np = cp.asnumpy(C_scf).dot(scipy.linalg.expm(build_rotation_generator(kappa0.reshape(-1), n_occ)))
    C0 = cp.asarray(C0_np)

    e0, F_mo = rhf_state(mf, C0, mo_occ)
    g = 4*F_mo[n_occ:, :n_occ]                  # gradient g_ai = 4 F_ai
    e = cp.diag(F_mo)                           # orbital energies, F_aa and F_ii in the MO basis
    d_e = e[n_occ:, None] - e[:n_occ][None, :]  # e_a - e_i
    gap = float(d_e.min())
    assert gap > 0, f'{name}: the occupied/virtual ordering is not energy ordered'

    # preconditioned step kappa[a, i] = -g_ai/(4 (e_a - e_i)), rescaled below
    kappa = -g/(4*d_e)
    v_norm = float(cp.linalg.norm(kappa))
    x = (kappa/v_norm).reshape(-1)              # unit step direction, vir major
    g_norm = float(cp.linalg.norm(g))

    print(f'{"="*118}')
    print(f' {name} / {BASIS}    nao = {nao}, nocc = {n_occ}, nvir = {n_vir}')
    print(f'   GPU RHF (density fitting) converged: E_scf = {e_scf:.10f} Ha, residual |4 F_ai| = {g_scf:.2e}')
    print(f'   step state C0 = C_scf expm(K0), max|kappa0| = {MAX_KAPPA0:.1e} (seed {KAPPA0_SEED}, '
          f'non-stationary as required by appendix B)')
    print(f'   E0 = {e0:.10f} Ha (E0 - E_scf = {e0 - e_scf:+.2e}), |4 F_ai| = {g_norm:.2e}, '
          f'|kappa| of the raw step = {v_norm:.2e}, min(e_a - e_i) = {gap:.3f} Ha')
    print(f'{"|k|":>9} {"orth(expm)":>12} {"orth(taylor2)":>14} {"|Cgpu-Cscipy|":>14} '
          f'{"|Ctaylor-Cgpu|":>15} {"dE (Ha)":>13} {"g.k (Ha)":>13} {"dE/(g.k)":>10} {"dE<0":>6}')
    print(f'{"-"*118}')

    rows = []
    for norm in STEP_NORMS:
        k = cp.asarray((norm*x).astype(np.float64))
        C_expm = orbital_rotation.update_mo_coeff(C0, k, n_occ)
        C_taylor = orbital_rotation.update_mo_coeff(C0, k, n_occ, method='taylor', order=2)

        err_expm = orbital_rotation.mo_orthonormality_error(C_expm, S)
        err_taylor = orbital_rotation.mo_orthonormality_error(C_taylor, S)

        # CPU reference: identical generator K, rotation by scipy.linalg.expm
        K_ref = build_rotation_generator(cp.asnumpy(k), n_occ)
        C_ref = C0_np.dot(scipy.linalg.expm(K_ref))
        ref_diff = np.abs(cp.asnumpy(C_expm) - C_ref).max()
        taylor_diff = np.abs(cp.asnumpy(C_taylor) - cp.asnumpy(C_expm)).max()

        # linear energy model, the true energy is the RHF functional of C'
        e_new = rhf_state(mf, C_expm, mo_occ)[0]
        d_e_tot = e_new - e0
        g_dot_k = float(cp.dot(g.reshape(-1), k))
        ratio = d_e_tot/g_dot_k

        rows.append((norm, err_expm, err_taylor, ref_diff, taylor_diff, d_e_tot, g_dot_k, ratio))
        # (norm, orthonormality error of both updates, CPU reference diff,
        #  taylor minus expm, dE, g.k, dE/(g.k))
        print(f'{norm:9.1e} {err_expm:12.2e} {err_taylor:14.2e} {ref_diff:14.2e} '
              f'{taylor_diff:15.2e} {d_e_tot:+13.6e} {g_dot_k:+13.6e} {ratio:10.5f} '
              f'{"yes" if d_e_tot < 0 else "no":>6}')

    # appendix B.1: the central difference of E(t) = E[C0 expm(t K_x)] along the
    # step direction x is the directional derivative g.x of the linear model
    x_np = cp.asnumpy(x)
    K_x = build_rotation_generator(FD_STEP*x_np, n_occ)
    e_plus = rhf_state(mf, cp.asarray(C0_np.dot(scipy.linalg.expm(K_x))), mo_occ)[0]
    e_minus = rhf_state(mf, cp.asarray(C0_np.dot(scipy.linalg.expm(-K_x))), mo_occ)[0]
    fd = (e_plus - e_minus)/(2*FD_STEP)
    g_dot_x = float(cp.dot(g.reshape(-1), x))
    fd_dev = (fd - g_dot_x)/g_dot_x
    print(f'   central difference (appendix B.1) at h = {FD_STEP:.0e}: {fd:+.8e} vs g.x = {g_dot_x:+.8e}, '
          f'relative deviation {fd_dev:+.2e}')

    return {'name': name, 'g_scf': g_scf, 'fd_dev': abs(fd_dev), 'rows': rows}


print('='*118)
print(' Validation of the orbital rotation update k -> C')
print('   molecules: H2O, NH3, C2H6, (CH3)2NNH2 (UDMH), basis def2-SVP, GPU RHF with density fitting')
print('   gradient g = vec(4 F_ai), step kappa[a, i] = -g_ai/(4 (e_a - e_i)), update C_new = C @ expm(K)')
print()

results = [validate_molecule(name, atom) for name, atom in MOLECULES]

# ----------------------------------------------------------------------------
# summary against the acceptance thresholds
# ----------------------------------------------------------------------------
max_orth = max(row[1] for res in results for row in res['rows'])
max_ref = max(row[3] for res in results for row in res['rows'])
max_ratio_dev = max(abs(row[7] - 1) for res in results for row in res['rows']
                    if row[0] == STEP_NORMS[-1])
max_fd_dev = max(res['fd_dev'] for res in results)

def energies_decrease(norm):
    '''True when the step along -g lowers the energy at this norm in every molecule.'''
    return all(row[5] < 0 for res in results for row in res['rows'] if row[0] == norm)

print(f'{"="*118}')
print(' summary over the four molecules')
print(f'   {"largest residual |4 F_ai| at convergence":<49}: {max(res["g_scf"] for res in results):.2e}')
print(f'   {"max orthonormality error of the expm update":<49}: {max_orth:.2e}  (threshold {ORTH_TOL:.0e})')
print(f'   {"max |C_gpu - C_scipy| against the CPU reference":<49}: {max_ref:.2e}  '
      f'(threshold {REF_TOL:.0e})')
print(f'   {f"max |ratio - 1| at |k| = {STEP_NORMS[-1]:.0e}":<49}: {max_ratio_dev:.2e}  '
      f'(threshold {RATIO_TOL:.0e})')
print(f'   {"max relative deviation of the central difference":<49}: {max_fd_dev:.2e}')
print(f'   the step along -g lowers the energy for |k| <= '
      f'{max(norm for norm in STEP_NORMS if energies_decrease(norm)):.0e} in every molecule')
print()

assert max_orth < ORTH_TOL, f'expm update lost orthonormality, max error {max_orth}'
assert max_ref < REF_TOL, f'GPU update deviates from the CPU reference by {max_ref}'
assert max_ratio_dev < RATIO_TOL, f'the linear energy model is off by {max_ratio_dev} at |k| = {STEP_NORMS[-1]:.0e}'
for norm in (1e-2, 1e-3, 1e-4):
    assert energies_decrease(norm), f'the step along -g raises the energy at |k| = {norm:.0e}'
print(' all checks passed')
