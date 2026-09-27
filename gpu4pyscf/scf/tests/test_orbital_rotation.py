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

import unittest
import numpy as np
import scipy.linalg
import cupy as cp
from gpu4pyscf.scf import orbital_rotation

def random_orthonormal_mo(nao, nmo, seed, overlap=None):
    '''Random (nao, nmo) MO coefficients with C.T S C = I.

    S is the identity matrix when overlap is None, otherwise the given
    positive definite AO overlap matrix.
    '''
    rng = np.random.default_rng(seed)
    C = np.linalg.qr(rng.standard_normal((nao, nmo)))[0]
    if overlap is not None:
        C = scipy.linalg.solve_triangular(np.linalg.cholesky(overlap).T, C)
    return C

class TestOrbitalRotation(unittest.TestCase):
    # (a) the generator K is antisymmetric with the expected blocks
    def test_generator_blocks(self):
        for n_occ, n_vir in ((1, 1), (3, 5), (4, 2)):
            with self.subTest(n_occ=n_occ, n_vir=n_vir):
                nmo = n_occ + n_vir
                k = cp.asarray(np.random.default_rng(10*n_vir + n_occ).standard_normal(n_vir*n_occ))
                K = orbital_rotation.build_rotation_generator(k, n_occ)
                self.assertEqual(K.shape, (nmo, nmo))
                self.assertEqual(K.dtype, np.float64)

                K_np = cp.asnumpy(K)
                kappa = cp.asnumpy(orbital_rotation.k_to_kappa(k, n_occ))
                np.testing.assert_allclose(K_np[n_occ:, :n_occ], kappa, rtol=0, atol=1e-15)
                np.testing.assert_allclose(K_np[:n_occ, n_occ:], -kappa.T, rtol=0, atol=1e-15)
                np.testing.assert_allclose(K_np[:n_occ, :n_occ], 0, rtol=0, atol=1e-15)
                np.testing.assert_allclose(K_np[n_occ:, n_occ:], 0, rtol=0, atol=1e-15)
                np.testing.assert_allclose(K_np, -K_np.T, rtol=0, atol=1e-15)
                np.testing.assert_allclose(np.diagonal(K_np), 0, rtol=0, atol=1e-15)

    # (a) the generator agrees with the masked construction used by the CIAH
    # solver in gpu4pyscf/scf/soscf.py, dr = x1 - x1.T with x1[vir, occ] = k
    def test_matches_ciah_generator_convention(self):
        n_occ, n_vir = 5, 7
        nmo = n_occ + n_vir
        k = cp.asarray(np.random.default_rng(14).standard_normal(n_vir*n_occ))
        mo_occ = cp.asarray([2.]*n_occ + [0.]*n_vir)
        occidxa = mo_occ > 0
        viridxa = ~occidxa
        x1 = cp.zeros((nmo, nmo), dtype=np.float64)
        x1[viridxa[:, None] & occidxa] = k
        np.testing.assert_array_equal(cp.asnumpy(orbital_rotation.build_rotation_generator(k, n_occ)),
                                      cp.asnumpy(x1 - x1.conj().T))

    # (a) + eqs. 3.11-3.13: a single kappa[a, i] = theta rotates virtual a into
    # occupied i, phi_i' = cos(theta) phi_i + sin(theta) phi_a
    def test_single_rotation_angle(self):
        nao, n_occ, n_vir = 8, 3, 4
        nmo = n_occ + n_vir
        C_np = random_orthonormal_mo(nao, nmo, seed=1)
        C = cp.asarray(C_np)
        for i, a in ((0, 0), (2, 3), (1, 2)):
            for theta in (1e-1, 1e-3):
                with self.subTest(occ=i, vir=a, theta=theta):
                    k = cp.zeros(n_vir*n_occ)
                    k[a*n_occ + i] = theta
                    U_ref = np.eye(nmo)
                    U_ref[i, i] = np.cos(theta)
                    U_ref[i, n_occ+a] = -np.sin(theta)
                    U_ref[n_occ+a, i] = np.sin(theta)
                    U_ref[n_occ+a, n_occ+a] = np.cos(theta)
                    np.testing.assert_allclose(cp.asnumpy(orbital_rotation.rotation_matrix(k, n_occ)),
                                               U_ref, rtol=0, atol=1e-12)
                    np.testing.assert_allclose(cp.asnumpy(orbital_rotation.update_mo_coeff(C, k, n_occ)),
                                               C_np.dot(U_ref), rtol=0, atol=1e-12)

    # (e) reshape round trip k -> kappa -> k
    def test_kappa_round_trip(self):
        for n_occ, n_vir in ((1, 1), (3, 5), (4, 2), (2, 7)):
            with self.subTest(n_occ=n_occ, n_vir=n_vir):
                k_np = np.random.default_rng(100*n_occ + n_vir).standard_normal(n_vir*n_occ)
                k = cp.asarray(k_np)
                kappa = orbital_rotation.k_to_kappa(k, n_occ)
                self.assertEqual(kappa.shape, (n_vir, n_occ))
                self.assertIsInstance(kappa, cp.ndarray)
                np.testing.assert_array_equal(cp.asnumpy(kappa), k_np.reshape(n_vir, n_occ))
                np.testing.assert_array_equal(cp.asnumpy(orbital_rotation.kappa_to_k(kappa)), k_np)
                # kappa is the occupied-virtual block of the generator
                K = cp.asnumpy(orbital_rotation.build_rotation_generator(k, n_occ))
                np.testing.assert_array_equal(K[n_occ:, :n_occ], cp.asnumpy(kappa))
                np.testing.assert_array_equal(K[:n_occ, n_occ:], -cp.asnumpy(kappa).T)

    # (b) the expm update preserves orthonormality, with S = I and with a
    # nontrivial positive definite overlap
    def test_expm_update_preserves_orthonormality(self):
        nao, n_occ, n_vir = 14, 4, 10
        nmo = n_occ + n_vir
        rng = np.random.default_rng(2)
        kappa = rng.standard_normal((n_vir, n_occ))
        kappa /= np.linalg.norm(kappa)

        overlap = np.random.default_rng(3).standard_normal((nao, nao))
        overlap = overlap.dot(overlap.T) + nao*np.eye(nao)
        for S in (None, cp.asarray(overlap)):
            with self.subTest(overlap=S is not None):
                S_np = np.eye(nao) if S is None else overlap
                C = cp.asarray(random_orthonormal_mo(nao, nmo, seed=4, overlap=S_np))
                self.assertLess(orbital_rotation.mo_orthonormality_error(C, None if S is None else S), 1e-12)
                for norm in (1e-1, 1e-2, 1e-3, 1e-4, 1e-6):
                    k = cp.asarray((norm*kappa).reshape(-1))
                    C_new = orbital_rotation.update_mo_coeff(C, k, n_occ)
                    error = orbital_rotation.mo_orthonormality_error(C_new, None if S is None else S)
                    self.assertLess(error, 1e-12, f'orthonormality error {error} at |k| = {norm}')
                    self.assertEqual(C_new.shape, C.shape)
                    self.assertIsInstance(C_new, cp.ndarray)

    # (c) the order 2 Taylor update reproduces eq. 8.7 and differs from the
    # expm update by O(|k|^3)
    def test_taylor_order2(self):
        nao, n_occ, n_vir = 14, 4, 10
        nmo = n_occ + n_vir
        C = cp.asarray(random_orthonormal_mo(nao, nmo, seed=5))
        C_np = cp.asnumpy(C)
        kappa = np.random.default_rng(6).standard_normal((n_vir, n_occ))
        kappa /= np.linalg.norm(kappa)

        errors = {}
        for norm in (1e-1, 1e-2, 1e-3, 1e-4, 1e-6):
            with self.subTest(norm=norm):
                k = cp.asarray((norm*kappa).reshape(-1))
                C_expm = orbital_rotation.update_mo_coeff(C, k, n_occ)
                C_taylor = orbital_rotation.update_mo_coeff(C, k, n_occ, method='taylor', order=2)
                # eq. 8.7: Co' = Co + Cv @ kappa - Co @ kappa.T @ kappa / 2
                expected = C_np.copy()
                expected[:, :n_occ] = (C_np[:, :n_occ] + C_np[:, n_occ:].dot(norm*kappa)
                                       - 0.5*C_np[:, :n_occ].dot((norm*kappa).T.dot(norm*kappa)))
                np.testing.assert_allclose(cp.asnumpy(C_taylor)[:, :n_occ], expected[:, :n_occ],
                                           rtol=0, atol=1e-15)
                errors[norm] = cp.asnumpy(cp.abs(C_taylor - C_expm)).max()

        self.assertTrue(errors[1e-1] > errors[1e-2] > errors[1e-3] > errors[1e-4])
        for large, small in ((1e-2, 1e-3), (1e-3, 1e-4)):
            ratio = errors[large]/errors[small]
            self.assertTrue(300 < ratio < 3000,
                            f'|k|^3 scaling expected, got a ratio of {ratio} for {large} -> {small}')

    # (c) the order 1 Taylor update is the first order formula of eq. 8.7
    def test_taylor_order1(self):
        nao, n_occ, n_vir = 14, 4, 10
        nmo = n_occ + n_vir
        C = cp.asarray(random_orthonormal_mo(nao, nmo, seed=7))
        C_np = cp.asnumpy(C)
        kappa = np.random.default_rng(8).standard_normal((n_vir, n_occ))
        kappa /= np.linalg.norm(kappa)

        errors = {}
        for norm in (1e-1, 1e-2, 1e-3, 1e-4):
            with self.subTest(norm=norm):
                k = cp.asarray((norm*kappa).reshape(-1))
                C_taylor = orbital_rotation.update_mo_coeff(C, k, n_occ, method='taylor', order=1)
                # Co' = Co + Cv @ kappa (up to first order), Cv' = Cv - Co @ kappa.T
                expected = C_np.copy()
                expected[:, :n_occ] = C_np[:, :n_occ] + C_np[:, n_occ:].dot(norm*kappa)
                expected[:, n_occ:] = C_np[:, n_occ:] - C_np[:, :n_occ].dot((norm*kappa).T)
                np.testing.assert_allclose(cp.asnumpy(C_taylor), expected, rtol=0, atol=1e-15)
                errors[norm] = cp.asnumpy(cp.abs(C_taylor - orbital_rotation.update_mo_coeff(C, k, n_occ))).max()

        for large, small in ((1e-2, 1e-3), (1e-3, 1e-4)):
            ratio = errors[large]/errors[small]
            self.assertTrue(30 < ratio < 300,
                            f'|k|^2 scaling expected, got a ratio of {ratio} for {large} -> {small}')

    # (d) the expm update agrees with a scipy.linalg.expm reference on the CPU
    def test_expm_matches_cpu_reference(self):
        nao, n_occ, n_vir = 13, 3, 8
        nmo = n_occ + n_vir
        C_np = random_orthonormal_mo(nao, nmo, seed=9)
        C = cp.asarray(C_np)
        kappa = np.random.default_rng(10).standard_normal((n_vir, n_occ))
        for norm in (1e-1, 1e-3, 1e-6):
            with self.subTest(norm=norm):
                k_np = (norm*kappa).reshape(-1)
                k = cp.asarray(k_np)
                K_ref = np.zeros((nmo, nmo))
                K_ref[n_occ:, :n_occ] = k_np.reshape(n_vir, n_occ)
                K_ref[:n_occ, n_occ:] = -K_ref[n_occ:, :n_occ].T
                np.testing.assert_allclose(cp.asnumpy(orbital_rotation.build_rotation_generator(k, n_occ)),
                                           K_ref, rtol=0, atol=1e-15)
                U_ref = scipy.linalg.expm(K_ref)
                np.testing.assert_allclose(cp.asnumpy(orbital_rotation.rotation_matrix(k, n_occ)),
                                           U_ref, rtol=0, atol=1e-12)
                np.testing.assert_allclose(cp.asnumpy(orbital_rotation.update_mo_coeff(C, k, n_occ)),
                                           C_np.dot(U_ref), rtol=0, atol=1e-12)

    # numpy inputs are transferred to the GPU, S=None means the identity overlap
    def test_numpy_inputs(self):
        nao, n_occ, n_vir = 9, 2, 5
        nmo = n_occ + n_vir
        C = random_orthonormal_mo(nao, nmo, seed=11)
        k = np.random.default_rng(12).standard_normal(n_vir*n_occ)
        for method, order in (('expm', 2), ('taylor', 1), ('taylor', 2)):
            with self.subTest(method=method, order=order):
                C_new = orbital_rotation.update_mo_coeff(C, k, n_occ, method=method, order=order)
                self.assertIsInstance(C_new, cp.ndarray)
                np.testing.assert_allclose(cp.asnumpy(C_new),
                                           cp.asnumpy(orbital_rotation.update_mo_coeff(cp.asarray(C), cp.asarray(k),
                                                                                       n_occ, method=method,
                                                                                       order=order)),
                                           rtol=0, atol=0)
        self.assertLess(orbital_rotation.mo_orthonormality_error(cp.asarray(C), None), 1e-12)
        self.assertLess(orbital_rotation.mo_orthonormality_error(C, cp.eye(nao)), 1e-12)

    # (f) invalid inputs raise
    def test_invalid_inputs(self):
        nao, n_occ, n_vir = 8, 2, 3
        nmo = n_occ + n_vir
        C = cp.asarray(random_orthonormal_mo(nao, nmo, seed=13))
        k = cp.zeros(n_vir*n_occ)

        with self.assertRaises(ValueError):  # wrong k length
            orbital_rotation.update_mo_coeff(C, cp.zeros(n_vir*n_occ + 1), n_occ)
        with self.assertRaises(ValueError):  # k length not divisible by n_occ
            orbital_rotation.build_rotation_generator(cp.zeros(n_vir*n_occ - 1), n_occ)
        with self.assertRaises(ValueError):  # k must be 1D
            orbital_rotation.build_rotation_generator(cp.zeros((n_vir, n_occ)), n_occ)
        with self.assertRaises(ValueError):  # empty k
            orbital_rotation.build_rotation_generator(cp.zeros(0), n_occ)
        with self.assertRaises(AssertionError):  # k must be float64
            orbital_rotation.update_mo_coeff(C, cp.zeros(n_vir*n_occ, dtype=cp.float32), n_occ)
        with self.assertRaises(AssertionError):  # C must be float64
            orbital_rotation.update_mo_coeff(C.astype(cp.float32), k, n_occ)
        with self.assertRaises(ValueError):  # C width mismatch
            orbital_rotation.update_mo_coeff(C[:, :-1], k, n_occ)
        with self.assertRaises(ValueError):  # C must be 2D
            orbital_rotation.update_mo_coeff(C.reshape(-1), k, n_occ)
        with self.assertRaises(ValueError):  # unknown method
            orbital_rotation.update_mo_coeff(C, k, n_occ, method='matrix_exponential')
        with self.assertRaises(ValueError):  # unknown order
            orbital_rotation.update_mo_coeff(C, k, n_occ, method='taylor', order=3)
        with self.assertRaises(TypeError):  # n_occ must be an integer
            orbital_rotation.update_mo_coeff(C, k, 2.0)
        with self.assertRaises(ValueError):  # n_occ must be positive
            orbital_rotation.build_rotation_generator(k, 0)
        with self.assertRaises(ValueError):  # S must match the AO dimension of C
            orbital_rotation.mo_orthonormality_error(C, cp.eye(nao + 1))
        with self.assertRaises(ValueError):  # S must be square
            orbital_rotation.mo_orthonormality_error(C, cp.ones((nao, nao + 1)))
        with self.assertRaises(AssertionError):  # S must be float64
            orbital_rotation.mo_orthonormality_error(C, cp.eye(nao, dtype=cp.float32))

if __name__ == '__main__':
    print("Full Tests for the orbital rotation k -> C update")
    unittest.main()
