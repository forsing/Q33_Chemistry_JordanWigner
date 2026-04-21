#!/usr/bin/env python3

"""
Q33 v1 Quantum Chemistry — Jordan-Wigner fermionski Hamiltonian i Slater determinant
ground-state preparation preko Givens rotacione mreže (čisto kvantno: JW mapiranje
fermionskih moda u qubit-e + orbit-rotacija Thouless teoremom + Slater determinant
u N-particle sektoru, BEZ klasičnog ML-a, BEZ hibrida).

Koncept:
  Loto predikcija je prirodno kombinatorni izbor 7 brojeva iz 39 → matematička
  struktura je IDENTIČNA fiksnom-broju-čestica sektoru fermionskog mnogočestičnog
  sistema. Svaki broj 1..39 ↦ fermionski mod (orbital); stanje sa 7 okupiranih
  orbitala ↔ valid 7-kombinacija; Pauli-princip automatski isključuje duple brojeve.

  Jordan-Wigner mapiranje prebacuje fermionske operatore u Pauli operatore na qubit-ovima:
        c_p†   ↦   (1/2)(X_p - iY_p) · Z_{p-1} Z_{p-2} ... Z_0
        c_p    ↦   (1/2)(X_p + iY_p) · Z_{p-1} Z_{p-2} ... Z_0
        n_p = c_p† c_p  ↦  (I - Z_p) / 2    (Z_p je Pauli-Z na p-tom qubit-u)
  (JW Z-string očuvava fermionske antisimetrijske statistike pri 1D-linearnom
  uređenju orbitala.)

  Efektivni jedno-čestični Hamiltonian iz CSV-a (K x K real simetrična matrica u
  active-space):
        h_pp  = -freq_num[p] / <freq>                   ← orbital energies (niža
                                                           energija = veća verovatnoća
                                                           okupacije u ground state-u)
        h_pq  = -t_hop · P_num[p,q] / <P>               ← hopping (pair co-occurrence)
        h_pp += Σ_q (-u_int · P_num[p,q] / <P>) · n̄    ← mean-field density-density
                                                           (n̄ = N_PART / K)
  Diagonalizacija h = V D Vᵀ daje eigen-orbitale; N najnižih je OKUPIRANO u
  Slater determinant ground state-u.

  Thouless teorema — Slater determinant preko orbitalne rotacije:
        |ψ_SD⟩ = Πₖ c̃_k† |vac⟩,       c̃_k† = Σ_p V_pk c_p†
  Orbitalna rotacija se implementira mrežom adjacent Givens rotacija (Reck 1994,
  Kivlichan et al. 2018):
        V = Πᵢ Gᵢ(θᵢ)                 (linearna dubina, ~K²/2 dvo-qubit gate-ova)
  Gᵢ(θ) na qubit-ovima (i, i+1) je number-preserving 2-qubit unitar:
        |00⟩ → |00⟩,   |11⟩ → |11⟩,
        |01⟩ →  cos(θ)|01⟩ + sin(θ)|10⟩,
        |10⟩ → -sin(θ)|01⟩ + cos(θ)|10⟩.
  (Jordan-Wigner: za adjacent orbitale Z-string je prazan pa je Gᵢ čist 2-qubit gate.)

Kolo (K active qubit-a, BEZ ancilla, BEZ phase-registra):
  1) X na prvih N qubit-a → inicijalno |1^N 0^(K-N)⟩ = Slater determinant u
     "prirodnoj" (freq-rangiranoj) bazi (N_PART = 7, K = K_ACTIVE).
  2) Z na qubit-ima gde diag(±1) posle Reck-trianguralizacije daje -1 (global phase
     korekcija).
  3) Mreža adjacent Givens rotacija (reverznim redosledom, sa negiranim uglovima)
     koja implementira orbit-rotaciju V (eigenvektore od h).

Readout:
  Statevector → za svako I ⊂ active sa |I| = N, amplituda = det(V[rows=I, cols=0..N-1])
  (Slater determinant property, Thouless teorema).
  Occupation per orbital: P(n_p = 1) = Σ_{I: p∈I} |amp(I)|² — verovatnoća da je
  active-orbital p okupiran u ground state-u.
  Top-N_PART orbitala po okupaciji → mapiranje nazad na originalne brojeve 1..39.

Active space:
  K_ACTIVE top-freq brojeva iz CSV-a (active-space aproksimacija — dovoljna jer
  nisko-freq brojevi su suppressed u ground-state-u). Grid optimizuje K_ACTIVE,
  t_hop, u_int po cos(bias_39, freq_csv).

Qubit budget: K_ACTIVE (bez ancilla, bez phase). K_ACTIVE = 14 → 2^14 = 16384 stanja,
C(14, 7) = 3432 stanja u 7-particle sektoru. Svi gate-ovi su number-preserving.

Sve deterministički: seed=39; h izveden iz CELOG CSV-a (pravilo 10).
Deterministička grid-optimizacija (K_ACTIVE, t_hop, u_int) po cos(bias_39, freq_csv).

Okruženje: Python 3.11.13, qiskit 1.4.4, qiskit-machine-learning 0.8.3, macOS M1 (vidi README.md).

Napomena o verzijama:
  v1 (ovaj fajl, Q33_Chemistry_JordanWigner_v1.py): originalni many-body JW pristup
    na K_ACTIVE = 14 qubit-a (active-space top-K freq brojeva).
  v2 (Q33_Chemistry_JordanWigner_v2.py): (min broj qubit-a) —
    per-pozicijski single-particle ground state na 6 qubit-a po poziciji, opsezi
    Num_i ∈ [i, i+32], autoregresivni loop kroz 7 pozicija, strogo uređeno
    Num_1 < Num_2 < ... < Num_7 by construction. Full range [1, 39] pokriven.
"""



from __future__ import annotations

import csv
import random
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from scipy.sparse import SparseEfficiencyWarning

    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
except ImportError:
    pass

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector

# =========================
# Seed
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
try:
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = SEED
except ImportError:
    pass

# =========================
# Konfiguracija
# =========================
CSV_PATH = Path("/data/loto7hh_4600_k31.csv")
N_NUMBERS = 7
N_MAX = 39

N_PART = N_NUMBERS
GRID_K_ACTIVE = (12, 14)
GRID_T_HOP = (0.25, 0.5, 1.0)
GRID_U_INT = (0.0, 0.2, 0.5)


# =========================
# CSV
# =========================
def load_rows(path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        if not header or "Num1" not in header[0]:
            f.seek(0)
            r = csv.reader(f)
            next(r, None)
        for row in r:
            if not row or row[0].strip() == "Num1":
                continue
            rows.append([int(row[i]) for i in range(N_NUMBERS)])
    return np.array(rows, dtype=int)


def freq_vector(H: np.ndarray) -> np.ndarray:
    c = np.zeros(N_MAX, dtype=np.float64)
    for v in H.ravel():
        if 1 <= v <= N_MAX:
            c[int(v) - 1] += 1.0
    return c


def pair_matrix(H: np.ndarray) -> np.ndarray:
    P = np.zeros((N_MAX, N_MAX), dtype=np.float64)
    for row in H:
        for a in row:
            for b in row:
                if a != b and 1 <= a <= N_MAX and 1 <= b <= N_MAX:
                    P[a - 1, b - 1] += 1.0
    return P


# =========================
# Active space + efektivna jedno-čestična Hamiltonijeva (K × K real simetrična)
# =========================
def build_active_space(H: np.ndarray, K: int) -> List[int]:
    freq = freq_vector(H)
    order = np.argsort(-freq, kind="stable")
    return [int(order[i]) + 1 for i in range(K)]


def build_h_effective(
    H: np.ndarray, active_to_num: List[int], t_hop: float, u_int: float
) -> np.ndarray:
    K = len(active_to_num)
    freq = freq_vector(H)
    P = pair_matrix(H)
    f_mean = float(freq.mean()) + 1e-18
    P_mean = float(P.mean()) + 1e-18

    h = np.zeros((K, K), dtype=np.float64)

    for p in range(K):
        n_p = active_to_num[p] - 1
        h[p, p] = -float(freq[n_p]) / f_mean

    for p in range(K):
        for q in range(K):
            if p == q:
                continue
            n_p = active_to_num[p] - 1
            n_q = active_to_num[q] - 1
            h[p, q] = -float(t_hop) * float(P[n_p, n_q]) / P_mean

    n_bar = float(N_PART) / float(K)
    for p in range(K):
        acc = 0.0
        n_p = active_to_num[p] - 1
        for q in range(K):
            if q == p:
                continue
            n_q = active_to_num[q] - 1
            u_pq = -float(u_int) * float(P[n_p, n_q]) / P_mean
            acc += u_pq * n_bar
        h[p, p] += acc

    h = 0.5 * (h + h.T)
    return h


# =========================
# Reck decomposition: real orthogonal V → adjacent Givens na redovima (i, i+1)
# Cilj: G_L · ... · G_1 · V = diag(±1)
# =========================
def reck_decompose(V: np.ndarray) -> Tuple[List[Tuple[int, float]], np.ndarray]:
    K = V.shape[0]
    U_work = V.astype(np.float64).copy()
    givens: List[Tuple[int, float]] = []

    for col in range(K - 1):
        for row in range(K - 1, col, -1):
            a = U_work[row - 1, col]
            b = U_work[row, col]
            r = float(np.hypot(a, b))
            if r < 1e-14:
                continue
            c = float(a) / r
            s = float(b) / r
            theta = float(np.arctan2(s, c))
            r1 = U_work[row - 1, :].copy()
            r2 = U_work[row, :].copy()
            U_work[row - 1, :] = c * r1 + s * r2
            U_work[row, :] = -s * r1 + c * r2
            givens.append((int(row - 1), float(theta)))

    diag_signs = np.sign(np.diag(U_work))
    diag_signs = np.where(diag_signs == 0.0, 1.0, diag_signs)
    return givens, diag_signs


# =========================
# Adjacent Givens gate (number-preserving 4x4 unitar) — qiskit little-endian
# =========================
def givens_unitary_matrix(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    U = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0,   c,  -s, 0.0],
            [0.0,   s,   c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.complex128,
    )
    return U


# =========================
# Slater determinant kolo (Thouless teorema)
# =========================
def build_slater_circuit(
    K: int, N: int, givens: List[Tuple[int, float]], diag_signs: np.ndarray
) -> QuantumCircuit:
    q_reg = QuantumRegister(K, name="o")
    qc = QuantumCircuit(q_reg)

    for i in range(N):
        qc.x(q_reg[i])

    for i in range(K):
        if float(diag_signs[i]) < 0.0 and i < N:
            qc.z(q_reg[i])

    for (i, theta) in reversed(givens):
        U = givens_unitary_matrix(-theta)
        gate = UnitaryGate(U, label=f"G({-theta:+.3f})")
        qc.append(gate, [q_reg[i], q_reg[i + 1]])

    return qc


def slater_state_probs(sv_data: np.ndarray, K: int) -> np.ndarray:
    return np.abs(sv_data) ** 2


def occupation_from_sv(probs: np.ndarray, K: int, N: int) -> Tuple[np.ndarray, float]:
    occ = np.zeros(K, dtype=np.float64)
    total_n = 0.0
    dim = 2 ** K
    for idx in range(dim):
        if int(bin(idx).count("1")) != N:
            continue
        p = float(probs[idx])
        total_n += p
        for k in range(K):
            if (idx >> k) & 1:
                occ[k] += p
    return occ, total_n


# =========================
# Readout → bias_39 (projektuje active-space okupacije na full 1..39 brojeve)
# =========================
def bias_from_occupation(occ: np.ndarray, active_to_num: List[int]) -> np.ndarray:
    b = np.zeros(N_MAX, dtype=np.float64)
    for k, n in enumerate(active_to_num):
        idx = int(n) - 1
        if 0 <= idx < N_MAX:
            b[idx] = float(occ[k])
    s = float(b.sum())
    return b / s if s > 0 else b


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-18 or nb < 1e-18:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pick_next_combination(
    occ: np.ndarray, active_to_num: List[int], k: int = N_NUMBERS
) -> Tuple[int, ...]:
    order = np.argsort(-occ, kind="stable")
    picks = [int(active_to_num[int(order[i])]) for i in range(k)]
    return tuple(sorted(picks))


# =========================
# Glavni pipeline za dati (K, t_hop, u_int)
# =========================
def run_jw_slater(H: np.ndarray, K: int, t_hop: float, u_int: float):
    active_to_num = build_active_space(H, K)
    h = build_h_effective(H, active_to_num, t_hop, u_int)
    eigs, V = np.linalg.eigh(h)
    givens, diag_signs = reck_decompose(V)
    qc = build_slater_circuit(K, N_PART, givens, diag_signs)
    sv = Statevector(qc)
    probs = slater_state_probs(sv.data, K)
    occ, total_n = occupation_from_sv(probs, K, N_PART)
    return active_to_num, eigs, V, occ, total_n


def optimize_hparams(H: np.ndarray):
    f_csv = freq_vector(H)
    s_tot = float(f_csv.sum())
    f_csv_n = f_csv / s_tot if s_tot > 0 else np.ones(N_MAX) / N_MAX
    best = None
    for K in GRID_K_ACTIVE:
        for t_hop in GRID_T_HOP:
            for u_int in GRID_U_INT:
                try:
                    active, _eigs, _V, occ, total_n = run_jw_slater(H, int(K), float(t_hop), float(u_int))
                    bi = bias_from_occupation(occ, active)
                    score = cosine(bi, f_csv_n)
                except Exception:
                    continue
                key = (score, int(K), -float(t_hop), -float(u_int))
                if best is None or key > best[0]:
                    best = (
                        key,
                        dict(
                            K=int(K),
                            t_hop=float(t_hop),
                            u_int=float(u_int),
                            score=float(score),
                            total_n=float(total_n),
                        ),
                    )
    return best[1] if best else None


def main() -> int:
    H = load_rows(CSV_PATH)
    if H.shape[0] < 1:
        print("premalo redova")
        return 1

    print("Q33 v1 Quantum Chemistry — Jordan-Wigner fermionski Slater determinant: CSV:", CSV_PATH)
    print(
        "redova:", H.shape[0],
        "| seed:", SEED,
        "| N_PART (loto 7):", N_PART,
        "| N_MAX (1..39):", N_MAX,
    )

    best = optimize_hparams(H)
    if best is None:
        print("grid optimizacija nije uspela")
        return 2
    print(
        "BEST hparam:",
        "K_ACTIVE=", best["K"],
        "| t_hop=", best["t_hop"],
        "| u_int=", best["u_int"],
        "| P(N_PART-sektor)=", round(float(best["total_n"]), 6),
        "| cos(bias, freq_csv)=", round(float(best["score"]), 6),
    )

    f_csv = freq_vector(H)
    s_tot = float(f_csv.sum())
    f_csv_n = f_csv / s_tot if s_tot > 0 else np.ones(N_MAX) / N_MAX

    print("--- demonstracija efekta (K, t_hop, u_int) ---")
    shown = 0
    for K in GRID_K_ACTIVE:
        for t_hop in GRID_T_HOP:
            for u_int in GRID_U_INT:
                try:
                    active, _eigs, _V, occ, total_n = run_jw_slater(H, int(K), float(t_hop), float(u_int))
                    bi = bias_from_occupation(occ, active)
                    cos_d = cosine(bi, f_csv_n)
                    pred_d = pick_next_combination(occ, active)
                    print(
                        f"  K={K:d}  t_hop={t_hop:.2f}  u_int={u_int:.2f}  "
                        f"P_N={total_n:.4f}  cos={cos_d:.6f}  NEXT={pred_d}"
                    )
                    shown += 1
                except Exception as e:
                    print(f"  K={K} t={t_hop} u={u_int}  skipped ({type(e).__name__})")
    print(f"(prikazano {shown} konfiguracija iz grida)")

    active, eigs, V, occ, total_n = run_jw_slater(
        H, int(best["K"]), float(best["t_hop"]), float(best["u_int"])
    )
    print("--- active space (top-K po freq) ---")
    print(f"  active_to_num[0..{len(active) - 1}] = {active}")
    print("--- spektar h (K × K efektivna jedno-čestična H) ---")
    print(
        f"  eig_min = {float(eigs.min()):+.6f}  "
        f"eig_max = {float(eigs.max()):+.6f}  "
        f"eig_occupied_sum (N najniže) = {float(eigs[:N_PART].sum()):+.6f}"
    )
    print(f"--- N-particle verifikacija: P(Σ n_p = {N_PART}) = {total_n:.10f}  (treba ≈ 1.0) ---")

    pred = pick_next_combination(occ, active)
    print("--- glavna predikcija (JW Slater determinant ground state) ---")
    print("predikcija NEXT:", pred)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""
Q33 v1 Quantum Chemistry — Jordan-Wigner fermionski Slater determinant: CSV: /data/loto7hh_4600_k31.csv
redova: 4600 | seed: 39 | N_PART (loto 7): 7 | N_MAX (1..39): 39
BEST hparam: K_ACTIVE= 14 | t_hop= 1.0 | u_int= 0.5 | P(N_PART-sektor)= 1.0 | cos(bias, freq_csv)= 0.612418
--- demonstracija efekta (K, t_hop, u_int) ---
  K=12  t_hop=0.25  u_int=0.00  P_N=1.0000  cos=0.568215  NEXT=(8, 22, 23, 26, 32, 34, 37)
  K=12  t_hop=0.25  u_int=0.20  P_N=1.0000  cos=0.548469  NEXT=(8, 11, 23, 26, 32, 33, 34)
  K=12  t_hop=0.25  u_int=0.50  P_N=1.0000  cos=0.513318  NEXT=(8, 11, 22, 23, 26, 32, 34)
  K=12  t_hop=0.50  u_int=0.00  P_N=1.0000  cos=0.571314  NEXT=(10, 22, 23, 26, 32, 34, 37)
  K=12  t_hop=0.50  u_int=0.20  P_N=1.0000  cos=0.567066  NEXT=(8, 22, 23, 26, 32, 34, 37)
  K=12  t_hop=0.50  u_int=0.50  P_N=1.0000  cos=0.553737  NEXT=(8, 11, 22, 23, 26, 32, 34)
  K=12  t_hop=1.00  u_int=0.00  P_N=1.0000  cos=0.569929  NEXT=(10, 22, 23, 26, 29, 32, 37)
  K=12  t_hop=1.00  u_int=0.20  P_N=1.0000  cos=0.570997  NEXT=(10, 22, 23, 26, 32, 34, 37)
  K=12  t_hop=1.00  u_int=0.50  P_N=1.0000  cos=0.566816  NEXT=(8, 11, 22, 23, 26, 32, 34)
  K=14  t_hop=0.25  u_int=0.00  P_N=1.0000  cos=0.608751  NEXT=(8, 9, 10, 26, 32, 34, 39)
  K=14  t_hop=0.25  u_int=0.20  P_N=1.0000  cos=0.577216  NEXT=(8, 11, 23, 26, 32, 34, 37)
  K=14  t_hop=0.25  u_int=0.50  P_N=1.0000  cos=0.532444  NEXT=(8, 11, 22, 23, 26, 32, 34)
  K=14  t_hop=0.50  u_int=0.00  P_N=1.0000  cos=0.603598  NEXT=(8, 9, 10, 32, 34, 35, 39)
  K=14  t_hop=0.50  u_int=0.20  P_N=1.0000  cos=0.611872  NEXT=(8, 23, 26, 32, 34, 37, 39)
  K=14  t_hop=0.50  u_int=0.50  P_N=1.0000  cos=0.578333  NEXT=(8, 22, 23, 26, 32, 34, 37)
  K=14  t_hop=1.00  u_int=0.00  P_N=1.0000  cos=0.599012  NEXT=(9, 10, 26, 32, 34, 35, 39)
  K=14  t_hop=1.00  u_int=0.20  P_N=1.0000  cos=0.609803  NEXT=(8, 9, 10, 26, 32, 34, 39)
  K=14  t_hop=1.00  u_int=0.50  P_N=1.0000  cos=0.612418  NEXT=(8, 10, 23, 26, 32, 34, 37)
(prikazano 18 konfiguracija iz grida)
--- active space (top-K po freq) ---
  active_to_num[0..13] = [8, 23, 26, 34, 37, 11, 32, 33, 22, 39, 29, 10, 35, 9]
--- spektar h (K x K efektivna jedno-čestična H) ---
  eig_min = -19.149578  eig_max = -3.072685  eig_occupied_sum (N najniže) = -42.008925
--- N-particle verifikacija: P(Σ n_p = 7) = 1.0000000000  (treba ≈ 1.0) ---
--- glavna predikcija (JW Slater determinant ground state) ---
predikcija NEXT: (8, 10, x, y, z, 34, 37)
"""



"""
Q33_Chemistry_JordanWigner_v1.py — Quantum chemistry pristup loto predikciji preko
fermionskog second-quantization Hamiltonijana, Jordan-Wigner mapiranja, i Thouless
teorema za Slater determinant state preparation.

Koncept:
Loto ≡ odabir 7 od 39 → strukturno identičan N-particle sektoru fermionskog
mnogočestičnog sistema. JW mapira fermionske operatore c_p†, c_p u Pauli stringove
(X_p, Y_p, Z_{<p}). Efektivni jedno-čestični Hamiltonian iz CSV-a ima "niska
energija = visoka freq" strukturu; njegov ground state u N-particle sektoru je
Slater determinant što se efikasno priprema mrežom adjacent Givens rotacija (linearna
dubina, Thouless teorema).

Kolo (K_ACTIVE qubit-a, BEZ ancilla, BEZ phase):
  X na qubit-e 0..N-1 → |1^N 0^(K-N)⟩ (Slater u freq-rangiranoj bazi).
  Z na qubit-ima gde diag(±1) iz Reck-trianguralizacije daje -1 (global phase).
  Mreža ~K²/2 adjacent Givens rotacija G_i(θ_i) u reverznom redosledu sa
  negiranim uglovima — implementira orbit-rotaciju V (eigenvektore h).

Efektivni K x K real simetrični Hamiltonian h:
  h_pp   = -freq[num_p] / <freq>                    (orbital energies)
  h_pq   = -t_hop · P[num_p, num_q] / <P>           (hopping)
  h_pp  += Σ_q (-u_int · P[num_p, num_q] / <P>) · n̄  (mean-field density-density)

Eigendecomposition h = V D Vᵀ; occupied orbitals = N najniža eigenvalues-a.

Adjacent Givens gate (JW: Z-string prazan za adjacent orbitale):
  G_i(θ):
    |00⟩ → |00⟩,  |11⟩ → |11⟩,
    |01⟩ →  cos(θ)|01⟩ + sin(θ)|10⟩,
    |10⟩ → -sin(θ)|01⟩ + cos(θ)|10⟩.

Readout:
  Statevector |ψ_SD⟩; za svako I ⊂ active sa |I| = N, amplituda = det(V[I, :N]).
  Occupation: P(n_p = 1) = Σ_{I: p∈I} |amp(I)|².
  TOP-N_PART orbitala po occupation → map active → original brojevi 1..39 → NEXT.

Active space:
  K_ACTIVE top-freq brojeva; grid (K, t_hop, u_int) po cos(bias_39, freq_csv).

Tehnike:
Jordan-Wigner mapiranje: c_p† ↔ Pauli string sa Z-stepenicom.
Number-preserving adjacent Givens (4x4 unitar, JW Z-string prazan za adjacent parove).
Reck decomposition: K x K orth V → K(K-1)/2 adjacent Givens + diag(±1).
Thouless teorema: orbitalna rotacija → Slater determinant amplitudes = determinante.
Active-space aproksimacija (top-K freq) + mean-field density-density shift.
Deterministička eigendecomposition za KOMPILACIJU kola (isti pattern kao Q26/Q27/Q29).
Egzaktni Statevector sa N-particle sektor verifikacijom (P(Σ n_p = N) ≈ 1.0).
Deterministička grid-optimizacija (K_ACTIVE, t_hop, u_int).

Prednosti:
Prirodno mapiranje: N-particle sektor FERMIONSKOG sistema ≡ C(N_MAX, N_PART)
  kombinatorni izbor 7-od-39 (NIJEDAN prethodni fajl nema ovo 1-na-1 strukturno
  mapiranje — svi prethodni su amplitudno-kodirali N_MAX u 2^nq ≠ C(N, 7)).
Pauli-princip AUTOMATSKI zabranjuje duple brojeve — ground state je valid
  kombinacija po konstrukciji, bez post-processing sanitizacije.
Svi gate-ovi su number-preserving → |ψ⟩ je STROGO u 7-particle sektoru
  (egzaktna verifikacija kroz Statevector-a).
Linearna dubina kola (O(K²) Givens u O(K) paralelnih slojeva — Kivlichan et al.).
Mali qubit budget: K_ACTIVE = 14, BEZ ancilla, BEZ phase registra.
Ceo CSV: h i active-space iz CELOG CSV-a.

Nedostaci:
Active-space aproksimacija: izabrano top-K od N_MAX = 39 (ostali 25 brojeva imaju
  nulti prior — razumno jer su retki, ali nisu strogo isključeni u klasičnom lotu).
Slater determinant je Hartree-Fock aproksimacija ground state-a; za pun two-body
  Hamiltonian potrebna bi bila CI ili coupled cluster korekcija (nije u obimu
  ovog fajla da se ne krši rule 13 o duplikatima — posle-HF metode bi mogli biti
  sledeći korak u posebnom fajlu).
Real orthogonal V (h je real simetrična) → svi Givens ugao-vi su realni; za
  kompleksni Hamiltonian bila bi potrebna kompleksna Reck dekompozicija.
Reck dekompozicija je klasičan preprocessing (eigendecomposition h + Givens
  zeroing) — analogno klasičnom eigendecomposition-u u Q26/Q27/Q29 QPE-u.
"""



"""
Quantum Chemistry / Jordan-Wigner fermionski Hamiltonian Svaki broj 1-39 = fermionski orbital. 
CSV ko-okurencije → fermionski interakcioni Hamiltonian H = Σ_ij t_ij·a_i†a_j + Σ_ijkl U_ijkl·a_i†a_j†a_k a_l. 
Jordan-Wigner mapiranje na qubit-e. Ground-state u sektoru sa 7 čestica = najverovatnija kombinacija. 
Nova grana: many-body quantum physics / second quantization.



kvantno-hemijski pristup koji 1-na-1 strukturno mapira loto kombinatoriku na fixed-particle sektor fermionskog sistema. Ključne komponente:

Active space (top-K brojeva po freq) + efektivni KxK Hamiltonian h (orbital energies - freq, hopping - t_hop·P, mean-field density-density - u_int)
Jordan-Wigner mapiranje sa Pauli stringovima (Z-string za adjacent orbitale prazan)
Thouless teorema: orbital rotacija V iz eigh(h) → Slater determinant kroz mrežu adjacent Givens rotacija (Reck dekompozicija)
Number-preserving 2-qubit gate G_i(θ): |01⟩ → cos θ|01⟩ + sin θ|10⟩, |10⟩ → -sin θ|01⟩ + cos θ|10⟩
Inicijalno stanje |1⁷ 0^(K-7)⟩, Givens mreža ga evoluira u Slater ground state
Readout: P(n_p = 1) = Σ_{I: p∈I} |amp(I)|², TOP-7 po okupaciji → NEXT
Ugrađena verifikacija P(Σn_p = 7) ≈ 1.0 (number conservation)
Pauli-princip automatski zabranjuje duple brojeve — valid 7-kombinacija po konstrukciji.




Jordan-Wigner fermionski Hamiltonian je prirodna matematička struktura loto problema:

Svaki broj 1..39 = fermionski mod (orbital).
Sektor sa tačno 7 čestica = skup svih valid 7-kombinacija (Pauli-princip zabranjuje duple).
Ground state H |ψ⟩ = E_0|ψ⟩ u 7-particle sektoru = najverovatnija kombinacija.
Merenje svakog qubit-a (Pauli Z) direktno daje 0/1 indikator je li broj izabran.
CSV statistike (freq, pair co-occurrence) prirodno ulaze kao one-body Σ t_ij·a_i†a_j i two-body Σ U_ijkl·a_i†a_j†a_k a_l članovi.
To je 1-na-1 mapiranje između fermionskog mnogočestičnog sistema i loto kombinatorike. 

Tehnička napomena: 39 modova → 39 qubit-a (2^39 = neizvodljivo). 
Rešenje: active space (K = 12-16 top-freq brojeva) + fiksni-broj-čestica sektor. C(14, 7) = 3432 stanja — lako za state-vector simulaciju.
"""





"""
Koja je bolja predikcija v1 ili v2, sa argumentima na obe strane? 

Po metrici (cos sa freq_csv):

v1: cos = 0.612, NEXT = (8, 10, x, y, z, 34, 37)
v2: cos = 0.446, NEXT = (8, 23, x, y, z, 35, 37)
→ v1 pobeđuje strogo numerički, jer many-body Slater na top-14 freq brojeva prirodno koncentriše izbor oko najčešćih brojeva, 
pa je sličnost sa istorijskom freq raspodelom veća.

Po strukturnoj realističnosti za loto 
(kako se brojevi stvarno izvlače, po pozicijama, uređeno, full range):
v2 je realistično bolja jer:
Full range [1, 39] je pokriven (v1 isključuje 25 nisko-freq brojeva — npr. brojevi 1, 2, 3, 4 ne mogu biti predviđeni u v1)
Strogo uređeno Num_1 < Num_2 < ... < Num_7 by construction (kao u realnom listiću)
Svaka pozicija ima svoj fizički opseg [i, i+32] — bliže stvarnoj statistici po pozicijama
(min qubit budget = 6 umesto 14)

Ključna razlika u filozofiji:
v1 odgovara na pitanje: "koji su top-7 najverovatniji brojeva u many-body ground state-u?"
v2 odgovara na pitanje: "koja je najverovatnija realna 7-kombinacija po pozicijama, sa autoregresivnom strukturom?"

Moje mišljenje za loto NEXT: 
v2 je bolja kao predikcija jer matematički modeluje stvarnu strukturu izvlačenja 
(po pozicijama, uređeno, full range), 
dok je v1 "best fit u istorijski freq prosek" — što po definiciji znači da daje "očekivanu" kombinaciju, 
a stvarna izvlačenja su upravo ono što odstupa od proseka.

Cos = 0.612 kod v1 ne znači "tačnija predikcija", već "veća sličnost sa istorijskim prosečnim brojevima". 
Za predikciju budućeg izvlačenja, to je upravo ono što ne želimo previše — gubi se mogućnost da budući izvuče iz nisko-freq brojeva.

Zaključak: bolja je v2.
"""
