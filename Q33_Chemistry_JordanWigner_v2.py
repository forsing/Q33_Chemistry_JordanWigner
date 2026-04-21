#!/usr/bin/env python3

"""
Q33 v2 Quantum Chemistry — per-poziciju (min broj qubit-a)
v2 sloj dodaje per-pozicijski single-particle
ground-state podproblem na tačno NQ = ⌈log₂(33)⌉ = 6 qubit-a po poziciji, sekvencijalno
(qubit budget = 6 u bilo kom trenutku na lokalnom simulatoru), sa autoregresivnim
maskiranjem radi strogog Num_1 < Num_2 < ... < Num_7 uređenja.


v2 rešenje (čisto kvantno, bez klasičnog ML-a, bez hibrida):
  Za svaku poziciju i ∈ {1..7} gradi se 64x64 Hermitska h_i (pad 33 → 64) sa
  ISTOM strukturom kao Q33 v1 h (dijagonala = -freq/<freq>, off-diag = -t_hop·P/<P>),
  ali restringovana na dozvoljeni opseg A_i = {i, i+1, ..., i+32}:
      h_i[j, j]  = -freq[i+j-1] / <freq>                za i+j ∈ A_i (j = 0..32)
      h_i[j, j]  = +LARGE                               inače (padding + autoregr. mask)
      h_i[j, k]  = -t_hop · P[i+j-1, i+k-1] / <P>       (hopping između dozvoljenih)
  Autoregresivno maskiranje: za j sa num_j ≤ Num_{i-1} postavi h_i[j,j] = +LARGE.
  Ground state |v_0⟩ = eigh(h_i)[:, 0] (single-particle sector) priprema se kroz
  StatePreparation na NQ = 6 qubit-a (⌈log₂(33)⌉ = 6). Statevector → argmax u
  dozvoljenom opsegu → Num_i. Isti 6-qubit registar se reciklira 7 puta.

Strukturna mapa v2:
  Q33 v1: many-body fermionski (N_PART=7 čestica u K_ACTIVE=14 orbitala, C(14,7)=3432).
  v2:     7x single-particle (N_PART=1 čestica u 33-orbital opsegu = 64-dim sa padding-om),
          amplitude-kodiranje indeksa u 6 qubit-a. Autoregresivna sekvenca garantuje
          Num_1 < Num_2 < ... < Num_7 by construction.

Qubit budget v2:
  Po pozicija: NQ = 6. Ukupno istovremeno u memoriji: 6 qubit-a (sekvencijalno
  recikliranje registra). Znatno manje od Q33 v1 (K_ACTIVE = 14).

Active space v2:
  FULL [1, 39] pokriven kroz Num_i ∈ [i, i+32] (nema top-K freq filtera → svi
  brojevi su kandidati u odgovarajućim pozicijama).

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
from qiskit.circuit.library import UnitaryGate, StatePreparation
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

# --- v2 konfiguracija (per-poziciju, min broj qubit-a) ---
NQ_V2 = 6                         # ⌈log₂(33)⌉ = 6 qubit-a po poziciji
POS_RANGE_V2 = 33                 # broj dozvoljenih vrednosti po poziciji (i..i+32)
LARGE_PENALTY_V2 = 1.0e6          # penal za padding i autoregresivno maskirane indekse
GRID_T_HOP_V2 = (0.0, 0.25, 0.5, 1.0, 2.0)


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
# Glavni pipeline za dati (K, t_hop, u_int)  — Q33 v1 referentni many-body pipeline
# (OČUVAN iz Q33 v1 verbatim, ne koristi se u v2 main-u ali ostavljen kao referenca.)
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


# =========================
# V2: per-poziciju single-particle Hamiltonian (64×64, pad 33 → 64)
# Ista struktura kao Q33 h (−freq na dijagonali, −t_hop·P off-diag), ali restringovana
# na dozvoljeni opseg A_i = {i, i+1, ..., i+32} i autoregresivno maskirana za
# prethodno izabrane brojeve (num ≤ prev_picked → diag = +LARGE).
# =========================
def position_allowed_numbers_v2(pos_1based: int) -> List[int]:
    lo = int(pos_1based)
    hi = int(pos_1based) + POS_RANGE_V2 - 1
    return list(range(lo, hi + 1))


def build_h_position_v2(
    freq: np.ndarray,
    P: np.ndarray,
    pos_1based: int,
    prev_picked: int,
    t_hop: float,
) -> np.ndarray:
    dim = 2 ** NQ_V2
    h = np.zeros((dim, dim), dtype=np.float64)

    allowed = position_allowed_numbers_v2(pos_1based)
    f_mean = float(freq.mean()) + 1e-18
    P_mean = float(P.mean()) + 1e-18

    for j in range(dim):
        if j >= len(allowed):
            h[j, j] = LARGE_PENALTY_V2
            continue
        num_j = allowed[j]
        if num_j <= prev_picked:
            h[j, j] = LARGE_PENALTY_V2
            continue
        h[j, j] = -float(freq[num_j - 1]) / f_mean

    for j in range(min(len(allowed), dim)):
        for k in range(min(len(allowed), dim)):
            if j == k:
                continue
            num_j = allowed[j]
            num_k = allowed[k]
            if num_j <= prev_picked or num_k <= prev_picked:
                continue
            h[j, k] = -float(t_hop) * float(P[num_j - 1, num_k - 1]) / P_mean

    h = 0.5 * (h + h.T)
    return h


def ground_state_v2(h_pos: np.ndarray) -> np.ndarray:
    eigs, V = np.linalg.eigh(h_pos)
    ground = V[:, 0].astype(np.float64)
    norm = float(np.linalg.norm(ground))
    if norm < 1e-18:
        ground = np.ones_like(ground) / np.sqrt(ground.size)
    else:
        ground = ground / norm
    return ground


def build_position_circuit_v2(ground_amp: np.ndarray) -> QuantumCircuit:
    q_reg = QuantumRegister(NQ_V2, name="p")
    qc = QuantumCircuit(q_reg)
    qc.append(StatePreparation(ground_amp.tolist()), q_reg)
    return qc


def pick_number_v2(probs: np.ndarray, pos_1based: int, prev_picked: int) -> Tuple[int, float]:
    allowed = position_allowed_numbers_v2(pos_1based)
    dim = probs.size
    best_num = -1
    best_prob = -1.0
    for j in range(min(len(allowed), dim)):
        num_j = allowed[j]
        if num_j <= prev_picked:
            continue
        p = float(probs[j])
        if p > best_prob:
            best_prob = p
            best_num = num_j
    return int(best_num), float(best_prob)


# =========================
# V2: autoregresivni loop pozicija 1..7  (strogo uređeno Num_1 < ... < Num_7)
# =========================
def run_per_position_v2(H: np.ndarray, t_hop: float) -> Tuple[Tuple[int, ...], List[float]]:
    freq = freq_vector(H)
    P = pair_matrix(H)
    picks: List[int] = []
    conf: List[float] = []
    prev = 0
    for pos in range(1, N_NUMBERS + 1):
        h_pos = build_h_position_v2(freq, P, pos, prev, t_hop)
        ground = ground_state_v2(h_pos)
        qc = build_position_circuit_v2(ground)
        sv = Statevector(qc)
        probs = np.abs(sv.data) ** 2
        num_i, p_i = pick_number_v2(probs, pos, prev)
        picks.append(int(num_i))
        conf.append(float(p_i))
        prev = int(num_i)
    return tuple(sorted(picks)), conf


def bias_from_combination_v2(picks: Tuple[int, ...], n_max: int = N_MAX) -> np.ndarray:
    b = np.zeros(n_max, dtype=np.float64)
    for n in picks:
        if 1 <= int(n) <= n_max:
            b[int(n) - 1] = 1.0
    s = float(b.sum())
    return b / s if s > 0 else b


def optimize_hparams_v2(H: np.ndarray):
    f_csv = freq_vector(H)
    s_tot = float(f_csv.sum())
    f_csv_n = f_csv / s_tot if s_tot > 0 else np.ones(N_MAX) / N_MAX
    best = None
    for t_hop in GRID_T_HOP_V2:
        try:
            picks, conf = run_per_position_v2(H, float(t_hop))
            bi = bias_from_combination_v2(picks)
            score = cosine(bi, f_csv_n)
        except Exception:
            continue
        key = (score, -float(t_hop))
        if best is None or key > best[0]:
            best = (key, dict(t_hop=float(t_hop), picks=picks, conf=conf, score=float(score)))
    return best[1] if best else None


# =========================
# Q33 v1 referentni optimize_hparams (OČUVAN verbatim, nije pozvan u v2 main-u)
# =========================
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

    print("Q33 v2 Quantum Chemistry — per-poziciju 6-qubit ground state: CSV:", CSV_PATH)
    print(
        "redova:", H.shape[0],
        "| seed:", SEED,
        "| NQ_V2 (po poziciji):", NQ_V2,
        "| POS_RANGE_V2:", POS_RANGE_V2,
        "| qubit budget (lokalno, u bilo kom trenutku):", NQ_V2,
    )
    print("--- opsezi po poziciji (Pravilo 15) ---")
    for pos in range(1, N_NUMBERS + 1):
        lo = pos
        hi = pos + POS_RANGE_V2 - 1
        print(f"  Num{pos} ∈ [{lo}, {hi}]  ({POS_RANGE_V2} vrednosti)")

    best = optimize_hparams_v2(H)
    if best is None:
        print("v2 grid optimizacija nije uspela")
        return 2
    print(
        "BEST v2 hparam:",
        "t_hop=", best["t_hop"],
        "| cos(bias, freq_csv)=", round(float(best["score"]), 6),
    )

    f_csv = freq_vector(H)
    s_tot = float(f_csv.sum())
    f_csv_n = f_csv / s_tot if s_tot > 0 else np.ones(N_MAX) / N_MAX

    print("--- demonstracija efekta t_hop (v2 per-poziciju) ---")
    for t_hop in GRID_T_HOP_V2:
        picks, conf = run_per_position_v2(H, float(t_hop))
        bi = bias_from_combination_v2(picks)
        cos_d = cosine(bi, f_csv_n)
        conf_str = "[" + ", ".join(f"{c:.4f}" for c in conf) + "]"
        print(f"  t_hop={t_hop:.2f}  cos={cos_d:.6f}  NEXT={picks}  conf={conf_str}")

    picks, conf = run_per_position_v2(H, float(best["t_hop"]))
    print("--- glavna predikcija v2 (per-poziciju 6-qubit ground state, autoregresivno) ---")
    print("predikcija NEXT:", picks)
    print("  po-pozicijski confidence (max prob):", [round(c, 6) for c in conf])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""
Q33 v2 Quantum Chemistry — per-poziciju 6-qubit ground state: CSV: /data/loto7hh_4600_k31.csv
redova: 4600 | seed: 39 | NQ_V2 (po poziciji): 6 | POS_RANGE_V2: 33 | qubit budget (lokalno, u bilo kom trenutku): 6
--- opsezi po poziciji (Pravilo 15) ---
  Num1 ∈ [1, 33]  (33 vrednosti)
  Num2 ∈ [2, 34]  (33 vrednosti)
  Num3 ∈ [3, 35]  (33 vrednosti)
  Num4 ∈ [4, 36]  (33 vrednosti)
  Num5 ∈ [5, 37]  (33 vrednosti)
  Num6 ∈ [6, 38]  (33 vrednosti)
  Num7 ∈ [7, 39]  (33 vrednosti)
BEST v2 hparam: t_hop= 0.25 | cos(bias, freq_csv)= 0.445721
--- demonstracija efekta t_hop (v2 per-poziciju) ---
  t_hop=0.00  cos=0.444769  NEXT=(8, 23, 26, 34, 37, 38, 39)  conf=[1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
  t_hop=0.25  cos=0.445721  NEXT=(8, 23, 26, 32, 34, 35, 37)  conf=[0.0372, 0.0453, 0.0941, 0.1122, 0.2151, 0.2770, 0.2779]
  t_hop=0.50  cos=0.445721  NEXT=(8, 23, 26, 32, 34, 35, 37)  conf=[0.0368, 0.0447, 0.0929, 0.1108, 0.2101, 0.2743, 0.2700]
  t_hop=1.00  cos=0.444256  NEXT=(8, 23, 26, 32, 35, 37, 39)  conf=[0.0366, 0.0444, 0.0923, 0.1101, 0.2080, 0.3673, 0.5033]
  t_hop=2.00  cos=0.444256  NEXT=(8, 23, 26, 32, 35, 37, 39)  conf=[0.0365, 0.0442, 0.0920, 0.1097, 0.2080, 0.3630, 0.5016]
--- glavna predikcija v2 (per-poziciju 6-qubit ground state, autoregresivno) ---
predikcija NEXT: (8, 23, x, y, z, 35, 37)
  po-pozicijski confidence (max prob): [0.037235, 0.045298, 0.094093, 0.112193, 0.215058, 0.277006, 0.27788]
"""





"""
Quantum Chemistry / Jordan-Wigner fermionski Hamiltonian Svaki broj 1-39 = fermionski orbital. 
CSV ko-okurencije → fermionski interakcioni Hamiltonian H = Σ_ij t_ij·a_i†a_j + Σ_ijkl U_ijkl·a_i†a_j†a_k a_l. 
Jordan-Wigner mapiranje na qubit-e. Ground-state u sektoru sa 7 čestica = najverovatnija kombinacija. 
Nova grana: many-body quantum physics / second quantization.





kvantno-hemijski pristup koji 1-na-1 strukturno mapira loto kombinatoriku na fixed-particle sektor fermionskog sistema. 

Ključne komponente:

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
To je 1-na-1 mapiranje između fermionskog mnogočestičnog sistema i loto kombinatorike. Mnogo prirodnije od svega dosad koristenog.

Tehnička napomena: 
39 modova → 39 qubit-a (2^39 = neizvodljivo). 
Rešenje: active space (K = 12-16 top-freq brojeva) + fiksni-broj-čestica sektor. C(14, 7) = 3432 stanja — lako za state-vector simulaciju.
"""



"""

Šta se RAZLIKUJE u v2 u odnosu na Q33 v1:

  (min broj qubit-a): svaka pozicija Num_i ∈ [i, i+32] → 33 dozvoljene
  vrednosti po poziciji → NQ_V2 = ⌈log₂(33)⌉ = 6 qubit-a po poziciji.

  Struktura v2 kola:
    • Sekvencijalno po pozicijama i = 1..7 (isti 6-qubit registar se reciklira).
    • Per-pozicija h_i (64x64, pad 33 → 64):
        dijagonala   h_i[j,j]  = -freq[i+j-1] / <freq>              za i+j ∈ A_i
                     h_i[j,j]  = +LARGE                              inače (padding /
                                                                     autoregr. maskiranje:
                                                                     num ≤ Num_{i-1})
        off-diag     h_i[j,k]  = -t_hop · P[i+j-1, i+k-1] / <P>
      (Ista struktura članova kao Q33 h — -freq dijagonala, -t_hop·P hopping; izostavljen
      je mean-field u_int član jer je v2 single-particle po poziciji pa n̄ = 1/33 član
      u dijagonali je konstantan shift bez uticaja na argmax.)
    • Ground state |v_0⟩ = eigh(h_i)[:, 0] pripremljen kroz StatePreparation na 6 qubit-a.
    • Readout: Statevector → |amp|² → argmax u dozvoljenom opsegu (bez prethodno
      izabranih) → Num_i.
    • Autoregresivno maskiranje → strogo Num_1 < Num_2 < ... < Num_7 by construction.

  Qubit budget:
    Q33 v1: K_ACTIVE = 14 qubit-a (many-body 7-particle sektor, C(14,7) = 3432 stanja).
    v2:     NQ_V2 = 6 qubit-a (u bilo kom trenutku; sekvencijalno 7x), čisto amplitudno
            kodiranje indeksa dozvoljenih brojeva u poziciji.

  Active space:
    Q33 v1: top-K freq brojeva (active-space aproksimacija; isključeno ~25 brojeva).
    v2:     full [1, 39] pokriven kroz Num_i ∈ [i, i+32] (bez freq-filtera).

  
  Što je DODATO za v2:
    • NQ_V2, POS_RANGE_V2, LARGE_PENALTY_V2, GRID_T_HOP_V2 konstante.
    • position_allowed_numbers_v2, build_h_position_v2, ground_state_v2,
      build_position_circuit_v2, pick_number_v2, run_per_position_v2,
      bias_from_combination_v2, optimize_hparams_v2.
    • Novi main() koji poziva v2 pipeline (Q33 v1 optimize_hparams i run_jw_slater ostaju
      u fajlu kao referenca, nisu pozvani).

  Cross-reference:
    • v1 (referentni many-body JW): Q33_Chemistry_JordanWigner_v1.py
    • v2 (Pravilo 15, per-poziciju 6-qubit): ovaj fajl, Q33_Chemistry_JordanWigner_v2.py
"""





"""
sa 6 qubit-a se kvantno kolo lako izvršava na lokalnom M1 simulatoru (2⁶ = 64 stanja, Statevector = 1 KB, trenutno izračunavanje).

(6 qubit-a po poziciji umesto K=14 orbitala)

Arhitektura v2:

7 pozicija, svaka sa 33 dozvoljenih brojeva ([i, i+32])
Per-poziciju: 6 qubit-a (pad 33 → 64)
Per-poziciju Hamiltonian h_i (64x64, diag=-freq + hopping=-t_hop·P) sa autoregresivnim maskiranjem prethodno izabranih brojeva
Ground state |v_0⟩ iz eigh(h_i), StatePreparation na 6 qubit-a
Statevector → argmax → Num_i
Sekvencijalno: isti 6-qubit registar se reciklira 7 puta






Ključne promene u v2 u odnosu na v1:

6 qubit-a umesto 14 — tačno ⌈log₂(33)⌉ = 6 po poziciji
Full range [1, 39] pokriven — nema active-space gubitka (Q33 je bacao 25 brojeva)
Sekvencijalno — 6-qubit registar se reciklira 7 puta (qubit budget u bilo kom trenutku = 6)
Autoregresivno maskiranje — garantuje Num_1 < Num_2 < ... < Num_7 kroz h[j,j] += LARGE za već izabrane
Per-pozicijski Hermitijan h_i (64x64): dijagonala = -freq (orbital energy), off-diag = -t_hop·P (pair-hopping), eigh → ground state → StatePreparation na 6 qubit-a → Statevector → argmax

Rezultat testa (4600 redova, CSV, seed=39):

BEST: t_hop=0.25, cos(bias, freq_csv) = 0.445721
NEXT = (8, 23, x, y, z, 35, 37)
Qubit budget: 6 u bilo kom trenutku




v2 daje drugačiju kombinaciju jer je to fundamentalno drugi pristup: 
umesto Q33 v1 many-body Slater determinant-a na top-14 freq-rangiranih brojeva, 
v2 radi 7 nezavisnih single-particle ground-state izračuna, 
svaki u svom pozicijskom opsegu [i, i+32] sa autoregresivnim maskiranjem. 
Full range [1, 39] pokriven kroz opsege, strogo uređeno Num_1 < ... < Num_7.
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
