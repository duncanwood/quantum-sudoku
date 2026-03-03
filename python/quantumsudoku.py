
'''
Quantum Sudoku solver by Duncan Wood

Strategy:
- (C) compute missing values for each row, column and box
- (Q) oracle: one n-bit qubit for each missing value. 
            for each set (rows, etc) with missing values, 
            flag whether each set contains all of them.
            AND flags all sets.

Subroutines:
    -  contains: is a given number present in a set of registers?
    - multi_and
'''

from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.quantum_info import Statevector, Operator

import matplotlib.pyplot as plt

import numpy as np
import math

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import MCXGate

def add_multi_and(qc, inputs, tmp, out):
    '''
    Add circuit to qc that places AND(regs) into register 'out'.
    Uses tmp[0 : max(0, n-2)] for computation
    '''
    n = len(inputs)
    if n == 0:
        return
    elif n == 1: 
        qc.cx(inputs[0], out)
    elif n == 2:
        qc.ccx(*inputs, out)
    else:
        qc.ccx(inputs[0], inputs[1], tmp[0])
        for k in range(1, n-2):
            qc.ccx(tmp[k-1], inputs[k+1], tmp[k])

        qc.ccx(tmp[n-3], inputs[-1], out) # result to out
        #uncompute
        for k in reversed(range(1, n-2)): 
            qc.ccx(tmp[k-1], inputs[k+1], tmp[k])
        qc.ccx(inputs[0], inputs[1], tmp[0])


def add_equals(qc, reg, m, tmp, out):
    '''
    Add circuit to qc which sets 'out' to the result of reg == m
    for some integer m and a binary encoded quantum register reg
    '''
    bits = [(m >> i) & 1 for i in range(len(reg))]
    for i,b in enumerate(bits):
        if b==0:
            qc.x(reg[i])
    add_multi_and(qc, reg, tmp, out)
    for i,b in enumerate(bits):
        if b==0:
            qc.x(reg[i])

add_equals_unc = add_equals   # self-inverse

def add_contains(qc, regs, m, eq_tmp, eq_outs, or_tmp, out):
    '''
    Computes whether the number m is contained in any of regs
    via chained OR(EQUALS(m, reg)...)

    eq_tmp   : max(0, n_bits-2) scratch for equals AND-ladder (reused per reg)
    eq_outs  : len(slots) ancillas, one per equals result
    or_tmp   : max(0, len(slots)-2) scratch for the OR AND-ladder
    out      : single qubit starting in |0>
    '''
    k = len(regs)

    for i, reg in enumerate(regs):
        add_equals(qc, list(reg), m, eq_tmp, eq_outs[i])
    
    for i in range(k):
        qc.x(eq_outs[i])
    add_multi_and(qc, eq_outs, or_tmp, out)
    qc.x(out)
    for i in range(k):
        qc.x(eq_outs[i])
    
    for i in reversed(range(k)):
        add_equals_unc(qc, list(regs[i]), m, eq_tmp, eq_outs[i])
    
add_contains_unc = add_contains 

def add_set_complete(qc, regs, missing_vals,
                        eq_tmp, eq_outs, or_tmp,
                        contains_outs, and_tmp, out):
    '''
    Computes whether all missing_vals are present in regs
    '''
    v = len(missing_vals)

    for j, m in enumerate(missing_vals):
        add_contains(qc, regs, m, eq_tmp, eq_outs, or_tmp, contains_outs[j])
    
    add_multi_and(qc, contains_outs, and_tmp, out)

    # uncompute contains
    for j in reversed(range(v)):
        add_contains_unc(qc, regs, missing_vals[j], 
                         eq_tmp, eq_outs, or_tmp, contains_outs[j])
    

def grover_oracle(qc, slot_regs, sets, n_bits,
                  eq_tmp, eq_outs, or_tmp,
                  contains_outs, sc_and_tmp,
                  set_flags, global_flag, global_and_tmp,
                  phase_anc):
    '''
    Full Grover oracle. Flips the phase of |x> iff ALL constraint sets pass.

    Parameters
    ----------
    slot_regs       list of QuantumRegister, one n_bits-wide reg per empty cell
    sets            list of (slot_indices, missing_values)
    n_bits          int, bits per slot value
    eq_tmp          list of qubits, size max(0, n_bits-2)
    eq_outs         list of qubits, size max_empty_slots_in_any_set
    or_tmp          list of qubits, size max(0, max_empty-2)
    contains_outs   list of qubits, size max_missing_vals_in_any_set
    sc_and_tmp      list of qubits, size max(0, max_missing-2)
    set_flags       list of qubits, one per constraint set
    global_flag     single qubit
    global_and_tmp  list of qubits, size max(0, n_sets-2)
    phase_anc       single qubit initialised to |−⟩ = H|1⟩

    '''
    def _add_set_complete_for(idx):
        slot_indices, missing_vals = sets[idx]
        slots = [slot_regs[i] for i in slot_indices]
        add_set_complete(qc, slots, missing_vals,
                     eq_tmp, eq_outs, or_tmp,
                     contains_outs, sc_and_tmp,
                     set_flags[idx])

    # Compute
    for idx in range(len(sets)):
        _add_set_complete_for(idx)

    add_multi_and(qc, set_flags, global_and_tmp, global_flag)

    # Phase kickback
    qc.cx(global_flag, phase_anc)

    # Uncompute (identical structure, reversed)
    add_multi_and(qc, set_flags, global_and_tmp, global_flag)

    for idx in reversed(range(len(sets))):
        _add_set_complete_for(idx)


# ─────────────────────────────────────────────────────────────────────────────
# Puzzle helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_sudoku(puzzle):
    '''
    Parse a sudoku puzzle into slot positions and constraint sets.

    puzzle  : n x n list of lists, 0 = empty, 1..n = given value
    Returns : (slot_positions, sets)
      slot_positions : list of (row, col) for each empty cell, row-major order
      sets           : list of (slot_indices, missing_values) for every
                       row/col/box that contains at least one empty cell
    '''
    n      = len(puzzle)
    sqrt_n = int(np.sqrt(n))
    assert sqrt_n * sqrt_n == n, "n must be a perfect square"

    slot_positions = [
        (r, c)
        for r in range(n)
        for c in range(n)
        if puzzle[r][c] == 0
    ]
    pos_to_idx = {pos: i for i, pos in enumerate(slot_positions)}

    def make_set(cells):
        given   = {puzzle[r][c] for r, c in cells if puzzle[r][c] != 0}
        empties = [pos_to_idx[(r, c)] for r, c in cells if puzzle[r][c] == 0]
        missing = [v for v in range(1, n + 1) if v not in given]
        if empties and missing:
            return (empties, missing)
        return None

    sets = []
    for r in range(n):
        s = make_set([(r, c) for c in range(n)])
        if s: sets.append(s)
    for c in range(n):
        s = make_set([(r, c) for r in range(n)])
        if s: sets.append(s)
    for br in range(sqrt_n):
        for bc in range(sqrt_n):
            cells = [(br * sqrt_n + dr, bc * sqrt_n + dc)
                     for dr in range(sqrt_n) for dc in range(sqrt_n)]
            s = make_set(cells)
            if s: sets.append(s)

    return slot_positions, sets


def build_grover_circuit(puzzle):
    '''
    Build a complete Grover circuit for a sudoku puzzle.

    Returns (qc, slot_regs, slot_positions, n_iter).
    '''
    n      = len(puzzle)
    n_bits = int(np.ceil(np.log2(n + 1)))

    slot_positions, sets = parse_sudoku(puzzle)
    n_slots = len(slot_positions)

    if not sets:
        raise ValueError("No constraints - puzzle is already solved!")

    max_empty   = max(len(si) for si, mv in sets)
    max_missing = max(len(mv) for si, mv in sets)
    n_sets      = len(sets)

    eq_tmp_sz  = max(0, n_bits - 2)
    eq_outs_sz = max_empty
    or_tmp_sz  = max(0, max_empty - 2)
    c_outs_sz  = max_missing
    sc_tmp_sz  = max(0, max_missing - 2)
    sf_sz      = n_sets
    g_tmp_sz   = max(0, n_sets - 2)

    total_anc = (eq_tmp_sz + eq_outs_sz + or_tmp_sz + c_outs_sz
                 + sc_tmp_sz + sf_sz + g_tmp_sz + 2)
    print(f"Puzzle {n}×{n}  |  {n_slots} empty cells  |  {n_sets} constraint sets")
    print(f"n_bits={n_bits}, max_empty={max_empty}, max_missing={max_missing}")
    print(f"Ancilla qubits: {total_anc}  "
          f"(+ {n_slots * n_bits} data = {total_anc + n_slots * n_bits} total)")

    slot_regs = [QuantumRegister(n_bits, f's{i}') for i in range(n_slots)]

    def maybe(size, name):
        return QuantumRegister(size, name) if size > 0 else None

    eq_tmp_reg  = maybe(eq_tmp_sz,  'eq_tmp')
    eq_outs_reg = QuantumRegister(eq_outs_sz, 'eq_outs')
    or_tmp_reg  = maybe(or_tmp_sz,  'or_tmp')
    c_outs_reg  = QuantumRegister(c_outs_sz,  'c_outs')
    sc_tmp_reg  = maybe(sc_tmp_sz,  'sc_tmp')
    sf_reg      = QuantumRegister(sf_sz,      's_flags')
    g_tmp_reg   = maybe(g_tmp_sz,  'g_tmp')
    g_flag_reg  = QuantumRegister(1, 'g_flag')
    p_anc_reg   = QuantumRegister(1, 'p_anc')

    opt_regs = [r for r in [eq_tmp_reg, or_tmp_reg, sc_tmp_reg, g_tmp_reg]
                if r is not None]
    all_regs = (slot_regs
                + [eq_outs_reg, c_outs_reg, sf_reg, g_flag_reg, p_anc_reg]
                + opt_regs)
    qc = QuantumCircuit(*all_regs)

    def ql(reg):
        return list(reg) if reg is not None else []

    eq_tmp_q  = ql(eq_tmp_reg)
    eq_outs_q = list(eq_outs_reg)
    or_tmp_q  = ql(or_tmp_reg)
    c_outs_q  = list(c_outs_reg)
    sc_tmp_q  = ql(sc_tmp_reg)
    sf_q      = list(sf_reg)
    g_tmp_q   = ql(g_tmp_reg)
    g_flag_q  = g_flag_reg[0]
    p_anc_q   = p_anc_reg[0]

    # Phase ancilla |-> initialised once, persists across all iterations
    qc.x(p_anc_q)
    qc.h(p_anc_q)

    # Slots in uniform superposition
    for slot in slot_regs:
        qc.h(slot)

    n_iter = max(1, int(np.round(np.pi / 4 * np.sqrt(2 ** (n_slots * n_bits)))))
    print(f"Grover iterations: {n_iter}")

    for _ in range(n_iter):

        # Oracle 
        grover_oracle(qc, slot_regs, sets, n_bits,
                      eq_tmp_q, eq_outs_q, or_tmp_q,
                      c_outs_q, sc_tmp_q,
                      sf_q, g_flag_q, g_tmp_q,
                      p_anc_q)

        # Diffuser (2|s⟩⟨s| − I)
        all_data = [q for slot in slot_regs for q in slot]
        for slot in slot_regs:
            qc.h(slot)
            qc.x(slot)
        qc.h(all_data[-1])
        qc.mcx(all_data[:-1], all_data[-1])
        qc.h(all_data[-1])
        for slot in slot_regs:
            qc.x(slot)
            qc.h(slot)

    return qc, slot_regs, slot_positions, n_iter


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from qiskit_aer import AerSimulator
    from qiskit import transpile, ClassicalRegister

    sim = AerSimulator(method="statevector")

    # ── Integration test: 4x4 puzzle
    puzzle = [
        [1, 2, 3, 4],
        [3, 4, 0, 2],
        [2, 0, 4, 1],
        [4, 1, 2, 3],
    ]

    # Classical solution check
    slot_positions, sets = parse_sudoku(puzzle)
    print("\nEmpty cells (row, col):", slot_positions)

    print("\n" + "=" * 60)
    print("Building circuit for 4x4 puzzle (4 empty cells)...")
    qc, slot_regs, slot_positions, n_iter = build_grover_circuit(puzzle)

    qc_meas = qc.copy()
    for i, slot in enumerate(slot_regs):
        cr = ClassicalRegister(len(slot), f'c{i}')
        qc_meas.add_register(cr)
        qc_meas.measure(slot, cr)

    result = sim.run(transpile(qc_meas, optimization_level=0), shots=1024).result()
    counts = result.get_counts()
    top    = sorted(counts.items(), key=lambda x: -x[1])[:5]

    print("Top measurement outcomes:")
    for bitstring, count in top:
        words = bitstring.split()[::-1]        # registers right-to-left in Qiskit
        vals  = [int(w, 2) for w in words]     # MSB on left within each register
        cells = [(r, c) for r, c in slot_positions]
        assignment = {cells[i]: vals[i] for i in range(len(vals))}
        print(f"  {bitstring}  →  {assignment}  (shots: {count})")
