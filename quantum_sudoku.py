from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.quantum_info import Statevector, Operator

import matplotlib.pyplot as plt

import numpy as np
import math

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import MCXGate

def compare_nbit(qc, reg1, reg2, ancillas, anc_start, flag_qubit):
    """
    Compare two n-qubit registers reg1, reg2:
    - Use ancillas[anc_start : anc_start+n] for bitwise XNORs
    - Write the final equality flag into flag_qubit
    - Uncompute ancillas
    """
    n = len(reg1)
    # bitwise XNOR into ancillas
    for i in range(n):
        a = ancillas[anc_start + i]
        qc.cx(reg1[i], a)
        qc.cx(reg2[i], a)
        qc.x(a)                  # now a = 1 iff reg1[i] == reg2[i]

    # AND all ancillas down to one flag via a small CCX tree
    if n == 1:
        qc.cx(ancillas[anc_start], flag_qubit)
    elif n == 2:
        qc.ccx(ancillas[anc_start], ancillas[anc_start+1], flag_qubit)
    else:
        # TODO: build a binary-tree of Toffolis into intermediate ancillas...
        raise NotImplementedError("Extend to n>2 by chaining CCX into more ancillas")

    # uncompute the bitwise XNORs
    for i in reversed(range(n)):
        a = ancillas[anc_start + i]
        qc.x(a)
        qc.cx(reg2[i], a)
        qc.cx(reg1[i], a)


def all_different_circuit(regs, ancillas, unique):
    """
    regs: list of QuantumRegisters (each n qubits)
    qc:   QuantumCircuit that already contains regs plus ancillas, uniq
    """
    m = len(regs)
    n = len(regs[0])
    assert(len(ancillas)>=2*math.comb(m, 2)+ n+1)

    qc = QuantumCircuit(*regs, ancillas, unique)
    comp_flags = []
    anc_ptr = math.comb(m, 2)  # pointer into ancillas

    # Pairwise comparisons
    for i in range(m):
        for j in range(i+1, m):
            # pick one ancilla qubit to hold the equality flag
            flag = ancillas[anc_ptr]
            anc_ptr += 1
            comp_flags.append(flag)

            # compare regs[i] vs regs[j], using next n ancillas
            compare_nbit(qc, regs[i], regs[j], ancillas, 0, flag)
            qc.x(flag)

    # OR-reduce comp_flags into one "dup" qubit via OR(p,q) = ¬(¬p ∧ ¬q)
    dup = ancillas[anc_ptr]
    anc_ptr += 1
    or_ptr = anc_ptr

    or_controls = comp_flags.copy()
    while len(or_controls) > 1:
        p = or_controls.pop()
        q = or_controls.pop()

        tgt = ancillas[or_ptr]
        
        # OR step:
        qc.ccx(p, q, tgt)
        or_controls.append(tgt)
        or_ptr += 1
    or_ptr -= 1

    first_half = qc.copy()
    
    qc.cx(or_controls[0], unique)
    qc.append(first_half.inverse(), qc.qubits)
    return qc

    
def n_from_sudoku(sudoku: np.array):
    return int(np.ceil(np.sqrt(sudoku.shape[0])))
def nbits_from_sudoku(sudoku: np.array):
    return int(np.ceil(np.log2(sudoku.shape[0])))

def init_qbits(qc, regs, values, nbits):
    '''
    Encode values in binary into registers "regs" of circuit qc with nbits
    '''
    for i, val in enumerate(values):
        bits = [(val >> j) & 1 for j in range(nbits)]
        for ibit, bit in enumerate(bits):
            if bit:
                qc.x(regs[i][ibit])

def make_register_matrix(start: np.array, unknown_regs, digit_regs):
    """
    Convert starting sudoku position 'start' into a matrix 
    of quantum registers representing each slot.
    unknown_regs contain registers of unknown values.
    digit_regs are hard-coded registers with each value in binary
    """
    reg_matrix = [[None]*len(start) for _ in range(len(start))]
    unknown_index = 0
    for i, row in enumerate(start):
        for j, val in enumerate(row):
            print(i,j,val)
            if np.isnan(val):
                reg_matrix[i][j] = unknown_regs[unknown_index]
                print(i, j, reg_matrix)
                unknown_index+=1
            else:
                reg_matrix[i][j] = digit_regs[int(val)]
    print(reg_matrix)
    return reg_matrix

def sudoku_solver(start: np.array):
    """
    start: 2D array with starting values, and np.nan's for empty slots
    returns quantum circuit implementing grover's algorithm to find 
    the solution
    """
    N = start.shape[0]
    n = int(np.sqrt(N))
    nbits = nbits_from_sudoku(start)

    unknown_regs = [QuantumRegister(size=nbits, name=str(pos)) for pos in np.stack(np.where(np.isnan(start))).T]
    print(unknown_regs)
    digit_regs = [QuantumRegister(size=nbits, name=str(i)) for i in range(N)] # keep a copy of each number
    ancillas  = QuantumRegister(3*math.comb(N, 2)+nbits+1, 'anc') # for computing distinctness
    flags  = [QuantumRegister(1, f'flag-{str(i).zfill(int(np.ceil(np.log10(3*N))))}',) for i in range(3*N)] # store whether row, col, boxes are distinct
    solution_flag = QuantumRegister(1, 'sol')
    
    qc = QuantumCircuit(*unknown_regs, *digit_regs, ancillas, *flags, solution_flag)
    oracle_qc = qc.copy()
    diff_qc = qc.copy()
    init_qbits(qc, digit_regs, range(N))
    oracle_qc.h(solution_flag)
    oracle_qc.x(solution_flag)

    reg_matrix = make_register_matrix(start, unknown_regs, digit_regs)
    print(reg_matrix)
    

    boxes = []
    for bi in range(n):
        for bj in range(n):
            box = []
            for i in range(n):
                for j in range(n):
                    box.append(reg_matrix[n*bi + i][n*bj + j])
            boxes.append(box)
    

    # make oracle circuit flipping phase if 
    # each row, column and box has distinct values
    for i, row in enumerate(reg_matrix):
        print(row)
        oracle_qc.compose(all_different_circuit(row, ancillas, flags[i]), inplace=True)
    for j in range(len(reg_matrix[0])):
        col = [row[j] for row in reg_matrix]
        print(col)
        oracle_qc.compose(all_different_circuit(col, ancillas, flags[N+j]), inplace=True)
    print(boxes)
    for k, box in enumerate(boxes):
        print(box)
        oracle_qc.compose(all_different_circuit(box, ancillas, flags[2*N+k]), inplace=True)
    oracle_qc.append(MCXGate(len(flags)), flags + [solution_flag])

    oracle_qc.x(solution_flag)
    oracle_qc.h(solution_flag)
    
    # make diffusion circuit
    diff_qc.h(diff_qc.qubits)
    diff_qc.x(diff_qc.qubits)
    diff_qc.cz(diff_qc.qubits[:-1],diff_qc.qubits[-1])
    diff_qc.x(diff_qc.qubits)
    diff_qc.h(diff_qc.qubits)
    D = diff_qc.to_gate()


    
    N = 2**(len(unknown_regs)*nbits)
    K = int(np.rint(np.pi / (4 * np.arcsin(1 / np.sqrt(N))) - 1/2))
    # apply Grover's algorithm
    for i in np.arange(1,np.floor(K)):
        qc.compose(oracle_qc, inplace=True)
        qc.compose(D, inplace=True)


    
    return qc