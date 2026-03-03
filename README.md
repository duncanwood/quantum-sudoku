# quantum-sudoku
Sudoku solver for quantum computers with Qiskit.

Run python/quantumsudoku.py to generate an example solution. 
A general Grover's Search circuit can be built with 
qc, slot_regs, slot_positions, n_iter = build_grover_circuit(puzzle), 
given some puzzle as a 2D array of numbers, 
with 0 representing an empty slot. 
