import os
import argparse
import numpy as np
import stim
import yaml
from loguru import logger

BB_CODE_DICT = {
    '72': (6, 6, 3, 1, 2, 3, 1, 2),
    '144': (12, 6, 3, 1, 2, 3, 1, 2),
    '288': (12, 12, 3, 2, 7, 3, 1, 2),
    '360': (30, 6, 9, 1, 2, 3, 25, 26),
    '784': (28, 14, 26, 6, 8, 7, 9, 20)
}

# CNOT order of the depth-8 syndrome cycle in Bravyi et al. (arXiv:2308.07915),
# copied from BivariateBicycleCodes/decoder_setup.py
IBM_SX = ['idle', 1, 4, 3, 5, 0, 2]
IBM_SZ = [3, 5, 0, 1, 2, 4, 'idle']

BB_DEFAULT_DISTANCES = {
    '72': 6,
    '144': 12,
    '288': 18,
    '360': 24,
    '784': 24
}

def resolve_bb_prefix(code_type: str, search_base: str = "examples/txt/BB") -> tuple:
    """
    Resolve code prefix and key from code_type string (e.g. '144', '360', '[[360,12,leq24]]').
    Returns (prefix_path, bb_key, distance).
    """
    s = str(code_type).strip()
    if s.startswith('[['):
        prefix = os.path.join(search_base, s)
        # Extract middle or first number
        parts = s.strip('[]').split(',')
        bb_key = parts[0].strip()
        dist_str = parts[2].strip().replace('leq', '') if len(parts) >= 3 else '24'
        try:
            dist = int(dist_str)
        except ValueError:
            dist = 24
    elif s in ('360', '[[360,12,leq24]]'):
        bb_key = '360'
        dist = 24
        prefix = os.path.join(search_base, "[[360,12,leq24]]")
    elif s in BB_CODE_DICT:
        bb_key = s
        dist = BB_DEFAULT_DISTANCES.get(bb_key, 12)
        prefix = os.path.join(search_base, f"[[{bb_key},12,{dist}]]")
        if not os.path.exists(f"{prefix}hx.txt"):
            prefix = os.path.join(search_base, f"{bb_key}")
    else:
        bb_key = s
        dist = BB_DEFAULT_DISTANCES.get(bb_key, 12)
        prefix = os.path.join(search_base, s)
        
    return prefix, bb_key, dist

def load_bb_matrices(code_type: str, search_base: str = "examples/txt/BB"):
    """
    Load Hx, Hz, and Lx matrices for a BB code.
    """
    prefix, bb_key, dist = resolve_bb_prefix(code_type, search_base)
    
    # Try different naming conventions
    hx_path = f"{prefix}hx.txt"
    hz_path = f"{prefix}hz.txt"
    lx_path = f"{prefix}lx.txt"
    
    if not os.path.exists(hx_path):
        # Fallback search in folder
        found = False
        if os.path.exists(search_base):
            for fname in os.listdir(search_base):
                if bb_key in fname and fname.endswith('hx.txt'):
                    base_name = fname[:-6]
                    hx_path = os.path.join(search_base, f"{base_name}hx.txt")
                    hz_path = os.path.join(search_base, f"{base_name}hz.txt")
                    lx_path = os.path.join(search_base, f"{base_name}lx.txt")
                    found = True
                    break
        if not found:
            raise FileNotFoundError(f"Cannot find BB code matrices for '{code_type}' in '{search_base}'.")

    Hx = np.loadtxt(hx_path, dtype=int)
    Hz = np.loadtxt(hz_path, dtype=int) if os.path.exists(hz_path) else None
    Lx = np.loadtxt(lx_path, dtype=int)
    return Hx, Hz, Lx, bb_key, dist

def get_bb_matchings(ell: int, m: int, a1: int, a2: int, a3: int, b1: int, b2: int, b3: int):
    I_ell = np.identity(ell, dtype=int)
    I_m = np.identity(m, dtype=int)
    x = {i: np.kron(np.roll(I_ell, i, axis=1), I_m) for i in range(ell)}
    y = {i: np.kron(I_ell, np.roll(I_m, i, axis=1)) for i in range(m)}
    
    A1, A2, A3 = x[a1], y[a2], y[a3]
    B1, B2, B3 = y[b1], x[b2], x[b3]
    
    n_checks = ell * m
    matchings = []
    
    for M, offset in [(A1, 0), (A2, 0), (A3, 0), (B1, n_checks), (B2, n_checks), (B3, n_checks)]:
        edges = []
        for i in range(n_checks):
            j = np.nonzero(M[i, :])[0][0]
            edges.append((i, j + offset))
        matchings.append(edges)
        
    return matchings

def _build_7layer_bb_circuit(
    matchings: list,
    perm_A: tuple,
    perm_B: tuple,
    num_data: int,
    num_x_checks: int,
    num_z_checks: int,
    logical_x_matrix: np.ndarray,
    rounds: int,
    p_gate: float,
    p_meas: float,
    p_idle: float,
    p_data: float,
    sX: list = None,
    sZ: list = None
) -> stim.Circuit:
    circuit = stim.Circuit()
    data_qubits = list(range(num_data))
    x_ancillas = list(range(num_data, num_data + num_x_checks))
    z_ancillas = list(range(num_data + num_x_checks, num_data + num_x_checks + num_z_checks))
    all_ancillas = x_ancillas + z_ancillas

    circuit.append("RX", data_qubits)
    if p_idle > 0: circuit.append("Z_ERROR", data_qubits, p_idle)
    circuit.append("TICK")

    M0, M1, M2 = matchings[perm_A[0]], matchings[perm_A[1]], matchings[perm_A[2]]
    M3, M4, M5 = matchings[perm_B[0]], matchings[perm_B[1]], matchings[perm_B[2]]
    
    if sX is None: sX = ['idle', 4, 2, 5, 1, 3, 0]
    if sZ is None: sZ = [3, 0, 4, 2, 5, 1, 'idle']
    
    dict_M = {0: M0, 1: M1, 2: M2, 3: M3, 4: M4, 5: M5}
    dict_M_Z = {}
    for k_z, k_x in [(0, 3), (1, 4), (2, 5)]:
        edges_z = []
        for (r, c_plus_n) in dict_M[k_x]:
            c = c_plus_n - num_x_checks
            edges_z.append((c, r))
        dict_M_Z[k_z] = edges_z
        
    for k_z, k_x in [(3, 0), (4, 1), (5, 2)]:
        edges_z = []
        for (r, c) in dict_M[k_x]:
            edges_z.append((c, r + num_x_checks))
        dict_M_Z[k_z] = edges_z

    for r in range(rounds):
        if p_data > 0: circuit.append("DEPOLARIZE1", data_qubits, p_data)

        if all_ancillas:
            circuit.append("R", all_ancillas)
            if p_idle > 0: circuit.append("X_ERROR", all_ancillas, p_idle)
        circuit.append("TICK")
        
        if x_ancillas:
            circuit.append("H", x_ancillas)
            if p_gate > 0: circuit.append("DEPOLARIZE1", x_ancillas, p_gate)
        circuit.append("TICK")

        for layer in range(7):
            cnot_targets = []
            if sX[layer] != 'idle':
                m_x = dict_M[sX[layer]]
                for (chk, dat) in m_x:
                    cnot_targets.extend([x_ancillas[chk], dat])
                    
            if sZ[layer] != 'idle':
                m_z = dict_M_Z[sZ[layer]]
                for (chk, dat) in m_z:
                    cnot_targets.extend([dat, z_ancillas[chk]])
            
            if cnot_targets:
                circuit.append("CNOT", cnot_targets)
                if p_gate > 0: circuit.append("DEPOLARIZE2", cnot_targets, p_gate)
            circuit.append("TICK")

        if x_ancillas:
            circuit.append("H", x_ancillas)
            if p_gate > 0: circuit.append("DEPOLARIZE1", x_ancillas, p_gate)
        circuit.append("TICK")

        if all_ancillas:
            if p_meas > 0: circuit.append("X_ERROR", all_ancillas, p_meas)
            circuit.append("M", all_ancillas)
        
        offset = -len(all_ancillas)
        for i in range(num_x_checks):
            rec_now = stim.target_rec(offset + i)
            if r == 0:
                circuit.append("DETECTOR", [rec_now], [i, r])
            else:
                rec_prev = stim.target_rec(offset - len(all_ancillas) + i)
                circuit.append("DETECTOR", [rec_now, rec_prev], [i, r])
                
        for j in range(num_z_checks):
            if r > 0:
                rec_now = stim.target_rec(offset + num_x_checks + j)
                rec_prev = stim.target_rec(offset - len(all_ancillas) + num_x_checks + j)
                circuit.append("DETECTOR", [rec_now, rec_prev], [num_x_checks + j, r])

        circuit.append("TICK")

    if p_meas > 0: circuit.append("Z_ERROR", data_qubits, p_meas)
    circuit.append("MX", data_qubits)
    
    for i in range(num_x_checks):
        detector_targets = []
        for m_idx in range(6):
            m_list = dict_M[m_idx]
            for (chk, dat) in m_list:
                if chk == i:
                    detector_targets.append(stim.target_rec(-num_data + dat))
                    break
                    
        last_ancilla_rec = stim.target_rec(-num_data - len(all_ancillas) + i)
        detector_targets.append(last_ancilla_rec)
        circuit.append("DETECTOR", detector_targets, [i, rounds])
        
    num_logicals = logical_x_matrix.shape[0]
    for L_idx in range(num_logicals):
        logical_qubits = np.where(logical_x_matrix[L_idx] == 1)[0]
        obs_targets = [stim.target_rec(-num_data + q) for q in logical_qubits]
        circuit.append("OBSERVABLE_INCLUDE", obs_targets, L_idx)

    return circuit

def _build_ibm_bb_circuit(
    matchings: list,
    num_data: int,
    num_checks: int,
    logical_x_matrix: np.ndarray,
    rounds: int,
    p: float,
    sX: list = None,
    sZ: list = None
) -> stim.Circuit:
    """
    Memory-X circuit with the cycle and noise model of decoder_setup.py:
    - t0: PrepX X checks, Z CNOTs; t1-t5: X and Z CNOTs; t6: MeasZ Z checks, X CNOTs;
      t7: MeasX X checks, PrepZ Z checks, all data idle
    - CNOT: DEPOLARIZE2(p); idle data: DEPOLARIZE1(p); PrepX/MeasX: Z flip p; PrepZ/MeasZ: X flip p
    - check qubits are never idle; initial state and final data readout are noiseless
      (IBM appends two noiseless cycles)
    - only X-check detectors, since IBM decodes Z errors from X-check syndromes alone
    """
    if sX is None: sX = IBM_SX
    if sZ is None: sZ = IBM_SZ

    data_qubits = list(range(num_data))
    x_ancillas = list(range(num_data, num_data + num_checks))
    z_ancillas = list(range(num_data + num_checks, num_data + 2 * num_checks))

    # x_nb[d][i]: data neighbor of X check i in direction d (0-2 left via A, 3-5 right via B)
    # z_nb[d][j]: data neighbor of Z check j in direction d (0-2 left via B^T, 3-5 right via A^T)
    x_nb = [dict(m) for m in matchings]
    z_nb = [None] * 6
    for d_z, d_x in [(0, 3), (1, 4), (2, 5)]:
        z_nb[d_z] = {dat - num_checks: chk for (chk, dat) in matchings[d_x]}
    for d_z, d_x in [(3, 0), (4, 1), (5, 2)]:
        z_nb[d_z] = {dat: chk + num_checks for (chk, dat) in matchings[d_x]}

    def cnot_layer(t):
        cnot_targets = []
        busy_data = set()
        if sX[t] != 'idle':
            for i in range(num_checks):
                cnot_targets.extend([x_ancillas[i], x_nb[sX[t]][i]])
                busy_data.add(x_nb[sX[t]][i])
        if sZ[t] != 'idle':
            for j in range(num_checks):
                cnot_targets.extend([z_nb[sZ[t]][j], z_ancillas[j]])
                busy_data.add(z_nb[sZ[t]][j])
        circuit.append("CNOT", cnot_targets)
        if p > 0: circuit.append("DEPOLARIZE2", cnot_targets, p)
        idle_data = [q for q in data_qubits if q not in busy_data]
        if p > 0 and idle_data: circuit.append("DEPOLARIZE1", idle_data, p)
        circuit.append("TICK")

    circuit = stim.Circuit()
    circuit.append("RX", data_qubits)
    circuit.append("R", z_ancillas)
    circuit.append("TICK")

    for r in range(rounds):
        circuit.append("RX", x_ancillas)
        if p > 0: circuit.append("Z_ERROR", x_ancillas, p)
        cnot_layer(0)
        for t in range(1, 6):
            cnot_layer(t)
        if p > 0: circuit.append("X_ERROR", z_ancillas, p)
        circuit.append("M", z_ancillas)
        cnot_layer(6)

        if p > 0: circuit.append("DEPOLARIZE1", data_qubits, p)
        if p > 0: circuit.append("Z_ERROR", x_ancillas, p)
        circuit.append("MX", x_ancillas)
        circuit.append("R", z_ancillas)
        if p > 0: circuit.append("X_ERROR", z_ancillas, p)

        # records per cycle: Z checks then X checks
        for i in range(num_checks):
            rec_now = stim.target_rec(-num_checks + i)
            if r == 0:
                circuit.append("DETECTOR", [rec_now], [i, r])
            else:
                rec_prev = stim.target_rec(-3 * num_checks + i)
                circuit.append("DETECTOR", [rec_now, rec_prev], [i, r])
        circuit.append("TICK")

    circuit.append("MX", data_qubits)
    for i in range(num_checks):
        detector_targets = [stim.target_rec(-num_data + x_nb[d][i]) for d in range(6)]
        detector_targets.append(stim.target_rec(-num_data - num_checks + i))
        circuit.append("DETECTOR", detector_targets, [i, rounds])

    for L_idx in range(logical_x_matrix.shape[0]):
        logical_qubits = np.where(logical_x_matrix[L_idx] == 1)[0]
        circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-num_data + q) for q in logical_qubits], L_idx)

    return circuit

def generate_bb_code_circuit_level(
    Hx_matrix: np.ndarray, 
    Hz_matrix: np.ndarray,
    logical_x_matrix: np.ndarray, 
    rounds: int, 
    p_gate: float, 
    p_meas: float,
    p_idle: float,
    p_data: float
) -> stim.Circuit:
    circuit = stim.Circuit()
    num_x_checks, num_data = Hx_matrix.shape
    num_z_checks = Hz_matrix.shape[0] if Hz_matrix is not None else 0
    
    data_qubits = list(range(num_data))
    x_ancillas = list(range(num_data, num_data + num_x_checks))
    z_ancillas = list(range(num_data + num_x_checks, num_data + num_x_checks + num_z_checks))
    all_ancillas = x_ancillas + z_ancillas

    circuit.append("RX", data_qubits) 
    if p_idle > 0: circuit.append("Z_ERROR", data_qubits, p_idle)
    circuit.append("TICK")

    for r in range(rounds):
        if p_data > 0: circuit.append("DEPOLARIZE1", data_qubits, p_data)
        if all_ancillas:
            circuit.append("R", all_ancillas)
            if p_idle > 0: circuit.append("X_ERROR", all_ancillas, p_idle)
        circuit.append("TICK")
        
        if x_ancillas:
            circuit.append("H", x_ancillas)
            if p_gate > 0: circuit.append("DEPOLARIZE1", x_ancillas, p_gate)
        circuit.append("TICK")

        x_edges = [(x_ancillas[i], d) for i in range(num_x_checks) for d in np.where(Hx_matrix[i] == 1)[0]]
        z_edges = [(d, z_ancillas[i]) for i in range(num_z_checks) for d in np.where(Hz_matrix[i] == 1)[0]] if num_z_checks > 0 else []

        for target_edges in [x_edges, z_edges]:
            while target_edges:
                current_layer = []
                used_qubits = set()
                remaining_edges = []
                for (ctrl, targ) in target_edges:
                    if ctrl not in used_qubits and targ not in used_qubits:
                        current_layer.append((ctrl, targ))
                        used_qubits.add(ctrl)
                        used_qubits.add(targ)
                    else:
                        remaining_edges.append((ctrl, targ))
                
                if current_layer:
                    cnot_targets = [q for pair in current_layer for q in pair]
                    circuit.append("CNOT", cnot_targets)
                    if p_gate > 0: circuit.append("DEPOLARIZE2", cnot_targets, p_gate)
                circuit.append("TICK")
                target_edges = remaining_edges

        if x_ancillas:
            circuit.append("H", x_ancillas)
            if p_gate > 0: circuit.append("DEPOLARIZE1", x_ancillas, p_gate)
        circuit.append("TICK")

        if all_ancillas:
            if p_meas > 0: circuit.append("X_ERROR", all_ancillas, p_meas)
            circuit.append("M", all_ancillas)
        
        offset = -len(all_ancillas)
        for i in range(num_x_checks):
            rec_now = stim.target_rec(offset + i)
            if r == 0:
                circuit.append("DETECTOR", [rec_now], [i, r])
            else:
                rec_prev = stim.target_rec(offset - len(all_ancillas) + i)
                circuit.append("DETECTOR", [rec_now, rec_prev], [i, r])
                
        for j in range(num_z_checks):
            if r > 0:
                rec_now = stim.target_rec(offset + num_x_checks + j)
                rec_prev = stim.target_rec(offset - len(all_ancillas) + num_x_checks + j)
                circuit.append("DETECTOR", [rec_now, rec_prev], [num_x_checks + j, r])

        circuit.append("TICK")

    if p_meas > 0: circuit.append("Z_ERROR", data_qubits, p_meas)
    circuit.append("MX", data_qubits)
    
    for i in range(num_x_checks):
        detector_targets = [stim.target_rec(-num_data + q) for q in np.where(Hx_matrix[i] == 1)[0]]
        last_ancilla_rec = stim.target_rec(-num_data - len(all_ancillas) + i)
        detector_targets.append(last_ancilla_rec)
        circuit.append("DETECTOR", detector_targets, [i, rounds])
        
    num_logicals = logical_x_matrix.shape[0]
    for L_idx in range(num_logicals):
        obs_targets = [stim.target_rec(-num_data + q) for q in np.where(logical_x_matrix[L_idx] == 1)[0]]
        circuit.append("OBSERVABLE_INCLUDE", obs_targets, L_idx)

    return circuit

def generate_bb_code_phenomenological_circuit(
    Hx_matrix: np.ndarray, 
    logical_x_matrix: np.ndarray, 
    rounds: int, 
    p_data: float, 
    p_meas: float
) -> stim.Circuit:
    circuit = stim.Circuit()
    num_checks, num_qubits = Hx_matrix.shape
    
    circuit.append("RX", range(num_qubits)) 
    circuit.append("TICK")

    for r in range(rounds):
        if p_data > 0: circuit.append("Z_ERROR", range(num_qubits), p_data)

        for i in range(num_checks):
            connected_qubits = np.where(Hx_matrix[i] == 1)[0]
            mpp_targets = []
            for q in connected_qubits:
                mpp_targets.append(stim.target_x(q))
                mpp_targets.append(stim.target_combiner())
            mpp_targets.pop()
            circuit.append("MPP", mpp_targets, p_meas)

        for i in range(num_checks):
            rec_now = stim.target_rec(-num_checks + i)
            if r == 0:
                circuit.append("DETECTOR", [rec_now], [i, r])
            else:
                rec_prev = stim.target_rec(-2 * num_checks + i)
                circuit.append("DETECTOR", [rec_now, rec_prev], [i, r])

        circuit.append("TICK")

    circuit.append("MX", range(num_qubits)) 
    for i in range(num_checks):
        detector_targets = [stim.target_rec(-num_qubits + q) for q in np.where(Hx_matrix[i] == 1)[0]]
        last_mpp_rec = stim.target_rec(-num_qubits - num_checks + i)
        detector_targets.append(last_mpp_rec)
        circuit.append("DETECTOR", detector_targets, [i, rounds])
    
    num_logicals = logical_x_matrix.shape[0]
    for L_idx in range(num_logicals):
        obs_targets = [stim.target_rec(-num_qubits + q) for q in np.where(logical_x_matrix[L_idx] == 1)[0]]
        circuit.append("OBSERVABLE_INCLUDE", obs_targets, L_idx)

    return circuit

def generate_bb_interface_yaml(
    code_type: str, 
    error_rate: float, 
    rounds: int = None, 
    output_yaml_path: str = None,
    noise_model: str = "CIRCUIT_LEVEL",
    schedule: str = "ibm",
    search_base: str = "examples/txt/BB",
    decompose_errors: bool = False
) -> str:
    """
    Main generator interface function: builds Stim circuit and saves to .interface.yaml
    """
    Hx, Hz, Lx, bb_key, default_dist = load_bb_matrices(code_type, search_base)
    if rounds is None or rounds <= 0:
        rounds = default_dist

    if noise_model == "PHENOMENOLOGICAL":
        circuit = generate_bb_code_phenomenological_circuit(
            Hx_matrix=Hx, logical_x_matrix=Lx, rounds=rounds,
            p_data=error_rate, p_meas=error_rate
        )
    elif schedule in ("ibm", "7layer") and bb_key in BB_CODE_DICT:
        ell, m_val, a1, a2, a3, b1, b2, b3 = BB_CODE_DICT[bb_key]
        matchings = get_bb_matchings(ell, m_val, a1, a2, a3, b1, b2, b3)
        num_data = 2 * ell * m_val
        num_checks = ell * m_val
        if schedule == "ibm":
            circuit = _build_ibm_bb_circuit(matchings, num_data, num_checks, Lx, rounds, error_rate)
        else:
            circuit = _build_7layer_bb_circuit(
                matchings, (0, 1, 2), (3, 4, 5), num_data, num_checks, num_checks,
                Lx, rounds, error_rate, error_rate, error_rate, error_rate
            )
    else:
        circuit = generate_bb_code_circuit_level(
            Hx_matrix=Hx, Hz_matrix=Hz, logical_x_matrix=Lx, rounds=rounds,
            p_gate=error_rate, p_meas=error_rate, p_idle=error_rate, p_data=error_rate
        )

    data = {
        "interface": {
            "backend": "stim",
            "decompose_errors": bool(decompose_errors),
            "circuit": str(circuit)
        }
    }

    if output_yaml_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_yaml_path)), exist_ok=True)
        with open(output_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Generated BB interface YAML for code={code_type}, p={error_rate}, rounds={rounds} -> {output_yaml_path}")

    return str(circuit)

def generate_surface_interface_yaml(
    distance: int,
    error_rate: float,
    rounds: int = None,
    output_yaml_path: str = None,
    decompose_errors: bool = False
) -> str:
    """
    Builds Surface code Stim circuit and saves to .interface.yaml
    """
    if rounds is None or rounds <= 0:
        rounds = distance

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=error_rate,
        after_reset_flip_probability=error_rate,
        before_measure_flip_probability=error_rate,
        before_round_data_depolarization=error_rate
    )

    data = {
        "interface": {
            "backend": "stim",
            "decompose_errors": bool(decompose_errors),
            "circuit": str(circuit)
        }
    }

    if output_yaml_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_yaml_path)), exist_ok=True)
        with open(output_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Generated Surface interface YAML for distance={distance}, p={error_rate}, rounds={rounds} -> {output_yaml_path}")

    return str(circuit)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BB and Surface Code Stim Interface YAML Generator")
    parser.add_argument("--code", type=str, default="144", help="BB code type (72, 144, 288, 784) or 'surface'")
    parser.add_argument("-d", "--distance", type=int, default=0, help="Code distance / rounds")
    parser.add_argument("-p", "--error_rate", type=float, default=0.005, help="Physical error rate")
    parser.add_argument("-r", "--rounds", type=int, default=0, help="Number of syndrome measurement rounds")
    parser.add_argument("-o", "--output", type=str, default="generated.interface.yaml", help="Output YAML path")
    parser.add_argument("--noise", type=str, default="CIRCUIT_LEVEL", choices=["CIRCUIT_LEVEL", "PHENOMENOLOGICAL"])
    parser.add_argument("--schedule", type=str, default="ibm", choices=["ibm", "7layer", "greedy"],
                        help="BB circuit: 'ibm' = IBM depth-8 cycle and noise, '7layer' = previous 7-layer circuit, 'greedy' = greedy CNOT layering")
    args = parser.parse_args()

    r = args.rounds if args.rounds > 0 else (args.distance if args.distance > 0 else None)
    if args.code.lower() == "surface":
        dist = args.distance if args.distance > 0 else 3
        generate_surface_interface_yaml(dist, args.error_rate, rounds=r, output_yaml_path=args.output)
    else:
        generate_bb_interface_yaml(args.code, args.error_rate, rounds=r, output_yaml_path=args.output,
                                   noise_model=args.noise, schedule=args.schedule)
