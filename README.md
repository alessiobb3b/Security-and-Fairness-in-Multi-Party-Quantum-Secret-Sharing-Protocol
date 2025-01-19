# Security and Fairness in Multi-Party Quantum Secret Sharing Protocol
 
This repository implements a hybrid quantum-classical cryptographic protocol that integrates quantum Dijkstra-based routing, quantum secret sharing, entanglement swap simulation, and classical AES-based encryption for secure communication.

## Features
- **Quantum Optimized Minimum Search Algorithm (Quantum OQMSA):** A Grover-like quantum algorithm for minimum index search.
- **Quantum Dijkstra Algorithm:** Identifies optimal routes in a network using Quantum OQMSA.
- **Multi-Circuit Quantum Secret Sharing:** Distributes a secret using quantum circuits with support for entanglement swapping over intermediate hops.
- **Classical Cryptography Integration:**
  - AES-GCM encryption for secure communication.
  - Post-quantum cryptographic algorithms for authentication and key exchange (Kyber, McEliece, and Dilithium).
- **Network Topology Visualization:** Displays and highlights network topologies with color-coded routes.

---

## Requirements

The code relies on the following dependencies:

- **Quantum Framework:** Qiskit
- **Post-Quantum Cryptography:** `pqc` (Kyber, McEliece, Dilithium)
- **Classical Cryptography:** `cryptography`
- **Network Visualization:** `networkx`, `matplotlib`
- **Python Standard Libraries:** `math`, `random`, `uuid`, `json`, etc.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/quantum-secret-sharing.git
   cd quantum-secret-sharing
   ```

2. Install the required dependencies:
   ```bash
   pip install qiskit networkx matplotlib cryptography
   ```

3. Ensure the `pqc` library is installed for post-quantum cryptography.

---

## Files

### Main Script: 
- `QSS.py`
  - Entry point for the program.
  - Initializes the environment and executes the Quantum Secret Sharing (QSS) protocol.

### Core Classes:
- `quantumCore`: Implements the polynomial-based quantum secret sharing logic.
- `QuantumOQMSA`: A Grover-inspired minimum search algorithm for Dijkstra.
- `QuantumDijkstraOQMSA`: A quantum version of the classical Dijkstra algorithm.
- `AES256`: Provides AES-GCM encryption and decryption.
- `dealer` & `player`: Represent protocol participants.
- `utilities`: Helper class for network topology visualization.
- `KyberKEM`: Wrapper for Kyber key encapsulation mechanism.
- `toolkit`: General-purpose utilities for quantum circuit handling and result visualization.

---

## Usage

1. Prepare a network topology file `network.json` with the following structure:
   ```json
   {
     "Network Topology": {
       "Dealer": { "Player 1": { "alpha": 1, "beta": 1 }, "Player 2": { "alpha": 2, "beta": 1 } },
       "Player 1": { "Dealer": { "alpha": 1, "beta": 1 }, "Player 2": { "alpha": 1, "beta": 2 } },
       "Player 2": { "Dealer": { "alpha": 2, "beta": 1 }, "Player 1": { "alpha": 1, "beta": 2 } }
     },
     "Network Parameters": { "K": 10 }
   }
   ```

2. Run the main script:
   ```bash
   python main.py
   ```

3. Output will include:
   - Network topology visualization.
   - Partial circuit execution results.
   - Reconstructed secret.

---

## Example Output

- Network topology visualization saved as `Network_Topology.png` and `QSS_involved_Nodes.png`.
- Console logs detailing the execution flow, including:
  - Authentication of players.
  - Quantum Dijkstra results.
  - Measurement results for each circuit.
  - Final secret reconstruction verification.

---

## Additional Considerations
<p align="justify">
An important design choice was the shift of the proposed code example from `mod 7` to `mod 3`. This decision arose primarily due to constraints in the Qiskit simulation environment, as well as the conceptual overhead that arises when faithfully simulating larger modular arithmetic on qubit-based hardware. %Initially, a `mod 7` approach captures the elegance of a prime dimension and the capacity to distribute the secret across a seven-element field. However, the mismatch between prime dimension seven and qubit-based hardware implies that each multi-level operation must be split into subroutines of rotations, partial-phase gates, and amplitude truncations. This leads to an explosion in circuit depth when representing {0,1,2,3,4,5,6} states via two-level qubits. Building amplitude initializations for each of the seven possible states not only proves lengthy and intricate but also suffers from the risk that amplitudes bleed into extraneous computational basis states. Such bleed increases when gates introduce rounding or phase errors, and classical feed-forward readout is often limited, preventing us from conditionally steering each multi-level transition. The sheer difficulty of verifying correct transformations for `mod 7` in qubit form means that each circuit quickly becomes unwieldy, especially once we chain together decoy qubits or partial entanglement swaps for multi-hop routes. Under `mod 3`, a three-element field, we still require partial-phase gate logic and truncated amplitude states, but the overhead of emulating {0,1,2} proves substantially lower. 
The partial-phase rotations, for instance, become $\( \frac{2\pi}{3} \times (S_i + \ell_i) \mod 3 \)$, whereas $\( \frac{2\pi}{7} \times (S_i + \ell_i) \mod 7 \)$.
The latter invites more subtle rounding effects and introduces additional complexity in verifying that the measured output truly lies in the domain $\( \{0, 1, \ldots, 6\} \)$ for each ephemeral qubit.
By downshifting to `mod 3`, the main structural features of the theoretical approach remain intact. We still encode each participant’s share in a partial-phase rotation, manage a truncated amplitude to mimic multi-level states, and embed decoy or entanglement-swap subroutines to replicate multi-hop connectivity. At the same time, we reduce the severity of circuit blowup and amplitude cross terms. We thereby preserve the essence of the paper’s cryptographic scheme—namely that each participant receives a masked contribution to the final secret—while avoiding some of the gate-sequencing tangles that hamper `mod 7` in a purely two-level environment.
These considerations do not diminish the viability or generality of the `mod 7` scenario in principle. Rather, they highlight the pragmatic difficulties of bridging multi-level cryptographic protocols and hardware built around binary qubit logic. Once qubit machines or more refined multi-qubit entanglement topologies become routine, we can return to `mod 7` (or higher dimensions) with fewer design compromises. In the interim, focusing on `mod 3` ensures we still convey the key features of threshold-based secret reconstruction and partial-phase embeddings, while containing the mismatch between the theoretical arithmetic and the classical-quantum hybrid code Qiskit can realistically support.
</p>
---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Acknowledgments

- **Qiskit:** Quantum framework for building and simulating circuits.
- **Cryptography Library:** For AES-GCM encryption and key management.
- **Post-Quantum Cryptography Library:** Kyber, McEliece, and Dilithium implementations.
