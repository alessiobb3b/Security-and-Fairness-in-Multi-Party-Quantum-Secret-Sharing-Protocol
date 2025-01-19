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

## Additional COnsiderations

Yet many protocols—particularly those involving modular arithmetic beyond a two-level system—naturally call for `qudit` hardware or advanced multi-qubit encodings. Our approach necessarily re-casts all higher dimensional arithmetic (for instance, the mod d computations) into routines that can be implemented with standard two-level qubits. This leads to truncated amplitude embeddings, partial-phase gates, and repeated ephemeral states. Although these strategies capture the conceptual essence of `mod d` secret sharing, they present notable overhead on real hardware and in large-scale simulations. A full hardware demonstration of multi-level quantum secret sharing would be constrained by device qubit quality, the requirement for multiple re-initializations, and difficulty in performing feed-forward classical corrections on ephemeral states.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Acknowledgments

- **Qiskit:** Quantum framework for building and simulating circuits.
- **Cryptography Library:** For AES-GCM encryption and key management.
- **Post-Quantum Cryptography Library:** Kyber, McEliece, and Dilithium implementations.
