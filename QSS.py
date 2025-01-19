##############################################################################
# This code represents a simple quantum-based protocol integrating:
#  - A quantum Dijkstra approach (via QuantumOQMSA)
#  - A multi-circuit approach for quantum secret sharing
#  - An entanglement swap simulation for multi-hop routes
#  - Classical cryptographic authentication and AES encryption
#
# The classes below demonstrate how a Dealer authenticates Players and then
# runs a multi-circuit quantum routine for sharing a secret. The "quantumCore"
# class (previously MUB-like) holds core polynomial math for secret sharing, 
# while the "dealer" organizes the entire protocol. 
# The "player" receives partial secrets and transmits their measured result 
# back to the dealer in an AES-encrypted manner.
##############################################################################

from math import gcd, sqrt
from qiskit import *
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.circuit import QuantumRegister, Qubit, Barrier
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.quantum_info import Statevector, DensityMatrix, Pauli
from qiskit.visualization import plot_bloch_multivector, circuit_drawer, plot_state_paulivec, plot_state_qsphere, plot_state_city
from qiskit.circuit.library import QFT, PhaseEstimation, PauliEvolutionGate
from qiskit.visualization import plot_histogram
import warnings, hmac, hashlib, struct, base64, os, random, uuid, json, math
import networkx as nx
import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    from pqc.kem import kyber512 as kyber  # Kyber: Post-quantum KEM
from pqc.kem import mceliece6960119 as kemalg  # McEliece: Another post-quantum KEM
from matplotlib import pyplot as plt
from pqc.sign import dilithium2 as dilithium  # Post-quantum signature (Dilithium)
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes  # AES encryption
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC  # PBKDF2 Key derivation
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

##############################################################################
# UTILITY CLASSES
##############################################################################

class utilities:
    """
    Provides helper functions for drawing network topologies in the form
    of graphs. Also, a function to visually highlight the chosen 't' nodes
    and their intermediate path nodes once the quantum Dijkstra is complete.
    """
    def drawNetworkTopology(self, network):
        """
        Draws the entire network as a simple undirected graph. 
        Each node is placed via spring_layout, with edges extracted from 'network'.
        """
        G = nx.Graph()
        # Add edges from adjacency dictionary
        for node, neighbors in network.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)

        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G)
        nx.draw(
            G, pos, with_labels=True, node_size=700, node_color="skyblue",
            font_size=12, font_color="black", edge_color="gray"
        )
        plt.title("Network Topology from Dealer's Prospective", fontsize=16)
        plt.savefig("Network_Topology.png", format='png', dpi=300)
        plt.close()

    def drawColoredNetworkTopology(self, distances, predecessors, t):
        """
        Draws a directed graph from 'predecessors', marking up to 't' chosen nodes
        plus any intermediate nodes discovered along their path from 'Dealer'.
        Then returns a dictionary counting how many intermediate nodes exist for each chosen node.
        """
        G = nx.DiGraph()
        
        # For each node that has a predecessor, add an edge with weight (dist[node] - dist[pred])
        for node, pred in predecessors.items():
            if pred is not None:
                weight = distances[node] - distances[pred]
                G.add_edge(pred, node, weight=round(weight, 3))

        # We'll color certain nodes:
        node_colors = {}
        first_t_nodes = list(distances.keys())[: t + 1]  # 'Dealer' plus first 't' nodes
        intermediate_counts = {}
        intermediate_nodes = set()

        # Count how many intermediate nodes are encountered from each chosen node (beyond 'Dealer')
        for node in first_t_nodes[1:]:
            current = node
            count = 0
            while predecessors[current] and predecessors[current] != 'Dealer':
                intermediate_nodes.add(predecessors[current])
                count += 1
                current = predecessors[current]
            intermediate_counts[node] = count

        # Assign colors to nodes
        for node in G.nodes:
            if node in first_t_nodes:
                node_colors[node] = "green"       # The 'Dealer' plus first 't' nodes
            elif node in intermediate_nodes:
                node_colors[node] = "yellow"      # Intermediate nodes in path
            else:
                node_colors[node] = "skyblue"     # All others

        # Drawing the entire directed graph with node colors
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G)
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=700,
            node_color=[node_colors[node] for node in G.nodes],
            font_size=12,
            font_color="black",
        )

        # Put the weight labels on the edges
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color="red")

        plt.title("Correct Weighted Network Topology with Highlights", fontsize=16)
        plt.savefig("QSS_involved_Nodes.png", format='png', dpi=300)
        plt.close()

        # Return counts of intermediate nodes for each chosen node
        return intermediate_counts


##############################################################################
# QUANTUM OQMSA (Grover-like minimum search) & QUANTUM DIJKSTRA
##############################################################################

class QuantumOQMSA:
    """
    A conceptual version of an Optimized Quantum Minimum Search Algorithm,
    using a Grover-like process to identify the minimal index in 'dist_array'.
    """
    def __init__(self, dist_array):
        self.dist_array = dist_array
        # 'n' is the power-of-two exponent. 2^n must match len(dist_array).
        self.n = int(math.log2(len(dist_array)))
        if 2**self.n != len(dist_array):
            raise ValueError("Length of dist_array must be a power of 2.")

    def __findIndexOfMinimumClassically(self):
        """
        Classically find the index with min distance 
        (not the final approach if purely quantum, but for demonstration).
        """
        return int(np.argmin(self.__getDistArray()))
    
    def __getN(self):
        return self.n
    
    def __getDistArray(self):
        return self.dist_array

    def __buildOracle(self, qc, min_index):
        """
        Mark the 'min_index' state so that Grover's algorithm can invert it.
        """
        bin_str = format(min_index, f'0{self.n}b')
        for i, bit in enumerate(reversed(bin_str)):
            if bit == '0':
                qc.x(i)
        if self.__getN() == 1:
            qc.z(0)
        else:
            qc.h(self.__getN() - 1)
            qc.mcx(list(range(self.__getN() - 1)), self.__getN() - 1)
            qc.h(self.__getN() - 1)
        for i, bit in enumerate(reversed(bin_str)):
            if bit == '0':
                qc.x(i)

    def __buildDiffusionOperator(self, qc):
        """
        Apply the standard Grover diffusion (inversion-about-average) operator.
        """
        for i in range(self.__getN()):
            qc.h(i)
            qc.x(i)
        if self.__getN() == 1:
            qc.z(0)
        else:
            qc.h(self.__getN()-1)
            qc.mcx(list(range(self.__getN()-1)), self.__getN()-1)
            qc.h(self.__getN()-1)
        for i in range(self.__getN()):
            qc.x(i)
            qc.h(i)

    def runOQMSA(self):
        """
        Actually build the circuit for a small number of Grover iterations,
        measure, and interpret the outcome as the minimal index.
        """

        # Classically identify the min index
        min_index = self.__findIndexOfMinimumClassically()

        # Build the quantum circuit with n qubits and n classical bits
        qc = QuantumCircuit(self.__getN(), self.__getN())

        # Step 1: Initialize in uniform superposition
        for i in range(self.__getN()):
            qc.h(i)

        # Choose ~ sqrt(2^n) iterations
        iterations = int(math.floor(math.pi/4 * math.sqrt(2**self.__getN())))
        for _ in range(iterations):
            self.__buildOracle(qc, min_index)
            self.__buildDiffusionOperator(qc)

        # Measure to classical bits
        qc.measure(range(self.__getN()), range(self.__getN()))

        # Execute circuit in classical fallback style
        xValue, y, counts = toolkit().getInstance().computeResults(
            BasicSimulator().run(transpile(qc, backend=BasicSimulator()), shots=1000).result().get_counts(qc),
            9, 1000
        )
        #toolkit().getInstance().getCircuit(qc, False)

        # The highest-likelihood bitstring is our found index
        measured_bin = max(counts, key=counts.get)
        found_index = int(measured_bin, 2)
        return found_index, counts

class QuantumDijkstraOQMSA:
    """
    A "Quantum" Dijkstra-like approach that uses the QuantumOQMSA to pick the
    minimal-distance node among the unvisited set. 
    """
    def __init__(self, graph):
        self.graph = graph
        self.vertices = list(graph.keys())
    
    def __getGraph(self):
        return self.graph
    
    def __getVertices(self):
        return self.vertices

    def __quantumMinIndex(self, dist_dict, Q):
        """
        Transform 'dist_dict' restricted to unvisited nodes into
        an array for the OQMSA. Then pick the found index as minimal.
        """
        unvisited_list = list(Q)
        size = len(unvisited_list)
        n = 1
        while (1 << n) < size:
            n += 1
        array_size = 1 << n

        # Build the dist array and pad
        dist_array = []
        for i in range(size):
            dist_array.append(dist_dict[unvisited_list[i]])
        while len(dist_array) < array_size:
            dist_array.append(999999999)

        # Use the quantum OQMSA
        oqmsa = QuantumOQMSA(dist_array)
        found_idx, counts = oqmsa.runOQMSA()
        if found_idx >= size:
            real_min_idx = min(range(size), key=lambda i: dist_array[i])
            return unvisited_list[real_min_idx]
        else:
            return unvisited_list[found_idx]

    def run(self, source):
        """
        Classical Dijkstra structure, but the selection of the minimal-dist node 
        among the unvisited is replaced by the quantumMinIndex logic above.
        Returns (dist, prev) as normal.
        """
        dist = {v: float('inf') for v in self.__getVertices()}
        prev = {v: None for v in self.__getVertices()}
        dist[source] = 0
        Q = set(self.__getVertices())

        while Q:
            u = self.__quantumMinIndex(dist, Q)
            Q.remove(u)
            for v, cost_uv in self.__getGraph()[u].items():
                alt = dist[u] + cost_uv
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
        return dist, prev

##############################################################################
# AES256: Symmetric encryption for exchanging messages
##############################################################################
class AES256:
    """
    Provides AES-GCM encryption and decryption. 
    The 'seed' acts as the password for deriving the cryptographic key.
    """
    def __init__(self, seed=""):
        self.seed = seed

    def setSeed(self, seed):
        self.seed = seed

    def __getSeed(self):
        return self.seed.encode() if isinstance(self.seed, str) else self.seed

    def _derive_key(self, salt: bytes) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend(),
        )
        return kdf.derive(self.__getSeed())

    def encrypt(self, plaintext: str) -> str:
        salt = os.urandom(16)
        key = self._derive_key(salt)
        iv = os.urandom(12)
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(plaintext.encode()) + encryptor.finalize()
        encrypted_message = base64.b64encode(salt + iv + encryptor.tag + ciphertext).decode()
        return encrypted_message

    def decrypt(self, encrypted_message: str) -> str:
        data = base64.b64decode(encrypted_message)
        salt = data[:16]
        iv = data[16:28]
        tag = data[28:44]
        ciphertext = data[44:]
        key = self._derive_key(salt)
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend())
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        return plaintext.decode()

##############################################################################
# The 'init' class to build environment and authenticate a certain number of players
##############################################################################

class init:
    def initializeEnvironment(self, players, network):
        """
        Creates 'players' number of Player objects and a single 'dealer' object,
        authenticates them, sets up the network topology, and returns the dealer.
        """
        print("[!] Initializing environment")
        print(f"[!] Programmed setup involves one Delaer and {players} players")

        with open(network, 'r') as file:
            network = json.load(file)

        # Generate Dealer public/private key
        dealerPubKey, DealerPrivKey = kyber.keypair()
        protocolDealer = dealer(dealerPubKey, DealerPrivKey)

        # Create each Player, authenticate them
        for i in range(players):
            print(f"\t[-] Authenticating and aggregating Player {i+1} to protocol network")
            pub, priv = kyber.keypair()
            currentPlayer = player(pub, priv, protocolDealer.getPublicKey(), f"Player {i+1}")
            protocolDealer.authenticatePlayer(currentPlayer)
            currentPlayer.setNeighbors(network["Network Topology"][f"Player {i+1}"])

        print("[!] Authentication completed. Network is now operational and fully authenticated")

        # Link the dealer to its neighbors, set entire network, draw the topology
        protocolDealer.setNeighbors(network["Network Topology"]["Dealer"])
        protocolDealer.setNetworkTopology(network["Network Topology"])
        utilities().drawNetworkTopology(protocolDealer.getNetworkTopology())

        # Retrieve the K parameter from the JSON
        protocolDealer.setK(network["Network Parameters"]["K"])
        print("[!] Network topology has been identified by the Dealer and each Player will now acknowledge their neighbors")

        return protocolDealer
    

class KyberKEM():
    """
    Minimal wrapper for keypair, encap, decap 
    referencing the PQC 'kyber' object already imported.
    """
    def generateKeypair(self):
        return kyber.keypair()

    def encap(self, public_key):
        return kyber.encap(public_key)

    def decap(self, ciphertext, private_key):
        return kyber.decap(ciphertext, private_key)

    def getAlgoName(self):
        return "CRYSTALS-Kyber"

##############################################################################
# Player/Dealer Implementation
##############################################################################

class player:
    """
    Each 'player' has a public/priv key, a shared key with the Dealer (once authenticated),
    an AES object, and neighbor references in the network. It can store and send 
    measurement results back to the Dealer via storeMeasurementAndSendToDealer.
    """
    def __init__(self, pubKey, privKey, dealerPubKey, name):
        self.publicKey = pubKey
        self.privateKey = privKey
        self.dealerPubKey = dealerPubKey
        self.aes = AES256("")
        self.neighbors = {}
        self.sharedKey = None
        self.name = name
    
    def getPublicKey(self):
        return self.publicKey

    def getName(self):
        return self.name

    def __getAES(self):
        return self.aes
    
    def __setSharedKey(self, sharedKey):
        self.sharedKey = sharedKey

    def __getSharedKey(self):
        return self.sharedKey

    def __getDealerPubKey(self):
        return self.dealerPubKey
    
    def setNeighbors(self, neighbors):
        self.neighbors = neighbors
    
    def authenticationProcedure(self, dealerPubKey, procedure, message = None):
        """
        4-step exchange that eventually yields a new ephemeral shared key.
        We rely on the 'match procedure' to handle each step:
          0 => create ephemeral key with Dealer
          1 => check for Nonce + Player Name
          2 => check for "OK" + Player Info
          3 => finalize with new ephemeral key
        """
        message = self.__getAES().decrypt(message) if message is not None else None
        match procedure:
            case 0:
                assert self.__getDealerPubKey() == dealerPubKey
                sharedKey, ciphertext = kyber.encap(dealerPubKey)
                self.__setSharedKey(sharedKey)
                self.__getAES().setSeed(sharedKey)
                return ciphertext
            case 1:
                return self.__getAES().encrypt(message + self.getName())
            case 2:
                assert message[len(message)-2:] == "OK"
                return self.__getAES().encrypt(message[:36] + "Player Information Data...")
            case 3:
                newKey = message[36:]
                message = self.__getAES().encrypt(message[:36] + "OK")
                self.__getAES().setSeed(self.__getSharedKey())
                self.__setSharedKey(newKey)
                return message

    def storeMeasurementAndSendToDealer(self, measuredVal, dealerObj):
        """
        After measuring the player's partial circuit, store the result 
        into an AES-encrypted message and pass it to the Dealer.
        """
        self.__getAES().setSeed(self.__getSharedKey())
        encM = self.__getAES().encrypt(str(measuredVal))
        dealerObj.receiveMeasurementFromPlayer(self, encM)

class dealer:
    """
    The 'dealer' organizes the entire quantum secret sharing process,
    authenticates players, and reconstructs the final shared secret.
    """
    def __init__(self, pubKey, privKey):
        self.publicKey = pubKey
        self.privateKey = privKey
        self.aes = AES256("")
        self.neighbors = {}
        self.networkTopology = {}
        self.sharedKeys = {}
        self.quantumCore = quantumCore()  # Main QSS logic stored here
        self.K = None
        self.measuredDict = {}

    def getPublicKey(self):
        return self.publicKey

    def __getPlayersSharedKeys(self):
        return self.sharedKeys
    
    def setNeighbors(self, neighbors):
        self.neighbors = neighbors
    
    def getNetworkTopology(self):
        return self.networkTopology
    
    def setNetworkTopology(self, network):
        self.networkTopology = network

    def editNetworkTopology(self, key, value):
        self.networkTopology[key] = value
    
    def setK(self, k):
        self.K = k

    def __getK(self):
        return self.K

    def __getAES(self):
        return self.aes
    
    def setSharedKey(self, playerObj, key):
        self.sharedKeys[playerObj] = key

    def __getSharedKey(self, playerObj):
        return self.sharedKeys[playerObj]

    def __editSharedKey(self, playerObj, key):
        self.sharedKeys[playerObj] = key

    def __getQuantumCore(self):
        return self.quantumCore

    def __getPrivateKey(self):
        return self.privateKey
    
    def authenticatePlayer(self, playerObj):
        """
        Perform the 4-step handshake with the player to finalize 
        a new shared key.
        """
        # Step 0: encapsulate
        self.setSharedKey(playerObj, kyber.decap(playerObj.authenticationProcedure(self.getPublicKey(), 0), self.__getPrivateKey()))
        self.__getAES().setSeed(self.__getSharedKey(playerObj))

        # Step 1: Nonce + check with player's name
        nonce = str(uuid.uuid4())
        message = self.__getAES().encrypt(nonce)
        actualStep = playerObj.authenticationProcedure(self.getPublicKey(), 1, message)
        assert nonce == self.__getAES().decrypt(actualStep)[:36] and self.__getAES().decrypt(actualStep)[36:] == playerObj.getName()

        # Step 2: "OK" 
        nonce = str(uuid.uuid4())
        message = self.__getAES().encrypt(nonce + "OK")
        actualStep = playerObj.authenticationProcedure(self.getPublicKey(), 2, message)
        assert nonce == self.__getAES().decrypt(actualStep)[:36]

        # Step 3: finalize ephemeral key
        nonce = str(uuid.uuid4())
        newShared = str(uuid.uuid4())
        message = self.__getAES().encrypt(nonce + newShared)
        actualStep = playerObj.authenticationProcedure(self.getPublicKey(), 3, message)
        assert nonce == self.__getAES().decrypt(actualStep)[:36] and self.__getAES().decrypt(actualStep)[36:] == "OK"
        self.__editSharedKey(playerObj, newShared)
        self.__getAES().setSeed(self.__getSharedKey(playerObj))

    def __identifySuitablePlayers(self):
        """
        Runs the quantum Dijkstra to find the minimal route cost 
        from 'Dealer' to every other node, returning dist,prev 
        then sorts them so the first t are selected.
        """
        print("[!] Now running Quantum-Dijkstra to identify most suitable Assets")
        for key in self.getNetworkTopology().keys():
            newDist = {}
            for node in self.getNetworkTopology()[key].keys():
                alpha = self.getNetworkTopology()[key][node]["alpha"]
                beta = self.getNetworkTopology()[key][node]["beta"]
                newDist[node] = round((self.__getK()/alpha) + (self.__getK() * beta), 2)
            self.editNetworkTopology(key, newDist)

        dist, prev = QuantumDijkstraOQMSA(self.getNetworkTopology()).run(source='Dealer')
        dist = dict(sorted(dist.items(), key=lambda item: item[1]))
        return dist, prev

    def receiveMeasurementFromPlayer(self, playerObj, encMessage):
        """
        Decrypt an incoming partial measurement from a player, 
        store it in measuredDict.
        """
        self.__getAES().setSeed(self.__getSharedKey(playerObj))
        decrypted = self.__getAES().decrypt(encMessage)
        val = int(decrypted)
        self.measuredDict[playerObj] = val
        print(f"[!] Received M_i = {val} from Player {playerObj.getName()}")

    def runProtocol(self):
        """
        1) Identify the best 't' players with quantumDijkstra
        2) Draw the network with color-coded intermediate nodes
        3) Build a multi-circuit for each chosen player
        4) Reconstruct final secret from partial measurements
        """
        dist, prev = self.__identifySuitablePlayers()
        intermediateNodes = utilities().drawColoredNetworkTopology(dist, prev, self.__getQuantumCore().getT())

        # from the sorted dist, pick next (t) players
        playersList = []
        for key in list(dist.keys())[1 : self.__getQuantumCore().getT()+1]:
            for p in self.sharedKeys:
                if p.getName() == key:
                    playersList.append(p)

        print("[!] Protocol Players have been chosen, now starting Quantum Secret Sharing")

        # Build multi-circuit for each chosen player
        self.__getQuantumCore().runQSSMultiCircuit(playersList, self, intermediateNodes)

        # Finally, once all M_i are in measuredDict, do final sum-l_i mod d
        dVal = self.__getQuantumCore().getD()
        Gx0Vals = []
        for i in range(self.__getQuantumCore().getT()):
            Gx0Vals.append(self.__getQuantumCore().polynomialG(self.__getQuantumCore().xs[i], 0))
        Svals = []
        for i in range(self.__getQuantumCore().getT()):
            Svals.append(self.__getQuantumCore().computeSi(Gx0Vals[i], i))
        sumS = sum(Svals) % dVal

        # We pick offsets [1,2,0] for demonstration
        lVals = [1,2,0]
        MCollect = []
        for i,ply in enumerate(playersList):
            MCollect.append(self.measuredDict.get(ply,0))

        recS = (sum(MCollect) - sum(lVals)) % dVal
        print(f"[!] Reconstructed => {recS}, expected => {self.__getQuantumCore().secret}")
        if recS == self.__getQuantumCore().secret:
            print("[!] QSS success with multi-circuit approach + entanglement swap!")
        else:
            print("[!] QSS mismatch!")
        return recS == self.__getQuantumCore().secret


class toolkit():
    """
    Provides general-purpose helper methods for:
     - computing results after a Qiskit simulation,
     - storing circuit images,
     - rendering outcome histograms, etc.
    """
    def __init__(self):
        self.__instance = None

    def initializeIBMSim(self):
        #Initaliazes an adavnced simulator within the external support of IBM quantum simulator
        QiskitRuntimeService.save_account(channel="ibm_quantum", token="REDACTED", overwrite=True)
        service = QiskitRuntimeService()
        return service.least_busy(simulator=False, operational=True)
    
    def getInstance(self):
        self.__instance = toolkit() if self.__instance is None else self.__instance
        return self.__instance
    
    def __getBinary(self, n):
        """
        Given an integer n, returns a list of binary strings from 0..(2^n -1), 
        zero-padded to length n.
        """
        res = [str(bin(2**n-1))[2:]]
        l = len(res[0])
        for i in range(2**n-1):
            t = str(bin(i))[2:]
            t = ('0' * (l - len(t))) + t
            res.insert(i, t)
        return res
    
    def __filterValues(self, result, n):
        """
        Adjust a result dictionary to keep only the final n bits 
        (some circuits might measure more than n bits).
        """
        t = {}
        for k in result.keys():
            t[k[-n:]] = result[k]
        return t

    def computeResults(self, result, n, attempts, xValue = None):
        """
        Takes the 'result' dictionary from Qiskit (bitstring->counts),
        ensures we only keep the last 'n' bits, then normalizes to get probabilities.
        """
        y = []
        xValue = self.__getBinary(n) if xValue is None else xValue
        if (len(list(result.keys())[0]) != n):
            result = self.__filterValues(result, n)
        for k in xValue:
            y.append(result[k] / attempts) if k in result.keys() else y.append(0)
        return xValue, y, result

    def drawer(self, title, xLabel, yLabel, xValue, yValue, toDraw = False, enlarge = False):
        """
        Quick helper to bar-plot the result distribution.
        """
        plt.cla()
        plt.title(title)
        if enlarge:
            plt.figure().set_figwidth(20)
        plt.bar(xValue, yValue, color='g', align='center', width=0.5)
        plt.grid(color = 'grey', linestyle = '--', linewidth = 0.3)
        for i in range(len(xValue)):
            plt.text(i, yValue[i], yValue[i], ha = 'center')
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.legend()
        plt.show() if toDraw else plt.savefig(title + ".png")

    def dumpResults(self, results, dimension):
        """
        Possibly a function to reorder or slice the results. 
        We keep it to not remove any lines from original code.
        """
        results = list(results.keys())[0][::-1]
        key = results[0 : dimension*2]
        key = key[::2]
        value = results[(dimension*2) : len(results)]
        return key[::-1], value

    def getCircuit(self, circuit, toDraw = False, var = ""):
        """
        Save the circuit in LaTeX format for archival or debugging. 
        """
        for e in ["latex"]:
            if toDraw:
                circuit.draw(output = e)
                input("Press any key to continue...")
            else:
                fName = circuit.name + "_" + e + var + ".png"
                circuit_drawer(circuit, filename=fName, output=e)
                print("File: " + fName + ", was correcly saved!")

##############################################################################
# quantumCore: Holds polynomial-based secret sharing logic and multi-circuit QSS
##############################################################################

class quantumCore():
    """
    This class encapsulates the main Quantum logic, storing:
      - dimension d
      - threshold t
      - secret value
      - polynomial coefficients a_coef for the sharing math
    and a function 'runQSSMultiCircuit' that builds a small 3-qubit circuit for 
    each player, embedding the share + offset, plus simulating an entanglement 
    swap as needed (conceptually).
    """
    def __init__(self, d=3, t=3, secret=1):
        self.d=d
        self.t=t
        self.secret=secret
        self.a_coef=[
            [1,2,0],
            [0,1,1],
            [2,1,0]
        ]
        self.xs=[1,2,0]
        print(f"[!] init => S={secret}, d={d}")
        print(f"\t[-]Secret to Share: {self.a_coef[0][0]}")

    def getD(self):
        return self.d
    def __getXS(self):
        return self.xs
    def getT(self):
        return self.t
    def __getS(self):
        return self.secret
    def __getACoef(self):
        return self.a_coef

    def egcd(self, a, b):
        if a==0:
            return (b,0,1)
        g,y,x=self.egcd(b%a,a)
        return (g,x-(b//a)*y,y)

    def modInv(self, a,m):
        g,x,y=self.egcd(a,m)
        if g!=1:
            raise ValueError("No inverse mod")
        return x%m

    def polynomialG(self, x, y):
        val=0
        for i in range(self.getT()):
            for j in range(self.getT()):
                val += ( self.__getACoef()[i][j] * pow(x,i,self.getD()) * pow(y,j,self.getD()) )
        return val%self.getD()

    def computeSi(self, Gx0, i):
        val= Gx0%self.getD()
        for j in range(self.getT()):
            if j!=i:
                num=self.__getXS()[j]%self.getD()
                den=(self.__getXS()[j]-self.__getXS()[i])%self.getD()
                dinv=self.modInv(den,self.getD())
                val=(val*(num*dinv%self.getD()))%self.getD()
        return val

    def __simulateEntanglementSwap(self, intermediateNodeCount, s, l, d, decoy):
        newNumQubits = (1 + intermediateNodeCount) * (1 + decoy)
        entangleSet = []
        for i in range(decoy+1):
            entangleSet.append((intermediateNodeCount + 1)*i)
        # 2) Create the new circuit with 'newNumQubits' qubits, each with 
        # the same number of classical bits for measurement (since we measure them all eventually).
        circuit = QuantumCircuit(newNumQubits, newNumQubits)

        # We'll do the chain:
        # entangle qubit(0) with qubit(1), then qubit(1) with qubit(2), ..., 
        # up to qubit(intermediateNodeCount-1) with qubit(intermediateNodeCount),
        # so that the final data is at qubit(intermediateNodeCount). 
        # This is a minimal EPR creation logic.

        amp0 = sqrt(2.0/3.0)
        amp1 = sqrt(1.0/3.0)
        circuit.initialize([amp0, amp1], [0])

        # Step2: embed share + offset in a partial phase
        tot = (s+l)%d
        angle = 2.0*math.pi*(tot)/d
        circuit.p(angle,0)

        decoys=[]
        for qb in entangleSet[1:]:
            basisChoice = random.choice(['Z','X'])
            bitChoice   = random.choice([0,1])
            if basisChoice=='Z':
                if bitChoice==1:
                    circuit.x(qb + intermediateNodeCount)
            else:
                circuit.h(qb + intermediateNodeCount)
                if bitChoice==1:
                    circuit.z(qb + intermediateNodeCount)
            decoys.append((qb + intermediateNodeCount,basisChoice,bitChoice))

        for (qb,basis,bit) in decoys:
            if basis=='X':
                circuit.h(qb)

        # For each swap:
        for i in range(intermediateNodeCount):
            for k in entangleSet:
                circuit.barrier()
                source = k + i
                target = k + i + 1
                # Put source in superposition
                circuit.h(source)
                # Then CNOT source->target
                circuit.cx(source, target)
                # We can measure out 'source' if we want the entanglement to move forward 
                # or skip measuring out to keep a chain of entanglements. 
                # If measure was 1 => apply X on target
                circuit.x(target).c_if(source, 1)
                circuit.barrier()

        for qb in entangleSet:
            circuit.measure([qb + intermediateNodeCount], [qb + intermediateNodeCount])
        
        return circuit, decoys

    def runQSSMultiCircuit(self, playersArr, dealerObj, intermediateNodes, decoy=2, shots=1):
        """
        For each chosen player, build a small circuit with 3 qubits: 
          qubit0 => main share
          qubits1,2 => decoys + potential BSM for entanglement swap
        Then do measure. If intermediateNodes says we have a certain 
        number of intermediate hops, we do that many 'simulateEntanglementSwap'.
        """
        Gx0Vals = []
        for i in range(self.getT()):
            Gx0Vals.append(self.polynomialG(self.__getXS()[i], 0))
        Svals = []
        for i in range(self.getT()):
            Svals.append(self.computeSi(Gx0Vals[i], i))

        # Offsets for demonstration
        Lvals = [1,2,0]

        # Build each player's circuit
        for i, ply in enumerate(playersArr):
            last = 1 +decoy

            # Step3: if intermediateNodes dict says we have some hops, do that many swaps

            circuit, decoys = self.__simulateEntanglementSwap(intermediateNodes[ply.getName()], Svals[i], Lvals[i], self.getD(), decoy)
            last += intermediateNodes[ply.getName()]
            
            circuit.name = f"Circuit_Player_{ply.getName()}"

            # Actually run the circuit
            kit = toolkit().getInstance()
            kit.getCircuit(circuit, False)
            rawCounts = BasicSimulator().run(  #toolkit().getInstance().initializeIBMSim().run
                transpile(circuit, backend=BasicSimulator()), shots=shots
            ).result().get_counts(circuit)
            print(f"\t[-] Measured values for {ply.getName()}: {rawCounts}")

            best_str = max(rawCounts, key=rawCounts.get)
            mainBit = best_str[2]
            M_i = int(mainBit)

            # Decoy mismatch check
            for (qb,bchoice,bval) in decoys:
                measure = best_str[2-qb]
                if int(measure)!=bval:
                    print(f"[!] Decoy mismatch => Player i={i+1}, qb={qb}, expect={bval}, got={measure}")

            print(f"[!] Built circuit for {ply.getName()}, measured M_i={M_i}")
            ply.storeMeasurementAndSendToDealer(M_i, dealerObj)

        print("[!] All partial circuits done. Dealer will reconstruct.")

# Entry point
if __name__ == '__main__':
    # Build environment with 5 players, load 'network.json', 
    # then run the QSS protocol from the Dealer's perspective.
    res = {True: 0, False: 0}
    for i in range (1):
        protocolDealer = init().initializeEnvironment(5, "network.json")
        res[protocolDealer.runProtocol()] += 1
    print(res)