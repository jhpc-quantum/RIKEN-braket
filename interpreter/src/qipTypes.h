#ifndef SQC_TYPES_H
#define SQC_TYPES_H

/// @def MAX_INFO
/// @brief Upper limit number of quantum circuit IR
/// @note Not used because multiple qubit declarations are not supported at this time.
#define MAX_INFO 48

/// @def MAX_N_GATES
/// @brief Upper limit of gate operation information
#define MAX_N_GATES 128

/// @def MAX_I_ARGS
/// @brief Upper limit on integer parameter number
#define MAX_I_ARGS  8

/// @def MAX_R_ARGS
/// @brief Upper limit on the number of floating-point parameters
#define MAX_R_ARGS  8

namespace qip {

typedef int qint;  ///< Quantum bit type

/// \brief enum representing operations in a quantum circuit IR
enum enumGates{
    HGate,
    CXGate,
    CZGate,
    RXGate,
    RYGate,
    RZGate,
    SGate,
    SdgGate,
    XGate,
    U1Gate,
    Measure,
    NGates /// Number of gates
};

/// @brief Structure defining gate operations
/// @details A structure that represents a single gate or other operation of a quantum circuit.
///          What the parameters mean is designed for each operation.
///          or example,
///             for id=_CXGate, niarg=2, rarg=0, iarg[0] is the control qubit number and iarg[1] is the target qubit number.
typedef struct{
  qint   id;                ///< Gate Type（enumGates）
  int    niarg;             ///< Number of integer parameters
  int    nrarg;             ///< Number of floating point number parameters
  qint   iarg[MAX_I_ARGS];  ///< Integer parameters
  double rarg[MAX_R_ARGS];  ///< Floating point number parameters
} gateInfoTy;

/// @brief Structures representing IRs in quantum circuits
/// @note At present, the structure area for representing operations such as gates is reserved for the number of MAX_N_GATES, 
///       and cannot hold more than MAX_N_GATES operations.
typedef struct{
  // --- common parameters ---
  int            nprocs;  ///< Number of processes
  qint           qubits;             ///< Number of quantum bit
  int            ngates;             ///< Number of gate operation information
  gateInfoTy     gate[MAX_N_GATES];  ///< Gate operation information
} qipInfoTy;

typedef qipInfoTy qipIrTy;  ///< Type of quantum circuit IR
}
#endif // SQC_TYPES_H
