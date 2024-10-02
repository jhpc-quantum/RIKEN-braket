#ifndef SQC_TYPES_H
#define SQC_TYPES_H

/// @def MAX_INFO
/// @brief 量子回路IRの上限数
/// @note 現時点では、複数回のqubit宣言には対応していないため、未使用
#define MAX_INFO 48

/// @def MAX_N_GATES
/// @brief ゲート操作情報数の上限
#define MAX_N_GATES 128

/// @def MAX_I_ARGS
/// @brief 整数パラメタ数の上限
#define MAX_I_ARGS  8

/// @def MAX_R_ARGS
/// @brief 浮動小数点数パラメタ数の上限
#define MAX_R_ARGS  8

namespace qip {

typedef int qint;  ///< 量子ビットの型

/// \brief 量子回路IRでの操作を表すenum
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

/// @brief ゲート操作を定義する構造体
/// @details 量子回路の１つのゲートなどの操作を表現する構造体。
///          パラメータが何を意味するかは、操作ごとに設計する。
///          例えば、id=_CXGateの場合、
///          niarg=2, rarg=0であり、iarg[0]が制御qubit番号、iarg[1]が標的qubit番号となっている。
typedef struct{
  qint   id;                ///< ゲート種別（enumGates）
  int    niarg;             ///< 整数パラメタの数
  int    nrarg;             ///< 浮動小数点数パラメタの数
  qint   iarg[MAX_I_ARGS];  ///< 整数パラメタ
  double rarg[MAX_R_ARGS];  ///< 浮動小数点数パラメタ
} gateInfoTy;

/// @brief 量子回路のIRを表現する構造体
/// @note 現時点では、ゲートなどの操作を表現する構造体領域をMAX_N_GATES数分確保しており、
///       MAX_N_GATES以上の操作を保持することはできない。
typedef struct{
  // --- common parameters ---
  int            nprocs;  ///< プロセス数
  qint           qubits;             ///< 量子ビット数
  int            ngates;             ///< ゲート操作の情報数
  gateInfoTy     gate[MAX_N_GATES];  ///< ゲート操作の情報
} qipInfoTy;

typedef qipInfoTy qipIrTy;  ///< 量子回路IRの型
}
#endif // SQC_TYPES_H
