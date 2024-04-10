#include "instruction_set.h"

#ifdef __linux__
#include <sys/auxv.h>
#endif

namespace milvus {
namespace bitset {
namespace detail {
namespace arm {

InstructionSet::InstructionSet() {}

#ifdef __linux__

#if defined(HWCAP_SVE)
bool InstructionSet::supports_sve() {
    const unsigned long cap = getauxval(AT_HWCAP);
    return ((cap & HWCAP_SVE) == HWCAP_SVE);
}
#else 
bool InstructionSet::supports_sve() {
    return false;
}
#endif

#else
bool InstructionSet::supports_sve() {
    return false;
}
#endif

}
}
}
}
