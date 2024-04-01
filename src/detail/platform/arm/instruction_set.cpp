#include "instruction_set.h"

#include <sys/auxv.h>

namespace milvus {
namespace bitset {
namespace detail {
namespace arm {

InstructionSet::InstructionSet() {}

bool InstructionSet::supports_sve() {
    const unsigned long cap = getauxval(AT_HWCAP);
    return ((cap & HWCAP_SVE) == HWCAP_SVE);
}

}
}
}
}
