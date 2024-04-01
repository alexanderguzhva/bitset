#pragma once

namespace milvus {
namespace bitset {
namespace detail {
namespace arm {

class InstructionSet {
public:
    static InstructionSet&
    GetInstance() {
        static InstructionSet inst;
        return inst;
    }

private:
    InstructionSet();

public:
    bool supports_sve();
};

}
}
}
}
