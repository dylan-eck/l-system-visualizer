#include <string>

#include <glm/glm.hpp>

#include "RendererTypes.h"

namespace lsv {
struct LSVariable {
    std::string character;
    Transformation transformation;
};

struct LSRule {
    std::string left;
    std::string right;
};

class LSystem {
public:
    bool stringStale = false;
    bool vertsStale = false;

    void addVariable(std::string character, Transformation transformation);
    void addRule(std::string left, std::string right);

    void generate();

    std::vector<LSVariable> variables;
    std::vector<LSRule> rules;
};
} // namespace lsv