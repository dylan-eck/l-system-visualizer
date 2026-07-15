#include <string>
#include <stack>

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
    bool stringStale = true;
    bool vertsStale = true;

    void addVariable(std::string character, Transformation transformation);
    void addRule(std::string left, std::string right);

    void generate();

    std::string axiom;
    std::vector<LSVariable> variables;
    std::vector<LSRule> rules;
    uint32_t iterationCount;

    std::string result;
    std::vector<Vertex> vertices = std::vector<Vertex>(8192);
    std::vector<uint32_t> indices = std::vector<uint32_t>(8192);
    std::stack<glm::mat4> stack;
};
} // namespace lsv