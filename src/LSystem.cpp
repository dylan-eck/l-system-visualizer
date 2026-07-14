#include "LSystem.h"

namespace lsv {
void LSystem::addVariable(std::string character,
                          Transformation transformation) {
    variables.emplace_back(
        LSVariable{.character = character, .transformation = transformation});
}

void LSystem::addRule(std::string left, std::string right) {
    rules.emplace_back(LSRule{.left = left, .right = right});
}

void LSystem::generate() {}
} // namespace lsv