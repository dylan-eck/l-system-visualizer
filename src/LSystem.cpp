#include <map>

#include <glm/gtc/matrix_transform.hpp>

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

void LSystem::generate() {
    std::map<std::string, glm::mat4> varMap;
    for (const auto &[var, trans] : variables) {
        glm::vec3 t = trans.translate;
        glm::vec3 r = trans.rotate;

        glm::mat4 transform =
            glm::translate(glm::mat4(1.0f), t) *
            glm::rotate(glm::mat4(1.0f), r.y, glm::vec3(0, 1, 0)) *
            glm::rotate(glm::mat4(1.0f), r.x, glm::vec3(1, 0, 0)) *
            glm::rotate(glm::mat4(1.0f), r.z, glm::vec3(0, 0, 1));

        varMap.insert({var, transform});
    }

    std::map<std::string, std::string> rulesMap;
    for (const auto &[left, right] : rules) {
        rulesMap.insert({left, right});
    }

    result = axiom;
    for (int i = 0; i < iterationCount; i++) {
        std::string next = "";

        for (const auto &c : result) {
            std::string s{c};
            auto it = rulesMap.find(s);

            if (it != rulesMap.end()) {
                next += rulesMap[s];
            } else {
                next += c;
            }
        }
        result = next;
    }

    glm::mat4 currTransform = glm::mat4(1.0f);
    int vertexCount = 0;
    int indexCount = 0;
    size_t i = 0;

    vertices[vertexCount++] =
        Vertex{.position = {0, 0, 0}, .color = {1, 1, 1, 1}};

    indices[i++] = indexCount;

    std::stack<glm::mat4> stack;

    for (const auto &c : result) {
        if (c == '[') {
            stack.push(currTransform);
        }

        if (c == ']') {
            currTransform = stack.top();
            stack.pop();
            indices[i++] = 0xFFFFFFFF;
        }

        currTransform *= varMap[std::string{c}];
        glm::vec3 currPosition = currTransform * glm::vec4(0, 0, 0, 1);

        if (currPosition != vertices[vertexCount - 1].position) {
            vertices[vertexCount++] =
                Vertex{.position = currPosition, .color = {1, 1, 1, 1}};

            indexCount++;
            indices[i++] = indexCount;
        }
    }

    indexCount = i;
    vertices.resize(vertexCount);
    indices.resize(indexCount);

    glm::vec3 avgPos{0};
    for (const auto &v : vertices) {
        avgPos += v.position;
    }
    avgPos /= vertices.size();

    for (auto &v : vertices) {
        v.position -= avgPos;
    }

    stringStale = false;
    vertsStale = false;
}
} // namespace lsv