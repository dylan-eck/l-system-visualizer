
struct VSInput {
    float3 position;
    float uvX;
    float3 normal;
    float uvY;
    float4 color;
}

struct VSOutput {
    float4 color;
    float4 sv_position : SV_Position;
};

struct PushConstants {
    float4x4 viewProjectionMatrix;
    VSInput *vertexBuffer;
}

[shader("vertex")]
VSOutput vertMain(uint vid: SV_VertexID, uniform PushConstants constants) {
    VSOutput output;
    output.sv_position = mul(constants.viewProjectionMatrix,
                             float4(constants.vertexBuffer[vid].position, 1.0));
    output.color = constants.vertexBuffer[vid].color;
    return output;
}

[shader("fragment")]
float4 fragMain(VSOutput inVert) : SV_Target {
    float4 color = inVert.color;
    return color;
}
