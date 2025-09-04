cbuffer transform_cbuff
{
	float3x3 normal_mat;
	float4x4 model;
	float4x4 view;
	float4x4 projection;
};

struct vs_out
{
	float4 col : COLOR;
	float4 pos : SV_POSITION;
};

vs_out main(float3 pos : POSITION, float3 norm : NORMAL)
{
	vs_out vs;
	vs.pos = mul(float4(pos, 1.0f), model);
	vs.pos = mul(vs.pos, view);
	vs.pos = mul(vs.pos, projection);

	float3 world_normal = normalize(mul(normal_mat, norm));
	vs.col = float4(0.5f * world_normal + 0.5f, 1.0f);
	return vs;
}
