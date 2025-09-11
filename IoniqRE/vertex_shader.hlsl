cbuffer transform_cbuff
{
	float3x3 normal_mat;
	float4x4 model;
	float4x4 view;
	float4x4 projection;
};

struct vs_out
{
	float3 frag_pos : FRAG_POSITION;
	float3 world_normal : WORLD_NORMAL;
	float4 pos : SV_POSITION;
};

vs_out main(float3 pos : POSITION, float3 norm : NORMAL)
{
	vs_out vs;
	vs.pos = mul(float4(pos, 1.0f), model);
	vs.frag_pos = float3(vs.pos.x, vs.pos.y, vs.pos.z);

	vs.pos = mul(vs.pos, view);
	vs.pos = mul(vs.pos, projection);

	vs.world_normal = normalize(mul(normal_mat, norm));
	return vs;
}
