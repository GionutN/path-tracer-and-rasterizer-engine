cbuffer transform_cbuff
{
	// TODO: add a normal model matrix
	matrix model;
	matrix view;
	matrix projection;
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

	vs.col = float4(0.5f * norm + 0.5f, 1.0f);
	return vs;
}
