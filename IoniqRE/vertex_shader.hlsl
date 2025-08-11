cbuffer transform_cbuff
{
	matrix model;
	matrix view;
	matrix projection;
};

struct vs_out
{
	float4 col : COLOR;
	float4 pos : SV_POSITION;
};

vs_out main(float3 pos : POSITION)
{
	vs_out vs;
	vs.pos = mul(float4(pos, 1.0), model);
	vs.pos = mul(vs.pos, view);
	vs.pos = mul(vs.pos, projection);
	return vs;
}
