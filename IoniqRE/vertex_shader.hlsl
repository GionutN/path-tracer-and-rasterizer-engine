cbuffer transform_cbuff
{
	matrix transform;
};

struct vs_out
{
	float4 col : COLOR;
	float4 pos : SV_POSITION;
};

vs_out main(float3 pos : POSITION)
{
	vs_out vs;
	vs.pos = mul(float4(pos.x, pos.y, 0.0, 1.0), transform);
	vs.pos.x *= 0.5625;	// cheap aspect ratio correction fix
	return vs;
}
