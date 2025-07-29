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

	//vs.col = float4(pos.x, pos.y, pos.z, 1.0);
	vs.col = vs.pos;
	vs.col *= 2;
	vs.col = (vs.col + 1) * 0.5;
	vs.col.z = 0.0;
	return vs;
}
