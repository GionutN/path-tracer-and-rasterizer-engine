struct vs_out
{
	float4 col : COLOR;
	float4 pos : SV_POSITION;
};

vs_out main(float2 pos : POSITION)
{
	vs_out vs;
	pos *= 2.0;
	vs.col = lerp(float4(0.75, 0.85, 1.0, 1.0), float4(0.65, 0.79, 1.0, 1.0), pos.y);
	vs.pos = float4(pos.x, pos.y, 0.0, 1.0);
	return vs;
}
