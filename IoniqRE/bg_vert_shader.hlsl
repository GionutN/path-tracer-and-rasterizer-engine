float4 main(float2 pos : POSITION) : SV_POSITION
{
	pos *= 2.0;
	return float4(pos.x, pos.y, 0.0, 1.0);
}
