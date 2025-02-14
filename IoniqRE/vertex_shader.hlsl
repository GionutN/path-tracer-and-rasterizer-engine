float4 main(float2 pos : POSITION) : SV_POSITION
{
	pos.x *= 0.5625;	// cheap fix for non square image
	return float4(pos.x, pos.y, 0.0, 1.0);
}
