cbuffer transform_cbuff
{
	matrix transform;
};

float4 main(float2 pos : POSITION) : SV_POSITION
{
	//pos.x *= 0.5625;	// cheap fix for non square image
	float4 out_pos = mul(float4(pos.x, pos.y, 0.0, 1.0), transform);
	return out_pos;
}
