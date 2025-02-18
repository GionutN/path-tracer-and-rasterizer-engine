// pass-through shader

struct VSOut
{
	float4 position : SV_POSITION;
	float2 texcoord : TEXCOORD0;
};

VSOut main( float4 pos : POSITION, float2 texcoord : TEXCOORD0 )
{
	VSOut output;
	output.position = pos;
	output.texcoord = texcoord;
	return output;
}