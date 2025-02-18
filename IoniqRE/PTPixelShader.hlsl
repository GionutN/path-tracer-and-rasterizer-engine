Texture2D tex : register(t0);
SamplerState smpl : register(s0);

float4 main(float4 pos : SV_POSITION, float2 texcoord : TEXCOORD0) : SV_TARGET
{
	return tex.Sample(smpl, texcoord);
}