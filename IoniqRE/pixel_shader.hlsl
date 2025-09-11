float4 main(float3 frag_pos : FRAG_POSITION, float3 wn : WORLD_NORMAL) : SV_TARGET
{
	float3 light_color = float3(1.0f, 1.0f, 1.0f);
	float3 ambient_color = float3(0.62f, 0.84f, 1.0f);	// this is the clear color
	float3 albedo = float3(1.0f, 0.0f, 0.0f);
	float3 final_color = float3(0.0f, 0.0f, 0.0f);
	float ambient_strength = 0.2f;

	float3 ambient = ambient_strength * ambient_color;

	float3 light_dir = normalize(float3(1.0f, 0.0f, 1.0f));
	float diffuse = max(dot(-wn, light_dir), 0.0f);
	float3 diffuse_col = diffuse * light_color;

	final_color = (ambient + diffuse_col) * albedo;

	return float4(final_color, 1.0f);
}
