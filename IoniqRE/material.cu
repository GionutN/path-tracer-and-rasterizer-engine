#include "material.h"

__device__ bool diffuse::scatter(const ray& r_in, const hit_record& hr, iqvec* attenuation, ray* r_out, curandState* local_state) const
{
	// uniform scattering
	*r_out = ray(hr.p + 0.0001f * hr.n, random::on_unit_hemisphere(local_state, hr.n));
	*attenuation = m_albedo / pi;
	return true;
}
