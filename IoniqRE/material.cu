#include "material.h"

#include "onb.h"

__device__ bool diffuse_uniform::scatter(const ray& r_in, const hit_record& hr, scatter_record* srec, ray* r_out, curandState* local_state) const
{
	// uniform scattering
	*r_out = ray(hr.p + 0.0001f * hr.n, random::on_unit_hemisphere(local_state, hr.n));
	srec->attenuation = m_albedo / pi;
	srec->pdf_val = this->pdf(r_out->direction());
	srec->cos_law_weight = fmaxf(0.0f, hr.n.dot3(r_out->direction()));
	return true;
}

__device__ bool diffuse_lambertian::scatter(const ray& r_in, const hit_record& hr, scatter_record* srec, ray* r_out, curandState* local_state) const
{
	// write a cos weighted random direction
	onb uvw(hr.n);
	*r_out = ray(hr.p + 0.0001f * hr.n, uvw.transform(random::cosine_weighted(local_state)));
	srec->attenuation = m_albedo / pi;
	srec->pdf_val = uvw.w().dot3(r_out->direction()) / pi;
	srec->cos_law_weight = fmaxf(0.0f, hr.n.dot3(r_out->direction()));

	// generate a new ray if the first one is perpendicular to the normal
	if (srec->pdf_val < 0.00001f) {
		*r_out = ray(hr.p + 0.0001f * hr.n, hr.n);
		srec->pdf_val = 1 / pi;
		srec->cos_law_weight = 1.0f;
	}
	return true;
}
