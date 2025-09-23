#include "material.h"

#include "onb.h"

__device__ bool diffuse_uniform::scatter(const ray& r_in, const hit_record& hr, scatter_record* srec, ray* r_out, curandState* local_state) const
{
	// uniform scattering
	*r_out = ray(hr.p + 0.0001f * hr.n, random::on_unit_hemisphere(local_state, hr.n));
	srec->attenuation = m_albedo / pi;
	srec->pdf_val = this->pdf();
	srec->cos_law_weight = fmaxf(0.0f, hr.n.dot3(r_out->direction()));
	return true;
}

__device__ bool oren_nayar::scatter(const ray& r_in, const hit_record& hr, scatter_record* srec, ray* r_out, curandState* local_state) const
{
	onb uvw(hr.n);
	const iqvec wo = -r_in.direction();
	
	// get a new ray scattered with cosine weighting
	*r_out = ray(hr.p + 0.0001f * hr.n, uvw.transform_to_world(random::cosine_weighted(local_state)));
	srec->pdf_val = this->pdf(r_out->direction(), hr.n);

	// if it is perpendicular to the normal, ignore and cast it along the normal
	if (srec->pdf_val < 0.00001f) {
		*r_out = ray(hr.p + 0.0001f * hr.n, hr.n);
		srec->pdf_val = 1 / pi;
	}
	srec->cos_law_weight = fmaxf(0.0f, hr.n.dot3(r_out->direction()));
	const iqvec wi = r_out->direction();

	// constants used to compute the bsdf value
	const float sigma2 = m_sigma * m_sigma;
	const float A = 1.0f - 0.5f * sigma2 / (sigma2 + 0.33f);
	const float B = 0.45f * sigma2 / (sigma2 + 0.09f);

	// azimuthal angles
	const float phi_o = atan2f(wo.y, wo.x);
	const float phi_i = atan2f(wi.y, wi.x);

	// polar angles
	const float costheta_o = fmaxf(0.0f, wo.dot3(hr.n));	// should not be greater than 1, but check anyway
	const float theta_o = costheta_o > 1.0f ? 0.0f : acosf(costheta_o);
	const float costheta_i = fmaxf(0.0f, wi.dot3(hr.n));
	const float theta_i = costheta_i > 1.0f ? 0.0f : acosf(costheta_i);
	const float alpha = fmaxf(theta_i, theta_o);
	const float beta = fminf(theta_i, theta_o);

	const float coeff = A + B * cosf(phi_i - phi_o) * sinf(alpha) * tanf(beta);
	srec->attenuation = m_albedo * coeff / pi;
	return true;

}

__device__ float oren_nayar::pdf(const iqvec& wo, const iqvec& normal) const
{
	return normal.dot3(wo) / pi;
}
