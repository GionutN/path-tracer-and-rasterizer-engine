#include "scene.h"

#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>

#include "renderer.h"

scene::scene()
{
	m_model_types.fill(0);
}

void scene::add_mesh(const std::string& name, const mesh& m)
{
	auto result = m_meshes.emplace(name, m);
	if (!result.second) {
		// log that the insertion did not take place
	}
}

void scene::rename_mesh(const std::string& old_name, const std::string& new_name)
{
	std::map<std::string, mesh>::iterator it = m_meshes.find(old_name);
	if (it == m_meshes.end()) {
		// the mesh does not exist
		return;
	}

	mesh m = m_meshes[old_name];
	m_meshes.erase(it);
	m_meshes[new_name] = std::move(m);
}

void scene::delete_mesh(const std::string& name)
{
	std::map<std::string, mesh>::iterator it = m_meshes.find(name);
	if (it == m_meshes.end()) {
		// the mesh does not exist
		return;
	}

	m_meshes.erase(it);
}

void scene::add_model(const std::string& name, const model& m)
{
	m_modified = true;
	auto result = m_models_x.emplace(name, m);
	if (!result.second) {
		// log that the insertion did not take place
		return;
	}

	m_model_types[m_meshes[m.get_mesh_name()].get_type()]++;
	model* addr = &((*result.first).second);
	m_sorted_by_mesh_name.insert(addr);
}

void scene::rename_model(const std::string& old_name, const std::string& new_name)
{
	std::map<std::string, model>::iterator it = m_models_x.find(old_name);
	if (it == m_models_x.end()) {
		// the mesh does not exist
		return;
	}

	model m = m_models_x[old_name];
	std::set<model*, model_comparator>::iterator sorted_it = m_sorted_by_mesh_name.find(&m_models_x[old_name]);	// it must exist
	m_sorted_by_mesh_name.erase(sorted_it);

	m_models_x.erase(it);
	this->add_model(new_name, m);
}

void scene::delete_model(const std::string& name)
{
	std::map<std::string, model>::iterator it = m_models_x.find(name);
	if (it == m_models_x.end()) {
		// the mesh does not exist
		return;
	}

	std::set<model*, model_comparator>::iterator sorted_it = m_sorted_by_mesh_name.find(&m_models_x[name]);	// it must exist
	m_sorted_by_mesh_name.erase(sorted_it);

	m_model_types[m_meshes[it->second.get_mesh_name()].get_type()]--;
	m_models_x.erase(it);
}

void scene::change_model_mesh(const std::string& model_name, const std::string& new_mesh_name)
{
	// erase the entry for the old mesh name
	std::set<model*, model_comparator>::iterator sorted_it = m_sorted_by_mesh_name.find(&m_models_x[model_name]);	// it must exist
	m_sorted_by_mesh_name.erase(sorted_it);

	// change the mesh's name and add it again
	model& m = m_models_x[model_name];
	m.set_mesh_name(new_mesh_name);
	m_sorted_by_mesh_name.insert(&m);
}

scene::gpu_packet scene::build_packet() const
{
	// add all the meshes' names to a vector (already sorted)
	std::vector<std::string> names;
	for (const auto& m : m_meshes) {
		names.emplace_back(m.first);
	}

	m_modified = false;	// the scene is not modified while building the packet
	gpu_packet pkt;
	std::memcpy(pkt.num_drawcalls, m_model_types.data(), mesh::type::NUMTYPES * sizeof(UINT));

	// the arrays will be nullptr if no draw calls or triangle meshes exist
	pkt.tri_meshes = nullptr;
	pkt.tri_mesh_dcs = nullptr;
	pkt.sphere_dcs = nullptr;

	// copy the data for each mesh in the meshes array
	pkt.num_tri_meshes = m_meshes.size();
	if (m_meshes.size() != 0) {
		pkt.tri_meshes = new gpu_packet::tri_mesh[m_meshes.size()];
		UINT idx = 0;
		for (const auto& n : names) {
			const mesh& m = m_meshes.at(n);

			UINT num_indices = m.get_indices().size();
			UINT num_vertices = m.get_vertices().size();

			pkt.tri_meshes[idx].num_indices = num_indices;
			pkt.tri_meshes[idx].num_vertices = num_vertices;
			pkt.tri_meshes[idx].indices = new UINT[num_indices];
			pkt.tri_meshes[idx].vertices = new vertex[num_vertices];

			std::memcpy(pkt.tri_meshes[idx].indices, m.get_indices().data(), num_indices * sizeof(UINT));
			std::memcpy(pkt.tri_meshes[idx].vertices, m.get_vertices().data(), num_vertices * sizeof(vertex));

			idx++;
		}
	}

	// initialize the drawcalls arrays
	if (pkt.num_drawcalls[mesh::type::TRIANGLES] != 0) {
		pkt.tri_mesh_dcs = new gpu_packet::tri_mesh_drawcall[pkt.num_drawcalls[mesh::type::TRIANGLES]];
	}
	if (pkt.num_drawcalls[mesh::type::SPHERES] != 0) {
		pkt.sphere_dcs = new gpu_packet::sphere_drawcall[pkt.num_drawcalls[mesh::type::SPHERES]];
	}

	UINT idxs[2] = { 0, 0 };	// change this to hold indices for other primitive types
	UINT mesh_id = UINT32_MAX;
	std::string last_mesh_name = "";
	// go through the already sorted models with respect to the mesh name
	// update the mesh id only if another mesh was found
	for (const auto& pm : m_sorted_by_mesh_name) {
		// use a bsearch to find the mesh id in the array of names
		if (pm->get_mesh_name() != last_mesh_name) {
			last_mesh_name = pm->get_mesh_name();
			mesh_id = std::lower_bound(names.begin() + (mesh_id + 1), names.end(), last_mesh_name) - names.begin();
		}
		switch (m_meshes.at(last_mesh_name).get_type()) {
		case mesh::type::TRIANGLES:
			pkt.tri_mesh_dcs[idxs[mesh::type::TRIANGLES]].mesh_id = mesh_id;
			pkt.tri_mesh_dcs[idxs[mesh::type::TRIANGLES]].transform = pm->get_transform();
			idxs[mesh::type::TRIANGLES]++;
			break;
		case mesh::type::SPHERES:
			pkt.sphere_dcs[idxs[mesh::type::SPHERES]].radius = pm->get_scale().x;
			pkt.sphere_dcs[idxs[mesh::type::SPHERES]].center = pm->get_translation();
			idxs[mesh::type::SPHERES]++;
			break;
		}
	}

	// copy the data to the gpu
	cudaError cderr;
	gpu_packet::tri_mesh* tri_meshes = nullptr;
	gpu_packet::tri_mesh_drawcall* tri_mesh_dcs = nullptr;
	gpu_packet::sphere_drawcall* sphere_dcs = nullptr;

	if (pkt.num_tri_meshes != 0) {
		RENDERER_THROW_CUDA(cudaMalloc((void**)&tri_meshes, pkt.num_tri_meshes * sizeof(gpu_packet::tri_mesh)));
	}
	for (UINT i = 0; i < m_meshes.size(); i++) {
		vertex* vertices;
		UINT* indices;
		RENDERER_THROW_CUDA(cudaMalloc((void**)&vertices, pkt.tri_meshes[i].num_vertices * sizeof(vertex)));
		RENDERER_THROW_CUDA(cudaMalloc((void**)&indices,  pkt.tri_meshes[i].num_indices * sizeof(UINT)));
		RENDERER_THROW_CUDA(cudaMemcpy(vertices, pkt.tri_meshes[i].vertices, pkt.tri_meshes[i].num_vertices * sizeof(vertex), cudaMemcpyHostToDevice));
		RENDERER_THROW_CUDA(cudaMemcpy(indices, pkt.tri_meshes[i].indices,   pkt.tri_meshes[i].num_indices * sizeof(UINT), cudaMemcpyHostToDevice));

		// delete the cpu side data, no longer needed
		delete[] pkt.tri_meshes[i].vertices;
		delete[] pkt.tri_meshes[i].indices; 

		// assign to the final tri_mesh the pointers to the data on the gpu side
		RENDERER_THROW_CUDA(cudaMemcpy(&tri_meshes[i].num_indices, &pkt.tri_meshes[i].num_indices, sizeof(UINT), cudaMemcpyHostToDevice));
		RENDERER_THROW_CUDA(cudaMemcpy(&tri_meshes[i].num_vertices, &pkt.tri_meshes[i].num_vertices, sizeof(UINT), cudaMemcpyHostToDevice));
		RENDERER_THROW_CUDA(cudaMemcpy(&tri_meshes[i].vertices, &vertices, sizeof(size_t), cudaMemcpyHostToDevice));
		RENDERER_THROW_CUDA(cudaMemcpy(&tri_meshes[i].indices, &indices, sizeof(size_t), cudaMemcpyHostToDevice));
	}
	if (m_meshes.size() != 0) {
		delete[] pkt.tri_meshes; pkt.tri_meshes = tri_meshes;
	}

	if (pkt.num_drawcalls[mesh::type::TRIANGLES] != 0) {
		RENDERER_THROW_CUDA(cudaMalloc((void**)&tri_mesh_dcs, pkt.num_drawcalls[mesh::type::TRIANGLES] * sizeof(gpu_packet::tri_mesh_drawcall)));
		RENDERER_THROW_CUDA(cudaMemcpy(tri_mesh_dcs, pkt.tri_mesh_dcs, pkt.num_drawcalls[mesh::type::TRIANGLES] * sizeof(gpu_packet::tri_mesh_drawcall), cudaMemcpyHostToDevice));
		delete[] pkt.tri_mesh_dcs; pkt.tri_mesh_dcs = tri_mesh_dcs;
	}
	if (pkt.num_drawcalls[mesh::type::SPHERES] != 0) {
		RENDERER_THROW_CUDA(cudaMalloc((void**)&sphere_dcs, pkt.num_drawcalls[mesh::type::SPHERES] * sizeof(gpu_packet::sphere_drawcall)));
		RENDERER_THROW_CUDA(cudaMemcpy(sphere_dcs, pkt.sphere_dcs, pkt.num_drawcalls[mesh::type::SPHERES] * sizeof(gpu_packet::sphere_drawcall), cudaMemcpyHostToDevice));
		delete[] pkt.sphere_dcs; pkt.sphere_dcs = sphere_dcs;
	}

	return pkt;
}

void scene::free_packet(gpu_packet* pkt) const
{
	cudaError cderr;

	if (pkt->num_drawcalls[mesh::type::TRIANGLES] != 0) {
		RENDERER_THROW_CUDA(cudaFree(pkt->tri_mesh_dcs));
		pkt->tri_mesh_dcs = nullptr;
	}
	if (pkt->num_drawcalls[mesh::type::SPHERES] != 0) {
		RENDERER_THROW_CUDA(cudaFree(pkt->sphere_dcs));
		pkt->sphere_dcs = nullptr;
	}

	for (UINT i = 0; i < pkt->num_tri_meshes; i++) {
		RENDERER_THROW_CUDA(cudaFree(pkt->tri_meshes[i].vertices));
		RENDERER_THROW_CUDA(cudaFree(pkt->tri_meshes[i].indices));
		pkt->tri_meshes[i].vertices = nullptr;
		pkt->tri_meshes[i].indices = nullptr;
	}
	if (pkt->num_tri_meshes != 0) {
		RENDERER_THROW_CUDA(cudaFree(pkt->tri_meshes));
		pkt->tri_meshes = nullptr;
	}
}
