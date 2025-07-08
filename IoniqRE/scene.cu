#include "scene.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "renderer.h"

scene::scene()
{
	m_model_types.fill(0);
}

void scene::add(const mesh& m)
{
	m_modified = true;

	m_models.push_back(m);
	m_model_types[m.get_type()]++;

	m_vertices += m.get_vertices().size();
	m_indices += m.get_indices().size();
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
	//m_modified = true;
	auto result = m_models_x.emplace(name, m);
	if (!result.second) {
		// log that the insertion did not take place
		return;
	}

	m_sorted_by_mesh_name.insert(&(*result.first).second);
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
	m_modified = false;
	if (m_vertices == 0) {
		return { nullptr, nullptr, nullptr };
	}

	gpu_packet packet;
	packet.vertices = new vertex[m_vertices];
	packet.indices = new UINT[m_indices + 1];
	packet.indices[0] = m_indices;	// set the number of indices as the first element in the array
	packet.model_types = new UINT[2];

	// bundle all of the scene's vertices and indices in one big array
	size_t vert_idx = 0, index_idx = 1;
	UINT index_offset = 0;	// shift the indices of the model by the total number of vertices from the previously added meshes
	for (const auto& m : m_models) {
		for (const auto& v : m.get_vertices()) {
			packet.vertices[vert_idx++] = v;
		}

		// CCW sent to the gpu
		for (const auto& i : m.get_indices()) {
			packet.indices[index_idx++] = i + index_offset;
		}
		index_offset += (UINT)m.get_vertices().size();
	}

	packet.model_types[(size_t)mesh::type::TRIANGLES] = m_model_types[(size_t)mesh::type::TRIANGLES];
	packet.model_types[(size_t)mesh::type::SPHERES] = m_model_types[(size_t)mesh::type::SPHERES];

	// copy all the arrays to gpu memory;
	cudaError cderr;
	vertex* vertices;
	UINT* indices;
	UINT* model_types;

	RENDERER_THROW_CUDA(cudaMalloc((void**)&vertices,    m_vertices * sizeof(vertex)));
	RENDERER_THROW_CUDA(cudaMalloc((void**)&indices,     (m_indices + 1) * sizeof(UINT)));
	RENDERER_THROW_CUDA(cudaMalloc((void**)&model_types, (size_t)mesh::type::NUMTYPES * sizeof(mesh::type)));
	RENDERER_THROW_CUDA(cudaMemcpy(vertices,    packet.vertices,    m_vertices * sizeof(vertex), cudaMemcpyHostToDevice));
	RENDERER_THROW_CUDA(cudaMemcpy(indices,     packet.indices,     (m_indices + 1) * sizeof(UINT), cudaMemcpyHostToDevice));
	RENDERER_THROW_CUDA(cudaMemcpy(model_types, packet.model_types, (size_t)mesh::type::NUMTYPES * sizeof(mesh::type), cudaMemcpyHostToDevice));

	// delete and reassign the cpu memory data
	delete[] packet.vertices; packet.vertices = vertices;
	delete[] packet.indices; packet.indices = indices;
	delete[] packet.model_types; packet.model_types = model_types;

	return packet;
}

scene::gpu_packet_x scene::build_packet_x() const
{
	// add all the meshes' names to a vector (already sorted)
	std::vector<std::string> names;
	for (const auto& m : m_meshes) {
		names.emplace_back(m.first);
	}

	m_modified = false;	// the scene is not modified while building the packet
	gpu_packet_x pkt;
	std::memcpy(pkt.num_drawcalls, m_model_types.data(), mesh::type::NUMTYPES * sizeof(UINT));

	// the arrays will be nullptr if no draw calls or triangle meshes exist
	pkt.tri_meshes = nullptr;
	pkt.tri_mesh_dcs = nullptr;
	pkt.sphere_dcs = nullptr;

	// copy the data for each mesh in the meshes array
	pkt.num_tri_meshes = m_meshes.size();
	if (m_meshes.size() != 0) {
		pkt.tri_meshes = new gpu_packet_x::tri_mesh[m_meshes.size()];
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
		pkt.tri_mesh_dcs = new gpu_packet_x::tri_mesh_drawcall[pkt.num_drawcalls[mesh::type::TRIANGLES]];
	}
	if (pkt.num_drawcalls[mesh::type::SPHERES] != 0) {
		pkt.sphere_dcs = new gpu_packet_x::sphere_drawcall[pkt.num_drawcalls[mesh::type::SPHERES]];
	}

	UINT idxs[2] = { 0, 0 };
	UINT mesh_id = UINT32_MAX;
	std::string last_mesh_name = "";
	// go through the already sorted models with respect to the mesh name
	// update the mesh id only if another mesh was found
	for (const auto& pm : m_sorted_by_mesh_name) {
		if (pm->get_mesh_name() != last_mesh_name) {
			mesh_id++;
			last_mesh_name = pm->get_mesh_name();
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
	gpu_packet_x::tri_mesh* tri_meshes = nullptr;
	gpu_packet_x::tri_mesh_drawcall* tri_mesh_dcs = nullptr;
	gpu_packet_x::sphere_drawcall* sphere_dcs = nullptr;

	if (pkt.num_tri_meshes != 0) {
		RENDERER_THROW_CUDA(cudaMalloc((void**)&tri_meshes, pkt.num_tri_meshes * sizeof(gpu_packet_x::tri_mesh)));
	}
	for (UINT i = 0; i < m_meshes.size(); i++) {
		vertex* vertices;
		UINT* indices;
		tri_meshes[i].num_indices = pkt.tri_meshes[i].num_indices;
		tri_meshes[i].num_vertices = pkt.tri_meshes[i].num_vertices;
		RENDERER_THROW_CUDA(cudaMalloc((void**)&vertices, tri_meshes[i].num_vertices * sizeof(vertex)));
		RENDERER_THROW_CUDA(cudaMalloc((void**)&indices, tri_meshes[i].num_indices * sizeof(UINT)));
		RENDERER_THROW_CUDA(cudaMemcpy(vertices, pkt.tri_meshes[i].vertices, tri_meshes[i].num_vertices * sizeof(vertex), cudaMemcpyHostToDevice));
		RENDERER_THROW_CUDA(cudaMemcpy(indices, pkt.tri_meshes[i].indices, tri_meshes[i].num_indices * sizeof(UINT), cudaMemcpyHostToDevice));

		delete[] pkt.tri_meshes[i].vertices; pkt.tri_meshes[i].vertices = vertices;
		delete[] pkt.tri_meshes[i].indices; pkt.tri_meshes[i].indices = indices;
	}
	if (m_meshes.size() != 0) {
		delete[] pkt.tri_meshes; pkt.tri_meshes = tri_meshes;
	}

	if (pkt.num_drawcalls[mesh::type::TRIANGLES] != 0) {
		RENDERER_THROW_CUDA(cudaMalloc((void**)&tri_mesh_dcs, pkt.num_drawcalls[mesh::type::TRIANGLES] * sizeof(gpu_packet_x::tri_mesh_drawcall)));
		RENDERER_THROW_CUDA(cudaMemcpy(tri_mesh_dcs, pkt.tri_mesh_dcs, pkt.num_drawcalls[mesh::type::TRIANGLES] * sizeof(gpu_packet_x::tri_mesh_drawcall), cudaMemcpyHostToDevice));
		delete[] pkt.tri_mesh_dcs; pkt.tri_mesh_dcs = tri_mesh_dcs;
	}
	if (pkt.num_drawcalls[mesh::type::SPHERES] != 0) {
		RENDERER_THROW_CUDA(cudaMalloc((void**)&sphere_dcs, pkt.num_drawcalls[mesh::type::SPHERES] * sizeof(gpu_packet_x::sphere_drawcall)));
		RENDERER_THROW_CUDA(cudaMemcpy(sphere_dcs, pkt.sphere_dcs, pkt.num_drawcalls[mesh::type::SPHERES] * sizeof(gpu_packet_x::sphere_drawcall), cudaMemcpyHostToDevice));
		delete[] pkt.sphere_dcs; pkt.sphere_dcs = sphere_dcs;
	}

	return pkt;
}

void scene::free_packet_x(gpu_packet_x* pkt) const
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
