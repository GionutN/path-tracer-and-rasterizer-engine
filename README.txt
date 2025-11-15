	Overview
	This project represents a rendering engine which supports both path-traced and rasterized graphics, using CUDA and Direct3D 11.

	Requirements
	- Windows operating system
	- Visual Studio 2019 with CUDA toolkit installed
	- Direct3D 11 compatible GPU
	- CUDA Compute Capability 7.5 GPU

	Project Description
	This project is currently a work in progress. The main goal is to build an engine that supports two physically based rendering methods,
a path tracer and a rasterizer, similar to those used in the film and game production.

	Building the project
	To get the project up and running, download the zip archive, open the .sln file and compile the solution.

	Architecture overview
	Before any rendering takes place, the application must process incoming Windows messages. This takes place inside the application class,
which is responsible for two main aspects:
	1. handles incoming system messages
	2. divides the application loop into four main steps:
		- renderer::begin_frame
		- update_frame: updates scene logic
		- draw_frame: issues the draw calls for each of the engines
		- renderer::end_frame

	The update_frame and draw_frame methods are engine-agnostic. They modify internal state that both engines later consume. Each of
the engines has their own implementation of begin_frame and end_frame.

	The renderers form the core of the engine, and are organized as follows. Because the application runs on a single system, a singleton
handles the graphics-device communication layer. This class is called renderer_base, and it creates the resources used by both the rasterizer and
the path tracer: the device, the device context and the swap chain, and handles all the error messages. This class is not meant to be inherited
directly. Instead, both engines derive from renderer_template, which uses renderer_base for device setup.

	The rasterizer is built on top of the renderer base by adding a render target view and depth/stencil buffers. This engine renders the scene
to a multisampled texture, which in the end_frame method is copied to a non-multisampled texture before it is presented to the screen. This is because
the swap chain's swap effect DXGI_SWAP_EFFECT_DISCARD, which supports direct multisampled rendering, is not recommended, as stated on
https://learn.microsoft.com/en-us/windows/win32/api/dxgi/ne-dxgi-dxgi_swap_effect.
	The logic of the rasterizer takes place inside the draw_scene method, where all the meshes in the scene are bound along with their
shader and the draw call is issued.

	The path tracer works differently, architecturally. It uses the same device, device context and swap chain as the rasterizer,
but the entire frame is rendered in a texture, which is applied to a full-screen quad in NDC space, covering the whole screen. The sampler inside
the fragment shader samples from the path-traced texture to display the final pixel.
	The logic of the path tracer also takes place inside the draw_scene method. Here, every 0.1 seconds the engine synchronizes the GPU, copies the
resulting frame to the CPU, and then starts the CUDA kernel again. This approach ensures that the GPU threads have enough time to finish their work and
still maintain the responsiveness of the app.

	Both engines use the scene class in order to draw. The scene is organized in models, meshes and materials. As in most modelling and
rendering systems, all of these elements have a name which refers to them, using ordered maps. A model is defined as a mesh, a material
and a transform.
	Each engine takes its data from the scene in a different way. The rasterizer gets a set of model pointers grouped by their mesh
name. I chose to do it this way because if multiple models refer to the same mesh (instancing), then the mesh is bound only once, which
eliminates the overhead of repeated binding.
	Since the path tracer runs on the GPU, the data from the scene (which is on the CPU) must be transferred efficiently to the GPU. Every time
the scene changes (objects moving, not including the camera, materials/meshes changing) a new "gpu_packet" is constructed and sent before any
rendering. Since STL cannot be used inside a CUDA kernel, all the data is sent as a struct of pointers to arrays of vertices and transforms. The objects
in the scene are split between triangular meshes and spherical meshes because the path tracer treats them differently. This allows rendering of spherical
objects with "infinite" vertices at a fraction of the time it takes to render a polygonal mesh, by using analytic intersection. This is a neat feature which
other known rendering engines do not have.

	Features
	- both rasterized and path-traced frames
	- accurate oren-nayar and emissive materials
	- polygonal meshes and procedurally generated spherical meshes
	- orthographic and perspective projection camera
	- CUDA multithreading

	Planned Features
	The main goals for future development are:
	- adding glass, metal, volume materials
	- texture support
	- adding a principled bsdf shader
	- adding procedurally generated ellipsoids, cylinders and parabolas
	- means to render sequences of frames
	- mesh loading from file
	- automatically generating vertex data when changing between smooth and flat shading
	- path-tracer denoiser
	- BVH acceleration
	- a faster CPU-GPU sync
	- post-processing

	Long-term goals:
	- adding a physics engine
	- adding a fluid simulator

	Bibliography
	- https://learn.microsoft.com/en-us/
	- https://raytracing.github.io/
	- https://scratchapixel.com/
	- "Physically Based Rendering From Theory to Implementation" Third Edition by Matt Pharr, Wenzel Jakob and Greg Humphreys
	- "C++ 3D DirectX Programming" Series by ChiliTomatoNoodle


	License
	This project is provided for educational purposes. All third-party libraries retain their original licenses.