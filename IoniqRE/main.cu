#include "ioniq_windows.h"

#include "core.h"
#include "window.h"
#include "application.h"
#include "renderer.h"

#include <fstream>

#include <cuda.h>
#include <cuda_runtime.h>

#define CHECKCUDA(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t err, const char* func, const char* file, int line)
{
	if (err) {
		cudaDeviceReset();
		exit(99);
	}
}

struct pixel
{
	unsigned char r;
	unsigned char g;
	unsigned char b;
	unsigned char a;
};

__global__ void render(pixel* fb, int width, int height)
{
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= height || j >= width) {
		return;
	}
	int pixelid = i * width + j;
	fb[pixelid].r = (unsigned char)(256.0f * (float)i / height);
	fb[pixelid].g = (unsigned char)(256.0f * (float)j / width );
	fb[pixelid].b = 50;
}

int WINAPI WinMain(HINSTANCE hInstance,
	HINSTANCE hPrevInstance,
	LPSTR lpCmdLine,
	int nCmdShow)
{
	std::ofstream fout("image.ppm");
	size_t image_width = 1920;
	size_t image_height = 1080;
	size_t tx = 16, ty = 16;
	size_t num_pixels = image_height * image_width;
	size_t fbsize = num_pixels * sizeof(pixel);

	pixel* fb;
	CHECKCUDA(cudaMallocManaged((void**)&fb, fbsize));
	dim3 blocks(120, 68);
	dim3 threads(tx, ty);
	// Render

	render<<<blocks, threads>>> (fb, image_width, image_height);
	CHECKCUDA(cudaGetLastError());
	CHECKCUDA(cudaDeviceSynchronize());

	/*fout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

	for (int i = 0; i < image_height; i++) {
		for (int j = 0; j < image_width; j++) {
			int pixelid = i * image_width + j;
			fout << (int)fb[pixelid].r << ' ' << (int)fb[pixelid].g << ' ' << (int)fb[pixelid].b << '\n';
		}
	}*/
	fout.close();

	try {
		// make a shared ptr to a window because a unique ptr can not be passed to the application constructor
		ref<window> w = std::make_shared<window>(hInstance, 1920, 1080);
		application app(w);

		while (app.process_message()) {
			app.run();
		}

		return EXIT_SUCCESS;
	}
	catch (const ioniq_exception & e) {
		MessageBox(nullptr, e.what(), e.get_type(), MB_OK | MB_ICONEXCLAMATION);
	}
	catch (const std::exception& e) {
		MessageBox(nullptr, e.what(), "Standard exception", MB_OK | MB_ICONEXCLAMATION);
	}
	catch (...) {
		MessageBox(nullptr, "No details available", "Unknown exception type", MB_OK | MB_ICONEXCLAMATION);
	}

	return EXIT_FAILURE;
}
