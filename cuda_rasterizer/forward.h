/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* colors,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* features,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color);

	///////////////////////////////////////////////////////////////
	// LightGaussian
	///////////////////////////////////////////////////////////////

	/**
	 * @brief 执行渲染操作，并计算每个高斯分布的贡献次数（gaussian_count）和重要性评分（important_score）
	 * @param grid 网格大小
	 * @param block 块大小
	 * @param grid 网格大小
	 * @param block 块大小
	 * @param ranges 每个线程块的高斯分布范围
	 * @param point_list 点列表
	 * @param W 视图宽度
	 * @param H 视图高度
	 * @param points_xy_image 图像坐标
	 * @param features 特征
	 * @param conic_opacity 锥体不透明度
	 * @param final_T 最终 T
	 * @param n_contrib 贡献次数
	 * @param bg_color 背景颜色
	 * @param gaussians_count 高斯分布数量
	 * @param important_score 重要性评分
	 * @param out_color 输出颜色
	*/
	void count_gaussian(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* features,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		int* gaussians_count,
		float* important_score,
		float* out_color);
}


#endif