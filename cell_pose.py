# 导入必要的库
from skimage import io
import numpy as np
import pandas as pd
import scanpy as sc
import tifffile
import geopandas as gpd
from anndata import AnnData
from shapely.geometry import Point
from cellpose import models, core
import os
import gc
import dask.array as da
from PIL import Image
import matplotlib.pyplot as plt
import psutil  # 内存监控
from scipy.sparse import coo_matrix, csr_matrix

# 解除Pillow的像素限制
Image.MAX_IMAGE_PIXELS = None


# 内存监控函数
def mem_usage():
    return psutil.Process().memory_info().rss / (1024 ** 3)


# 显式释放内存资源
def release_memory(*objects):
    for obj in objects:
        del obj
    gc.collect()
    print(f"内存释放后: {mem_usage():.2f} GiB")


# 高效读取大尺寸图像
def read_large_image(img_path: str, downscale_factor: int = 1) -> np.ndarray:
    try:
        pil_img = Image.open(img_path).convert('L')
        w, h = pil_img.size
        new_w, new_h = w // downscale_factor, h // downscale_factor
        img = np.array(pil_img.resize((new_w, new_h), Image.BILINEAR))
        del pil_img
    except Exception as e:
        print(f"PIL读取失败，使用Dask备用方案: {e}")
        chunks = (2000, 2000)
        dask_img = da.from_zarr(img_path, chunks=chunks)
        img = downscale_image(dask_img.compute(), downscale_factor)
    return img.astype(np.uint16)


# 高效下采样图像
def downscale_image(img: np.ndarray, downscale_factor: int) -> np.ndarray:
    h, w = img.shape
    new_h, new_w = h // downscale_factor, w // downscale_factor
    valid_img = img[:new_h * downscale_factor, :new_w * downscale_factor]
    pooled = valid_img.reshape(new_h, downscale_factor, new_w, downscale_factor).mean(axis=(1, 3))
    return pooled.astype(img.dtype)


# 内存优化的Cellpose分割
def run_cellpose(img: np.ndarray, model_path: str = 'cyto3') -> np.ndarray:
    use_GPU = core.use_gpu()
    model = models.CellposeModel(gpu=use_GPU, model_type=model_path)
    masks, _, _ = model.eval(
        [img], channels=[0, 0], flow_threshold=0.8,
        cellprob_threshold=0.0, batch_size=1
    )
    return masks[0]


# 分块处理大图像
def process_image_chunks(img, chunk_per_axis=2, model_path='cyto3') -> np.ndarray:
    full_mask = np.zeros(img.shape, dtype=np.uint32)
    constant = 1000000
    h, w = img.shape
    chunk_h = h // chunk_per_axis
    chunk_w = w // chunk_per_axis
    chunks = [
        (0, chunk_h, 0, chunk_w), (0, chunk_h, chunk_w, w),
        (chunk_h, h, 0, chunk_w), (chunk_h, h, chunk_w, w)
    ]

    for i, (r_start, r_end, c_start, c_end) in enumerate(chunks):
        chunk = img[r_start:r_end, c_start:c_end]
        mask = run_cellpose(chunk, model_path)
        chunk_mask = mask.astype(np.uint32)
        non_zero = chunk_mask > 0
        chunk_mask[non_zero] += i * constant
        full_mask[r_start:r_end, c_start:c_end] = chunk_mask
        release_memory(chunk, mask, chunk_mask)

    full_mask[full_mask % constant == 0] = 0
    return full_mask


# 随机采样函数
def random_sample_data(adata, sample_fraction=1) -> AnnData:
    np.random.seed(42)
    sampled_indices = np.random.choice(
        adata.obs_names, size=int(adata.n_obs * sample_fraction), replace=False
    )
    return adata[sampled_indices].copy()


# 可视化结果
def visualize_results(img, masks) -> None:
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(masks, cmap='nipy_spectral')
    plt.title('Segmentation Mask')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('segmentation_result.png', dpi=300)
    plt.close()


# 主程序
def main():
    # 配置路径参数
    img_path = "Visium_HD_Human_Colon_Cancer_tissue_image.btf"
    seg_path = "./segmentation.tif"  # 分割结果文件路径
    dir_base = "./data/Visium_HD_Human_Colon_Cancer_binned_outputs/binned_outputs/square_002um/"
    downscale_factor = 2

    # ====================== 关键修改：添加条件跳过机制 ======================
    # 检查分割文件是否存在，存在则直接加载，否则执行处理流程
    if os.path.exists(seg_path):
        print("检测到已存在的分割文件，直接加载...")
        full_mask = tifffile.imread(seg_path)
        print(f"已加载分割掩码，尺寸: {full_mask.shape}")
    else:
        # 1. 读取并下采样大尺寸图像
        print("开始读取大尺寸图像...")
        dapi_image = read_large_image(img_path, downscale_factor)
        print(f"下采样后图像尺寸: {dapi_image.shape} | 数据类型: {dapi_image.dtype}")

        # 2. 分块处理图像分割
        print("开始分块细胞分割...")
        full_mask = process_image_chunks(dapi_image, chunk_per_axis=2)

        # 3. 保存并可视化分割结果
        tifffile.imwrite(seg_path, full_mask.astype(np.uint32))
        visualize_results(dapi_image, full_mask)
        release_memory(dapi_image)
    # ===================================================================

    # 4. 处理空间转录组数据
    print("加载空间转录组数据...")
    raw_h5_file = os.path.join(dir_base, "filtered_feature_bc_matrix.h5")
    adata = sc.read_10x_h5(raw_h5_file)
    print(f"初始内存: {mem_usage():.2f} GiB")

    # 5. 随机采样10%的数据
    adata.var_names_make_unique()

    # 6. 加载组织位置
    tissue_position_file = os.path.join(dir_base, "spatial/tissue_positions.parquet")
    df_tissue_positions = pd.read_parquet(tissue_position_file).set_index("barcode")
    adata.obs = adata.obs.join(df_tissue_positions, how='left')

    # 7. 创建几何图形并缩放坐标
    scale_factor = downscale_factor
    adata.obs['scaled_row'] = (adata.obs['pxl_row_in_fullres'] / scale_factor).astype(int)
    adata.obs['scaled_col'] = (adata.obs['pxl_col_in_fullres'] / scale_factor).astype(int)
    geometry = [Point(xy) for xy in zip(adata.obs['scaled_col'], adata.obs['scaled_row'])]
    gdf_coordinates = gpd.GeoDataFrame(adata.obs, geometry=geometry)

    # 8. 边界安全检查
    in_bounds = (
            (gdf_coordinates['scaled_row'] >= 0) &
            (gdf_coordinates['scaled_row'] < full_mask.shape[0]) &
            (gdf_coordinates['scaled_col'] >= 0) &
            (gdf_coordinates['scaled_col'] < full_mask.shape[1])
    )
    gdf_coordinates = gdf_coordinates[in_bounds]

    # 9. 分配捕获区域
    cells = full_mask[
        gdf_coordinates['scaled_row'].astype(int),
        gdf_coordinates['scaled_col'].astype(int)
    ]
    gdf_coordinates['cells'] = cells
    adata.obs = adata.obs.join(gdf_coordinates['cells'], how='left')
    adata = adata[~adata.obs['cells'].isna()]
    adata.obs['cells'] = adata.obs['cells'].astype(int)

    # 10. 汇总计数（COO格式优化）
    print("开始汇总计数...")
    cells = adata.obs['cells'].values
    unique_cells = np.unique(cells)
    cell_to_index = {cell: idx for idx, cell in enumerate(unique_cells)}
    row_indices = [cell_to_index[c] for c in cells]

    # 关键优化：使用COO格式避免内存爆炸
    X_coo = adata.X.tocoo()
    new_rows = np.array(row_indices)[X_coo.row]  # 映射行索引
    summed_counts = coo_matrix(
        (X_coo.data, (new_rows, X_coo.col)),
        shape=(len(unique_cells), adata.n_vars)
    ).tocsr()  # 转换为CSR格式

    # 11. 计算平均坐标
    coords = adata.obs[['scaled_row', 'scaled_col']].values
    cell_coords = np.array([coords[cells == cell].mean(axis=0) for cell in unique_cells])

    # 12. 创建新AnnData对象
    grouped_adata = AnnData(
        X=summed_counts,
        obs=pd.DataFrame(index=unique_cells),
        var=adata.var,
    )
    grouped_adata.obsm['spatial'] = cell_coords

    # 13. 保存结果
    grouped_adata.write('adata_cellpose_sampled.h5ad', compression='gzip')  # 压缩存储
    print("处理完成！结果已保存到 adata_cellpose_sampled.h5ad")


if __name__ == "__main__":
    main()