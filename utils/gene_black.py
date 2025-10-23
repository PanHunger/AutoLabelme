import cv2
import numpy as np
import random

def generate_dirt_mask(width, height, 
                         # --- 大块污渍参数 (Stains) ---
                         stain_scale=60, stain_intensity=0.6, blur_kernel_stain=(25, 25),
                         # --- 污点簇参数 (Clustered Speckles) ---
                         num_clusters=150,
                         min_cluster_size=3,
                         max_cluster_size=16,
                         cluster_density=0.9, # 提高密度，方便后续模糊
                         blur_kernel_cluster=(3, 3)): # 【新参数】控制黑点簇边缘的光滑度
    """
    生成一个真实的污渍和【光滑边缘的成簇】黑点遮罩。
    
    新参数 (污点簇):
    - blur_kernel_cluster: 对每个污点簇进行高斯模糊的核大小，控制其边缘的光滑度。
                           例如 (3,3), (5,5)。数值越大，边缘越模糊。
    """
    
    # 1. 创建基础云状噪声 (Stains) - 保持不变，但模糊核参数名区分开
    stain_mask = np.zeros((height, width), dtype=np.float32)
    if stain_intensity > 0:
        small_w, small_h = width // stain_scale, height // stain_scale
        if small_w == 0: small_w = 1
        if small_h == 0: small_h = 1
        
        noise = np.random.rand(small_h, small_w)
        
        stain_base = cv2.resize(noise, (width, height), interpolation=cv2.INTER_LINEAR)
        stain_base = cv2.GaussianBlur(stain_base, blur_kernel_stain, 0) # 使用区分的模糊核
        
        stain_mask = (stain_base / np.max(stain_base)) * stain_intensity

    # 2. 创建成簇的污点 (Clustered Speckles)
    # 先在一个临时遮罩上生成所有粗糙的簇
    temp_speckle_mask = np.zeros((height, width), dtype=np.float32)
    
    for _ in range(num_clusters):
        cluster_w = random.randint(min_cluster_size, max_cluster_size)
        cluster_h = random.randint(min_cluster_size, max_cluster_size)
        
        x = random.randint(0, width - cluster_w)
        y = random.randint(0, height - cluster_h)
        
        roi = temp_speckle_mask[y:y+cluster_h, x:x+cluster_w]
        
        random_fill = np.random.rand(cluster_h, cluster_w)
        fill_mask = random_fill < cluster_density
        
        # 填充污点时，给它一个较暗（接近1.0）的颜色，因为我们后面要反转作为混合图层
        # 这里设置为1.0（白色）以便模糊后能被更好地识别为“脏”
        dot_colors = np.random.uniform(0.7, 1.0, size=(cluster_h, cluster_w)) # 更偏向白色
        
        roi[fill_mask] = dot_colors[fill_mask]

    # 【核心修改】对整个污点簇遮罩进行局部高斯模糊，使其边缘光滑
    if blur_kernel_cluster[0] > 1 and blur_kernel_cluster[1] > 1: # 只有核大小大于1才模糊
        speckle_mask = cv2.GaussianBlur(temp_speckle_mask, blur_kernel_cluster, 0)
    else:
        speckle_mask = temp_speckle_mask # 如果不模糊，直接使用
        
    # 3. 合并污渍和黑点簇
    # 转换为 0-1 范围，方便混合
    stain_mask_norm = stain_mask # 已经在0-1
    speckle_mask_norm = speckle_mask # 已经在0-1
    
    # 将污渍和黑点作为“暗化”图层合并。白色（1.0）表示干净，黑色（0.0）表示脏。
    # 所以我们用 1 - mask 的方式来混合，表示污渍是“减少亮度”的效果
    combined_mask = 1 - (1 - stain_mask_norm) * (1 - speckle_mask_norm)
    
    # 转换为 8-bit 灰度图像（0-255）
    final_mask = (combined_mask * 255).astype(np.uint8)
    
    return final_mask


# apply_dirt_overlay 函数保持不变，这里省略...
def apply_dirt_overlay(image, dirt_mask, x, y, width, height, blend_mode='multiply', intensity=0.8):
    if x + width > image.shape[1] or y + height > image.shape[0]:
        raise ValueError("指定的矩形区域超出了图片边界。")
    roi = image[y:y+height, x:x+width]
    mask = cv2.resize(dirt_mask, (width, height))
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_float = mask_rgb.astype(np.float32) / 255.0
    roi_float = roi.astype(np.float32) / 255.0
    dirt_layer_float = 1.0 - mask_float 
    blended_roi_float = None
    if blend_mode == 'multiply':
        blended_roi_float = roi_float * dirt_layer_float
    elif blend_mode == 'overlay':
        low = 2 * roi_float * dirt_layer_float
        high = 1 - 2 * (1 - roi_float) * (1 - dirt_layer_float)
        blended_roi_float = np.where(roi_float < 0.5, low, high)
    elif blend_mode == 'soft_light':
        blended_roi_float = np.where(dirt_layer_float < 0.5,
                                    (2 * roi_float * dirt_layer_float + roi_float**2 * (1 - 2 * dirt_layer_float)),
                                    (2 * roi_float * (1 - dirt_layer_float) + np.sqrt(roi_float) * (2 * dirt_layer_float - 1)))
    else:
        print(f"未知的混合模式 '{blend_mode}'，将使用 'multiply'。")
        blended_roi_float = roi_float * dirt_layer_float
    blended_roi = (blended_roi_float * 255).astype(np.uint8)
    final_roi = cv2.addWeighted(blended_roi, intensity, roi, 1 - intensity, 0)
    result_image = image.copy()
    result_image[y:y+height, x:x+width] = final_roi
    return result_image


# --- 主程序 (更新调用参数) ---
if __name__ == '__main__':
    try:
        original_image = cv2.imread('your_image.jpg')
        if original_image is None:
            raise FileNotFoundError("图片文件未找到或无法读取。请检查路径。")
    except Exception as e:
        print(e)
        original_image = np.full((800, 1200, 3), (200, 200, 200), dtype=np.uint8)

    rect_x, rect_y, rect_w, rect_h = 200, 150, 600, 400

    dirt_mask = generate_dirt_mask(rect_w, rect_h,
                                   # 大块油污效果
                                   stain_scale=40,
                                   stain_intensity=0.5,
                                   blur_kernel_stain=(35, 35), # 大块污渍的模糊核
                                   # 污点簇效果
                                   num_clusters=250,
                                   min_cluster_size=3,
                                   max_cluster_size=12,
                                   cluster_density=0.9,
                                   blur_kernel_cluster=(3, 3)) # 【新参数】黑点簇的模糊核，使其边缘光滑
                                   
    final_image = apply_dirt_overlay(original_image, dirt_mask,
                                     rect_x, rect_y, rect_w, rect_h,
                                     blend_mode='multiply',
                                     intensity=0.9)

    cv2.imshow('Original Image', original_image)
    # 放大遮罩的一部分，以便看清簇的细节
    mask_detail = cv2.resize(dirt_mask, (rect_w * 2, rect_h * 2), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('Generated Dirt Mask (Zoomed)', mask_detail)
    cv2.imshow('Final Image with Dirt', final_image)
    
    cv2.imwrite('result_image_with_smooth_clustered_dirt.jpg', final_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()