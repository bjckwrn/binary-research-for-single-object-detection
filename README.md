# 图像边界检测算法（Image Boundary Detection Algorithm）

本项目提供了一种基于二分法和零样本学习的图像边界检测算法，该算法利用CLIP模型强大的图像与文本联合学习能力，通过计算特定区域的匹配概率并结合二分法，快速精确地确定图像中的目标边界。

This project offers an image boundary detection algorithm based on binary search and zero-shot learning. The algorithm leverages the powerful image-text joint learning capabilities of the CLIP model to calculate matching probabilities for specific regions, combined with binary search to quickly and accurately determine object boundaries in images.

## 核心功能 (Key Functions)

### `detect_probabilities` 函数

该函数裁剪图像的指定区域，并将其余部分填充为黑色，以便更准确地计算该区域与文本描述之间的匹配概率。

This function crops the specified region of an image and fills the remaining part with black, enabling a more accurate calculation of the matching probability between the region and the textual description.

```python
def detect_probabilities(image, x1, y1, x2, y2):
    black_image = Image.new('RGB', image.size, (0, 0, 0))
    cropped_image = image.crop((x1, y1, min(x2, image.width), min(y2, image.height)))
    black_image.paste(cropped_image, (x1, y1))
    return black_image
```
### `calculate_probabilities` 函数

此函数使用CLIP模型计算图像裁剪区域与文本输入之间的相似度，并返回一个概率值，表示匹配的可能性。

This function uses the CLIP model to calculate the similarity between a cropped image region and the text input, returning a probability value that indicates the likelihood of a match.

```python
def calculate_probabilities(image, text_inputs):
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]

    return similarity[0] * 100  # Convert to percentage
```
### `binary_search_boundary` 函数

该函数实现二分法算法，通过计算不同边界点的匹配概率，快速确定目标物体的边界。此方法相比暴力搜索方法，大幅提高了边界检测的效率和准确性。

This function implements a binary search algorithm to quickly determine the boundary of an object by calculating the matching probabilities at different boundary points. This method significantly improves the efficiency and accuracy of boundary detection compared to brute-force search methods.

```python
def binary_search_boundary(image, text_inputs, width, height, threshold, direction):
    low, high = 0, width if direction in ['left', 'right'] else height
    boundary = None

    while low <= high:
        mid = (low + high) // 2
        if direction == 'left':
            probabilities = calculate_probabilities(detect_probabilities(image, 0, 0, mid, height), text_inputs)
        elif direction == 'right':
            probabilities = calculate_probabilities(detect_probabilities(image, mid, 0, width, height), text_inputs)
        elif direction == 'top':
            probabilities = calculate_probabilities(detect_probabilities(image, 0, 0, width, mid), text_inputs)
        elif direction == 'bottom':
            probabilities = calculate_probabilities(detect_probabilities(image, 0, mid, width, height), text_inputs)

        if probabilities >= threshold:
            boundary = mid
            if direction in ['left', 'top']:
                high = mid - 1
            else:
                low = mid + 1
        else:
            if direction in ['left', 'top']:
                low = mid + 1
            else:
                high = mid - 1

    return boundary
```



