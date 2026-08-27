# C++ 快速排序实现

本目录包含了完整的C++快速排序实现，包括多种变体和优化版本。

## 文件说明

### 1. `quicksort_basic.cpp` - 基础快速排序实现
最简单的快速排序实现，适合学习和理解算法原理。
- 使用Lomuto分区方案
- 包含完整的使用示例
- 支持整数、浮点数、字符串等类型

### 2. `quicksort_simple.cpp` - 简化版快速排序
包含多种快速排序变体的简化版本：
- 基本快速排序（Lomuto分区）
- Hoare分区快速排序（更高效）
- 随机化快速排序（避免最坏情况）
- 包含边界情况处理

### 3. `quicksort.cpp` - 完整测试版本
最完整的实现，包含10个全面的测试用例：
- 整数数组排序
- 浮点数数组排序
- 随机大数组测试
- Hoare分区测试
- 随机化排序测试
- 已排序数组测试
- 逆序数组测试
- 空数组测试
- 单元素数组测试
- 重复元素数组测试

### 4. `quicksort.h` - 快速排序库头文件
可重用的快速排序库，包含：
- 基本快速排序（Lomuto分区）
- Hoare分区快速排序
- 随机化快速排序
- 三数取中快速排序
- 混合快速排序（结合插入排序）
- 辅助函数（检查排序、生成随机数组等）

### 5. `test_quicksort.cpp` - 库测试程序
测试`quicksort.h`库的功能和性能：
- 基本功能测试
- 不同类型数据测试
- 边界情况测试
- 性能测试（10000个随机整数）
- 最坏情况测试（已排序大数组）
- 小数组优化测试

## 编译和运行

### 编译基础版本
```bash
g++ -std=c++11 -o quicksort_basic quicksort_basic.cpp
./quicksort_basic
```

### 编译简化版本
```bash
g++ -std=c++11 -o quicksort_simple quicksort_simple.cpp
./quicksort_simple
```

### 编译完整测试版本
```bash
g++ -std=c++11 -o quicksort quicksort.cpp
./quicksort
```

### 编译库测试程序
```bash
g++ -std=c++11 -o test_quicksort test_quicksort.cpp
./test_quicksort
```

## 算法特点

### 快速排序基本思想
1. **分治策略**：将大问题分解为小问题
2. **选择基准**：从数组中选择一个元素作为基准
3. **分区**：将数组分为两部分，左边小于基准，右边大于基准
4. **递归**：对左右两部分递归排序

### 时间复杂度
- **平均情况**：O(n log n)
- **最坏情况**：O(n²) - 当数组已排序或逆序时
- **最佳情况**：O(n log n)

### 空间复杂度
- **递归栈空间**：O(log n) - 平均情况
- **最坏情况**：O(n) - 当每次分区都极不平衡时

### 稳定性
快速排序是**不稳定**的排序算法，相等元素的相对位置可能改变。

## 优化技术

### 1. 随机化选择基准
```cpp
int randomIndex = low + rand() % (high - low + 1);
std::swap(arr[randomIndex], arr[high]);
```
避免已排序数组的最坏情况，使算法性能更稳定。

### 2. 三数取中法
```cpp
int mid = low + (high - low) / 2;
// 选择low、mid、high的中位数作为基准
```
进一步优化基准选择，减少不平衡分区的概率。

### 3. Hoare分区方案
比Lomuto分区更高效，交换次数更少。

### 4. 混合排序
当子数组大小小于阈值时，使用插入排序：
```cpp
if (high - low + 1 <= threshold) {
    insertionSort(arr, low, high);
}
```
插入排序在小数组上比快速排序更高效。

### 5. 尾递归优化
将递归调用转换为循环，减少递归深度。

## 使用示例

### 基本使用
```cpp
#include <vector>
#include "quicksort.h"

int main() {
    std::vector<int> arr = {64, 34, 25, 12, 22, 11, 90};
    
    // 基本快速排序
    quicksort::quickSort(arr);
    
    // Hoare分区快速排序
    quicksort::quickSortHoare(arr);
    
    // 随机化快速排序
    quicksort::randomizedQuickSort(arr);
    
    // 三数取中快速排序
    quicksort::medianQuickSort(arr);
    
    // 混合快速排序（阈值=16）
    quicksort::hybridQuickSort(arr, 16);
    
    return 0;
}
```

### 检查数组是否已排序
```cpp
bool isSorted = quicksort::isSorted(arr);
```

### 生成随机数组
```cpp
// 生成1000个0-10000之间的随机整数
std::vector<int> randomArr = quicksort::generateRandomArray<int>(1000, 0, 10000);

// 生成100个0.0-1.0之间的随机浮点数
std::vector<double> randomDoubles = quicksort::generateRandomArray<double>(100, 0.0, 1.0);
```

## 性能对比

从测试结果可以看出：
1. **Hoare分区**比Lomuto分区效率更高
2. **随机化**和**三数取中**能有效避免最坏情况
3. **混合排序**在小数组上表现更好
4. 在已排序数组上，基本快速排序性能最差，优化版本性能稳定

## 适用场景

### 适合使用快速排序的情况：
- 数据量较大
- 对稳定性没有要求
- 内存有限（原地排序）
- 平均性能要求高

### 不适合使用快速排序的情况：
- 数据量很小（考虑插入排序）
- 要求稳定排序（考虑归并排序）
- 数据已基本有序（考虑TimSort）

## 扩展阅读

1. **Dual-Pivot QuickSort**：Java 7+使用的双基准快速排序
2. **IntroSort**：C++ STL的sort函数实现，结合快速排序、堆排序和插入排序
3. **TimSort**：Python和Java使用的混合排序算法，结合归并排序和插入排序
4. **Parallel QuickSort**：并行快速排序实现，利用多核处理器

## 许可证

这些代码示例可以自由使用、修改和分发。