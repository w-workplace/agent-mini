#include <iostream>
#include <vector>
#include <algorithm>

// ==================== 快速排序核心实现 ====================

/**
 * Lomuto分区方案
 * 参数:
 *   arr - 待排序数组
 *   low - 起始索引
 *   high - 结束索引
 * 返回值: 基准元素的最终位置
 */
template<typename T>
int partition(std::vector<T>& arr, int low, int high) {
    // 选择最后一个元素作为基准
    T pivot = arr[high];
    int i = low - 1; // 小于基准的元素的边界
    
    for (int j = low; j < high; j++) {
        // 如果当前元素小于或等于基准
        if (arr[j] <= pivot) {
            i++;
            std::swap(arr[i], arr[j]);
        }
    }
    
    // 将基准放到正确的位置
    std::swap(arr[i + 1], arr[high]);
    return i + 1;
}

/**
 * 快速排序递归函数
 * 参数:
 *   arr - 待排序数组
 *   low - 起始索引
 *   high - 结束索引
 */
template<typename T>
void quickSort(std::vector<T>& arr, int low, int high) {
    if (low < high) {
        // 分区索引
        int pi = partition(arr, low, high);
        
        // 递归排序分区
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

/**
 * 快速排序包装函数
 * 参数:
 *   arr - 待排序数组
 */
template<typename T>
void quickSort(std::vector<T>& arr) {
    if (!arr.empty()) {
        quickSort(arr, 0, arr.size() - 1);
    }
}

// ==================== 可选：Hoare分区方案（更高效） ====================

/**
 * Hoare分区方案
 * 通常比Lomuto分区更高效，交换次数更少
 */
template<typename T>
int hoarePartition(std::vector<T>& arr, int low, int high) {
    T pivot = arr[low];
    int i = low - 1;
    int j = high + 1;
    
    while (true) {
        // 从左边找到第一个大于等于基准的元素
        do {
            i++;
        } while (arr[i] < pivot);
        
        // 从右边找到第一个小于等于基准的元素
        do {
            j--;
        } while (arr[j] > pivot);
        
        // 如果指针相遇，返回分区点
        if (i >= j) {
            return j;
        }
        
        // 交换元素
        std::swap(arr[i], arr[j]);
    }
}

/**
 * 使用Hoare分区的快速排序
 */
template<typename T>
void quickSortHoare(std::vector<T>& arr, int low, int high) {
    if (low < high) {
        int pi = hoarePartition(arr, low, high);
        quickSortHoare(arr, low, pi);
        quickSortHoare(arr, pi + 1, high);
    }
}

template<typename T>
void quickSortHoare(std::vector<T>& arr) {
    if (!arr.empty()) {
        quickSortHoare(arr, 0, arr.size() - 1);
    }
}

// ==================== 可选：随机化快速排序（避免最坏情况） ====================

#include <cstdlib>
#include <ctime>

/**
 * 随机化分区（避免已排序数组的最坏情况）
 */
template<typename T>
int randomizedPartition(std::vector<T>& arr, int low, int high) {
    // 随机选择基准
    int randomIndex = low + rand() % (high - low + 1);
    std::swap(arr[randomIndex], arr[high]);
    
    return partition(arr, low, high);
}

/**
 * 随机化快速排序
 */
template<typename T>
void randomizedQuickSort(std::vector<T>& arr, int low, int high) {
    if (low < high) {
        int pi = randomizedPartition(arr, low, high);
        randomizedQuickSort(arr, low, pi - 1);
        randomizedQuickSort(arr, pi + 1, high);
    }
}

template<typename T>
void randomizedQuickSort(std::vector<T>& arr) {
    if (!arr.empty()) {
        randomizedQuickSort(arr, 0, arr.size() - 1);
    }
}

// ==================== 辅助函数 ====================

/**
 * 打印数组
 */
template<typename T>
void printArray(const std::vector<T>& arr) {
    for (const auto& elem : arr) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;
}

// ==================== 使用示例 ====================

int main() {
    // 初始化随机种子（用于随机化快速排序）
    std::srand(std::time(nullptr));
    
    std::cout << "=== C++ 快速排序示例 ===\n" << std::endl;
    
    // 示例1: 基本使用
    {
        std::vector<int> arr = {64, 34, 25, 12, 22, 11, 90};
        std::cout << "示例1 - 基本快速排序:" << std::endl;
        std::cout << "排序前: ";
        printArray(arr);
        
        quickSort(arr);
        std::cout << "排序后: ";
        printArray(arr);
        std::cout << std::endl;
    }
    
    // 示例2: 浮点数排序
    {
        std::vector<double> arr = {3.14, 2.71, 1.41, 1.73, 0.0, -1.0};
        std::cout << "示例2 - 浮点数排序:" << std::endl;
        std::cout << "排序前: ";
        printArray(arr);
        
        quickSort(arr);
        std::cout << "排序后: ";
        printArray(arr);
        std::cout << std::endl;
    }
    
    // 示例3: 使用Hoare分区
    {
        std::vector<int> arr = {5, 2, 9, 1, 5, 6};
        std::cout << "示例3 - Hoare分区快速排序:" << std::endl;
        std::cout << "排序前: ";
        printArray(arr);
        
        quickSortHoare(arr);
        std::cout << "排序后: ";
        printArray(arr);
        std::cout << std::endl;
    }
    
    // 示例4: 随机化快速排序
    {
        std::vector<int> arr = {9, 7, 5, 11, 12, 2, 14, 3, 10, 6};
        std::cout << "示例4 - 随机化快速排序:" << std::endl;
        std::cout << "排序前: ";
        printArray(arr);
        
        randomizedQuickSort(arr);
        std::cout << "排序后: ";
        printArray(arr);
        std::cout << std::endl;
    }
    
    // 示例5: 处理边界情况
    {
        std::vector<int> emptyArr;
        std::vector<int> singleArr = {42};
        std::vector<int> sortedArr = {1, 2, 3, 4, 5};
        
        std::cout << "示例5 - 边界情况测试:" << std::endl;
        
        quickSort(emptyArr);
        std::cout << "空数组排序后大小: " << emptyArr.size() << std::endl;
        
        quickSort(singleArr);
        std::cout << "单元素数组排序后: ";
        printArray(singleArr);
        
        quickSort(sortedArr);
        std::cout << "已排序数组排序后: ";
        printArray(sortedArr);
    }
    
    std::cout << "\n=== 示例结束 ===" << std::endl;
    
    return 0;
}